"""GUI front-end for the 4pi cone-beam projection generator.

Preview the STL, tune cone-beam geometry (magnification, rotation center, detector
resolution), then batch-generate a projection dataset ready for ASTRA reconstruction.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

import matplotlib
import numpy as np
import trimesh
from PySide6 import QtCore, QtWidgets

import geometry as geom
import io_utils
import raytrace_gpu as rt
import shape_metrics
from mesh_viewer import MeshViewer3D
from preview_gallery import PreviewGallery
from theme import FLAT_QSS

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

RIBBON_SECTIONS = ["输入 / 输出", "旋转中心", "锥束几何", "材料", "采样与设备", "样品评估", "序列生成"]

# preview_direction_combo index -> world-space direction (rotation_center -> source), unit vector.
# index 0 ("采样方案第1个视角") is handled separately by pulling from the sampling scheme.
AXIS_PREVIEW_DIRECTIONS = {
    1: np.array([1.0, 0.0, 0.0]),
    2: np.array([-1.0, 0.0, 0.0]),
    3: np.array([0.0, 1.0, 0.0]),
    4: np.array([0.0, -1.0, 0.0]),
    5: np.array([0.0, 0.0, 1.0]),
    6: np.array([0.0, 0.0, -1.0]),
}


def available_devices() -> list[str]:
    import warp as wp

    wp.init()
    devices = ["cpu"]
    try:
        if wp.is_cuda_available():
            for d in wp.get_cuda_devices():
                devices.append(str(d.alias))
    except Exception:
        pass
    return devices


class SectionDialog(QtWidgets.QDialog):
    """A settings section popped out into its own floating, non-modal window."""

    closed = QtCore.Signal()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class ScanThread(QtCore.QThread):
    progress = QtCore.Signal(int, int)
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            sinogram = rt.generate_sinogram(
                p["vertices"],
                p["faces"],
                p["views"],
                p["rows"],
                p["cols"],
                p["pixel_pitch"],
                p["mu"],
                p["device"],
                chunk_size=p["chunk_size"],
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
            astra_vectors = geom.to_astra_cone_vec(p["views"], p["rotation_center"], p["pixel_pitch"])
            meta = {
                "stl_path": p["stl_path"],
                "units": "same as STL file (assumed mm)",
                "sampling_scheme": p["scheme_name"],
                "n_views": len(p["views"]),
                "sod_mm": p["sod"],
                "sdd_mm": p["sdd"],
                "magnification": p["sdd"] / p["sod"],
                "pixel_pitch_mm": p["pixel_pitch"],
                "detector_rows": p["rows"],
                "detector_cols": p["cols"],
                "mu_per_mm": p["mu"],
                "rotation_center_mm": list(p["rotation_center"]),
                "device_used": p["device"],
            }
            io_utils.save_dataset(p["out_dir"], sinogram, astra_vectors, meta)
            self.done.emit(p["out_dir"])
        except Exception:
            self.failed.emit(traceback.format_exc())


class SequenceThread(QtCore.QThread):
    """Runs the full generation pipeline once per N value in a sweep, unattended.

    Each N gets its own subfolder (out_dir/N_00100/...) so datasets never collide; one
    failing N is recorded and skipped rather than aborting the whole sweep, and a single
    compact sequence_summary.json is written at the end for offline review.
    """

    outer_progress = QtCore.Signal(int, int, int)  # index, total, current_N
    inner_progress = QtCore.Signal(int, int)  # views done, views total (current N)
    n_done = QtCore.Signal(int, str, str)  # N, status, folder
    all_done = QtCore.Signal(str)  # summary_path
    failed = QtCore.Signal(str)

    def __init__(self, params: dict, n_values: list[int], vertices, faces, stl_path, skip_existing: bool):
        super().__init__()
        self.p = params
        self.n_values = n_values
        self.vertices = vertices
        self.faces = faces
        self.stl_path = stl_path
        self.skip_existing = skip_existing
        self._cancel = False

    def request_cancel(self):
        self._cancel = True

    def run(self):
        try:
            base_out = self.p["out_dir"]
            os.makedirs(base_out, exist_ok=True)
            summary = []
            total = len(self.n_values)

            for i, n in enumerate(self.n_values):
                if self._cancel:
                    summary.append({"n_views": n, "status": "cancelled"})
                    continue

                self.outer_progress.emit(i, total, n)
                sub_dir = os.path.join(base_out, f"N_{n:05d}")

                if self.skip_existing and os.path.isfile(os.path.join(sub_dir, "projections.npy")):
                    self.n_done.emit(n, "skipped_existing", sub_dir)
                    summary.append({"n_views": n, "status": "skipped_existing", "folder": sub_dir})
                    continue

                t0 = time.time()
                try:
                    scheme_fn = geom.SAMPLING_SCHEMES[self.p["scheme_name"]]
                    directions = scheme_fn(n)
                    views = geom.build_all_views(directions, self.p["rotation_center"], self.p["sod"], self.p["sdd"])
                    chunk_size = max(1, min(25, len(views)))

                    sinogram = rt.generate_sinogram(
                        self.vertices,
                        self.faces,
                        views,
                        self.p["rows"],
                        self.p["cols"],
                        self.p["pixel_pitch"],
                        self.p["mu"],
                        self.p["device"],
                        chunk_size=chunk_size,
                        progress_cb=lambda done, tot: self.inner_progress.emit(done, tot),
                    )
                    astra_vectors = geom.to_astra_cone_vec(views, self.p["rotation_center"], self.p["pixel_pitch"])
                    meta = {
                        "stl_path": self.stl_path,
                        "units": "same as STL file (assumed mm)",
                        "sampling_scheme": self.p["scheme_name"],
                        "n_views": n,
                        "sod_mm": self.p["sod"],
                        "sdd_mm": self.p["sdd"],
                        "magnification": self.p["sdd"] / self.p["sod"],
                        "pixel_pitch_mm": self.p["pixel_pitch"],
                        "detector_rows": self.p["rows"],
                        "detector_cols": self.p["cols"],
                        "mu_per_mm": self.p["mu"],
                        "rotation_center_mm": list(self.p["rotation_center"]),
                        "device_used": self.p["device"],
                    }
                    io_utils.save_dataset(sub_dir, sinogram, astra_vectors, meta)
                    elapsed = time.time() - t0
                    self.n_done.emit(n, "ok", sub_dir)
                    summary.append({"n_views": n, "status": "ok", "folder": sub_dir, "elapsed_s": round(elapsed, 2)})
                except Exception:
                    err = traceback.format_exc()
                    self.n_done.emit(n, "failed", sub_dir)
                    summary.append({"n_views": n, "status": "failed", "folder": sub_dir, "error": err})

            summary_path = os.path.join(base_out, "sequence_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": summary},
                    f, indent=2, ensure_ascii=False, default=str,
                )
            self.all_done.emit(summary_path)
        except Exception:
            self.failed.emit(traceback.format_exc())


class ShapeMetricsThread(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    done = QtCore.Signal(dict, str)
    failed = QtCore.Signal(str)

    def __init__(self, mesh, selected_ranks, n_views, voxel_target, percentile, out_dir, gen_params):
        super().__init__()
        self.mesh = mesh
        self.selected_ranks = selected_ranks
        self.n_views = n_views
        self.voxel_target = voxel_target
        self.percentile = percentile
        self.out_dir = out_dir
        self.gen_params = gen_params

    def run(self):
        try:
            results = shape_metrics.compute_shape_metrics(
                self.mesh,
                self.selected_ranks,
                n_views=self.n_views,
                voxel_target=self.voxel_target,
                percentile=self.percentile,
                progress_cb=lambda done, total, label: self.progress.emit(done, total, label),
            )
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, "shape_metrics.json")
            payload = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generation_parameters": self.gen_params,
                "shape_metrics": results,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            self.done.emit(results, out_path)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT 4π Cone-Beam Projection Generator")
        self.resize(1440, 900)

        self.mesh: trimesh.Trimesh | None = None
        self.stl_path: str | None = None
        self.scan_thread: ScanThread | None = None
        self.seq_thread: SequenceThread | None = None
        self._last_directions = None

        self._build_ui()
        self.statusBar().showMessage("就绪")

    # ---------------------------------------------------------- UI layout
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_ribbon_bar())
        root.addWidget(self._build_action_row())

        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.viewer = MeshViewer3D()
        self.gallery = PreviewGallery()
        top_splitter.addWidget(self.viewer)
        top_splitter.addWidget(self.gallery)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setSizes([1000, 360])

        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)

        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.log_box)
        main_splitter.setStretchFactor(0, 5)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([760, 140])

        root.addWidget(main_splitter, 1)

        self._build_section_dialogs()

    def _build_ribbon_bar(self) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        bar.setObjectName("ribbonBar")
        layout = QtWidgets.QHBoxLayout(bar)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(0)

        self.ribbon_buttons: list[QtWidgets.QToolButton] = []
        for i, name in enumerate(RIBBON_SECTIONS):
            btn = QtWidgets.QToolButton()
            btn.setObjectName("ribbonTab")
            btn.setText(name)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, idx=i: self._toggle_section_dialog(idx))
            layout.addWidget(btn)
            self.ribbon_buttons.append(btn)
        layout.addStretch(1)
        return bar

    def _build_section_dialogs(self):
        sections = [
            ("输入 / 输出", self._build_io_panel),
            ("旋转中心", self._build_rotation_center_panel),
            ("锥束几何", self._build_geometry_panel),
            ("材料", self._build_material_panel),
            ("采样与设备", self._build_sampling_device_panel),
            ("样品评估", self._build_shape_eval_panel),
            ("序列生成", self._build_sequence_panel),
        ]
        self.section_dialogs: list[SectionDialog] = []
        for i, (title, builder) in enumerate(sections):
            dialog = SectionDialog(self)
            dialog.setWindowTitle(title)
            layout = QtWidgets.QVBoxLayout(dialog)
            layout.addWidget(builder())
            dialog.closed.connect(lambda idx=i: self.ribbon_buttons[idx].setChecked(False))
            self.section_dialogs.append(dialog)

    def _build_action_row(self) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        row.setObjectName("actionRow")
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(16, 8, 16, 8)

        self.preview_btn = QtWidgets.QPushButton("预览单张投影")
        self.preview_btn.clicked.connect(self.on_preview_projection)
        layout.addWidget(self.preview_btn)

        layout.addWidget(QtWidgets.QLabel("预览角度:"))
        self.preview_direction_combo = QtWidgets.QComboBox()
        self.preview_direction_combo.addItems(
            ["采样方案第1个视角", "沿 +X 看", "沿 -X 看", "沿 +Y 看", "沿 -Y 看", "沿 +Z 看", "沿 -Z 看"]
        )
        layout.addWidget(self.preview_direction_combo)

        self.run_btn = QtWidgets.QPushButton("生成全部 4π 投影")
        self.run_btn.setObjectName("primaryAction")
        self.run_btn.clicked.connect(self.on_run_full_scan)
        layout.addWidget(self.run_btn)

        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar, 1)

        return row

    # -------------------------------------------------------- ribbon panels
    def _build_io_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)

        self.stl_path_edit = QtWidgets.QLineEdit()
        stl_btn = QtWidgets.QPushButton("浏览...")
        stl_btn.clicked.connect(self.on_browse_stl)
        stl_row = QtWidgets.QHBoxLayout()
        stl_row.addWidget(self.stl_path_edit)
        stl_row.addWidget(stl_btn)
        form.addRow("STL 文件:", self._wrap(stl_row))

        self.out_dir_edit = QtWidgets.QLineEdit()
        out_btn = QtWidgets.QPushButton("浏览...")
        out_btn.clicked.connect(self.on_browse_out_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.out_dir_edit)
        out_row.addWidget(out_btn)
        form.addRow("输出目录:", self._wrap(out_row))

        load_btn = QtWidgets.QPushButton("加载并预览模型")
        load_btn.clicked.connect(self.on_load_mesh)
        form.addRow(load_btn)

        self.mesh_info_label = QtWidgets.QLabel("尚未加载模型")
        self.mesh_info_label.setWordWrap(True)
        form.addRow("模型信息:", self.mesh_info_label)

        return panel

    def _build_rotation_center_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(panel)

        radio_col = QtWidgets.QVBoxLayout()
        self.center_mode_group = QtWidgets.QButtonGroup(self)
        self.rb_centroid = QtWidgets.QRadioButton("几何/质心 centroid (默认)")
        self.rb_bbox = QtWidgets.QRadioButton("包围盒中心 bounding-box center")
        self.rb_manual = QtWidgets.QRadioButton("手动输入")
        self.rb_centroid.setChecked(True)
        for i, rb in enumerate([self.rb_centroid, self.rb_bbox, self.rb_manual]):
            self.center_mode_group.addButton(rb, i)
            radio_col.addWidget(rb)
        self.center_mode_group.idClicked.connect(self.on_center_mode_changed)
        layout.addLayout(radio_col)

        form = QtWidgets.QFormLayout()
        self.center_x = self._make_spinbox(-1e5, 1e5, 0.0)
        self.center_y = self._make_spinbox(-1e5, 1e5, 0.0)
        self.center_z = self._make_spinbox(-1e5, 1e5, 0.0)
        for sb in (self.center_x, self.center_y, self.center_z):
            sb.setEnabled(False)
            sb.valueChanged.connect(self.on_center_value_edited)
        form.addRow("X (mm):", self.center_x)
        form.addRow("Y (mm):", self.center_y)
        form.addRow("Z (mm):", self.center_z)
        layout.addLayout(form)
        layout.addStretch(1)

        return panel

    def _build_geometry_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)

        self.sod_spin = self._make_spinbox(0.001, 1e6, 100.0)
        form.addRow("源-旋转中心距离 SOD (mm):", self.sod_spin)

        self.mag_spin = self._make_spinbox(1.0001, 1000.0, 2.0)
        form.addRow("放大倍率 M = SDD / SOD:", self.mag_spin)

        self.sdd_label = QtWidgets.QLabel()
        form.addRow("=> 源-探测器距离 SDD (mm):", self.sdd_label)
        self.sod_spin.valueChanged.connect(self.update_sdd_label)
        self.mag_spin.valueChanged.connect(self.update_sdd_label)
        self.update_sdd_label()

        self.pixel_pitch_spin = self._make_spinbox(1e-4, 1e4, 0.2)
        form.addRow("探测器像素物理尺寸 (mm/px):", self.pixel_pitch_spin)

        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(4, 8192)
        self.rows_spin.setValue(256)
        form.addRow("探测器分辨率 - 行 rows:", self.rows_spin)

        self.cols_spin = QtWidgets.QSpinBox()
        self.cols_spin.setRange(4, 8192)
        self.cols_spin.setValue(256)
        form.addRow("探测器分辨率 - 列 cols:", self.cols_spin)

        fit_btn = QtWidgets.QPushButton("自动适配像素尺寸 (按当前分辨率, 避免裁切)")
        fit_btn.clicked.connect(self.on_auto_fit_fov)
        form.addRow(fit_btn)

        self.fov_hint_label = QtWidgets.QLabel("")
        self.fov_hint_label.setWordWrap(True)
        self.fov_hint_label.setStyleSheet("color: #5b6472; font-size: 11px;")
        form.addRow(self.fov_hint_label)

        return panel

    def _build_material_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        self.mu_spin = self._make_spinbox(1e-6, 1e3, 0.05)
        self.mu_spin.setDecimals(6)
        form.addRow("线性吸收系数 mu (1/mm):", self.mu_spin)
        return panel

    def _build_sampling_device_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)

        self.scheme_combo = QtWidgets.QComboBox()
        self.scheme_combo.addItems(
            ["fibonacci_4pi", "random_4pi", "circular_z", "circular_y", "circular_x"]
        )
        form.addRow("采样方式:", self.scheme_combo)

        scheme_hint = QtWidgets.QLabel(
            "fibonacci/random_4pi: 全4π球面采样\n"
            "circular_x/y/z: 绕该轴旋转扫描 (source 在垂直于该轴的平面内环绕,\n"
            "不是沿该轴方向看 —— 想直接看某个轴向的视角, 用下方\"预览单张投影\"\n"
            "旁边的\"预览角度\"下拉框单独指定。"
        )
        scheme_hint.setWordWrap(True)
        scheme_hint.setStyleSheet("color: #5b6472; font-size: 11px;")
        form.addRow(scheme_hint)

        self.n_views_spin = QtWidgets.QSpinBox()
        self.n_views_spin.setRange(1, 200000)
        self.n_views_spin.setValue(100)
        form.addRow("投影角度数 N:", self.n_views_spin)

        self.device_combo = QtWidgets.QComboBox()
        try:
            self.device_combo.addItems(available_devices())
        except Exception:
            self.device_combo.addItems(["cpu"])
        if self.device_combo.count() > 1:
            self.device_combo.setCurrentIndex(self.device_combo.count() - 1)
        form.addRow("Warp device:", self.device_combo)

        return panel

    def _build_shape_eval_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        hint = QtWidgets.QLabel(
            "样品形态参数评估(用于研究\"4π采样密度\"和\"样品复杂度\"的关系)。\n"
            "勾选需要计算的参数, 点\"计算并导出\"后连同当前的生成参数一起写入\n"
            "输出目录下的 shape_metrics.json。定义/算法来源见 shape_metrics.py 顶部注释。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5b6472; font-size: 11px;")
        layout.addWidget(hint)

        grid = QtWidgets.QGridLayout()
        self.metric_checkboxes: dict[int, QtWidgets.QCheckBox] = {}
        for i, m in enumerate(shape_metrics.METRIC_ORDER):
            cb = QtWidgets.QCheckBox(f"{m['rank']}. {m['label']}")
            cb.setChecked(m["default"])
            self.metric_checkboxes[m["rank"]] = cb
            grid.addWidget(cb, i // 2, i % 2)
        layout.addLayout(grid)

        adv_form = QtWidgets.QFormLayout()
        self.voxel_target_spin = QtWidgets.QSpinBox()
        self.voxel_target_spin.setRange(16, 256)
        self.voxel_target_spin.setValue(64)
        adv_form.addRow("体素分辨率 (最长边体素数, 用于 T5/fractal dimension):", self.voxel_target_spin)

        self.percentile_spin = self._make_spinbox(0.1, 50.0, 5.0)
        adv_form.addRow("Local thickness 百分位数:", self.percentile_spin)
        layout.addLayout(adv_form)

        btn_row = QtWidgets.QHBoxLayout()
        self.compute_metrics_btn = QtWidgets.QPushButton("计算并导出形态参数")
        self.compute_metrics_btn.clicked.connect(self.on_compute_shape_metrics)
        btn_row.addWidget(self.compute_metrics_btn)
        self.metrics_progress = QtWidgets.QProgressBar()
        btn_row.addWidget(self.metrics_progress, 1)
        layout.addLayout(btn_row)

        self.metrics_result_box = QtWidgets.QPlainTextEdit()
        self.metrics_result_box.setReadOnly(True)
        self.metrics_result_box.setMinimumHeight(220)
        layout.addWidget(self.metrics_result_box)

        return panel

    def _build_sequence_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        hint = QtWidgets.QLabel(
            "批量生成一系列不同\"投影角度数 N\"的数据集(比如做采样密度扫描实验),\n"
            "不用每次手动改N再点一次生成。填多段(起始N, 结束N, 步长), 会自动合并\n"
            "去重成一串N值; 每个N单独生成到 out_dir/N_00100/ 这样的子文件夹, 互不覆盖。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #5b6472; font-size: 11px;")
        layout.addWidget(hint)

        self.seq_table = QtWidgets.QTableWidget(0, 3)
        self.seq_table.setHorizontalHeaderLabels(["起始 N", "结束 N", "步长"])
        self.seq_table.horizontalHeader().setStretchLastSection(True)
        self.seq_table.setMaximumHeight(160)
        layout.addWidget(self.seq_table)
        self._add_seq_row(100, 1000, 100)
        self._add_seq_row(1000, 3000, 200)

        row_btns = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("+ 添加一段")
        add_btn.clicked.connect(lambda: self._add_seq_row(100, 200, 100))
        del_btn = QtWidgets.QPushButton("- 删除选中段")
        del_btn.clicked.connect(self._remove_seq_row)
        preview_btn = QtWidgets.QPushButton("预览 N 列表")
        preview_btn.clicked.connect(self.on_preview_n_sequence)
        row_btns.addWidget(add_btn)
        row_btns.addWidget(del_btn)
        row_btns.addWidget(preview_btn)
        layout.addLayout(row_btns)

        self.seq_preview_label = QtWidgets.QLabel("")
        self.seq_preview_label.setWordWrap(True)
        layout.addWidget(self.seq_preview_label)

        self.seq_skip_existing = QtWidgets.QCheckBox("跳过已生成过的N (按子文件夹里是否已有 projections.npy 判断, 方便中断后续跑)")
        self.seq_skip_existing.setChecked(True)
        layout.addWidget(self.seq_skip_existing)

        run_row = QtWidgets.QHBoxLayout()
        self.seq_run_btn = QtWidgets.QPushButton("开始批量生成序列")
        self.seq_run_btn.setObjectName("primaryAction")
        self.seq_run_btn.clicked.connect(self.on_run_sequence)
        self.seq_stop_btn = QtWidgets.QPushButton("停止")
        self.seq_stop_btn.setEnabled(False)
        self.seq_stop_btn.clicked.connect(self.on_stop_sequence)
        run_row.addWidget(self.seq_run_btn)
        run_row.addWidget(self.seq_stop_btn)
        layout.addLayout(run_row)

        prog_form = QtWidgets.QFormLayout()
        self.seq_outer_progress = QtWidgets.QProgressBar()
        prog_form.addRow("总进度 (第几个 N):", self.seq_outer_progress)
        self.seq_inner_progress = QtWidgets.QProgressBar()
        prog_form.addRow("当前 N 内部进度:", self.seq_inner_progress)
        layout.addLayout(prog_form)

        return panel

    def _add_seq_row(self, start: int, end: int, step: int):
        row = self.seq_table.rowCount()
        self.seq_table.insertRow(row)
        for col, val in enumerate([start, end, step]):
            item = QtWidgets.QTableWidgetItem(str(val))
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.seq_table.setItem(row, col, item)

    def _remove_seq_row(self):
        row = self.seq_table.currentRow()
        if row >= 0:
            self.seq_table.removeRow(row)

    def _parse_seq_table(self) -> list[int]:
        segments = []
        for row in range(self.seq_table.rowCount()):
            try:
                start = int(self.seq_table.item(row, 0).text())
                end = int(self.seq_table.item(row, 1).text())
                step = int(self.seq_table.item(row, 2).text())
            except (AttributeError, ValueError):
                raise ValueError(f"第 {row + 1} 行填的不是有效整数")
            segments.append((start, end, step))
        if not segments:
            raise ValueError("请至少填一段 (起始N, 结束N, 步长)")
        return geom.parse_n_sequence(segments)

    @staticmethod
    def _wrap(inner_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(inner_layout)
        return w

    @staticmethod
    def _make_spinbox(lo: float, hi: float, value: float) -> QtWidgets.QDoubleSpinBox:
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(4)
        sb.setValue(value)
        return sb

    # -------------------------------------------------------------- ribbon
    def _toggle_section_dialog(self, idx: int):
        dialog = self.section_dialogs[idx]
        if dialog.isVisible():
            dialog.hide()
            self.ribbon_buttons[idx].setChecked(False)
        else:
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.ribbon_buttons[idx].setChecked(True)

    # -------------------------------------------------------------- slots
    def log(self, msg: str):
        self.log_box.appendPlainText(msg)
        self.statusBar().showMessage(msg, 6000)

    def update_sdd_label(self):
        sod = self.sod_spin.value()
        mag = self.mag_spin.value()
        self.sdd_label.setText(f"{sod * mag:.4f}")

    def on_auto_fit_fov(self):
        if self.mesh is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载 STL 模型")
            return

        rotation_center = np.array([self.center_x.value(), self.center_y.value(), self.center_z.value()])
        sod = self.sod_spin.value()
        sdd = sod * self.mag_spin.value()

        radius = float(np.max(np.linalg.norm(self.mesh.vertices - rotation_center, axis=1)))
        if radius >= sod:
            QtWidgets.QMessageBox.warning(
                self,
                "几何无效",
                f"模型上离旋转中心最远的点 ({radius:.3f} mm) 已经超出了 SOD ({sod:.3f} mm)，"
                "源会跑到模型内部/背面去。请先增大 SOD 或检查旋转中心是否选对了。",
            )
            return

        margin = 1.1  # 10% headroom so the silhouette doesn't sit exactly on the detector edge
        half_angle = np.arcsin(radius / sod)
        required_half_extent = sdd * np.tan(half_angle) * margin

        rows, cols = self.rows_spin.value(), self.cols_spin.value()
        new_pitch = 2.0 * required_half_extent / min(rows, cols)
        self.pixel_pitch_spin.setValue(new_pitch)

        self.fov_hint_label.setText(
            f"模型半径(到旋转中心) {radius:.3f} mm -> 探测器需要覆盖直径 >= {2 * required_half_extent / margin:.3f} mm "
            f"(已加10%余量) -> 像素尺寸设为 {new_pitch:.5f} mm/px"
        )
        self.log(f"自动适配像素尺寸: {new_pitch:.5f} mm/px (模型半径={radius:.3f}mm, SOD={sod:.3f}mm, SDD={sdd:.3f}mm)")

    def on_browse_stl(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择 STL 文件", "", "STL Files (*.stl)")
        if path:
            self.stl_path_edit.setText(path)

    def on_browse_out_dir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.out_dir_edit.setText(path)

    def on_center_mode_changed(self, idx: int):
        manual = idx == 2
        for sb in (self.center_x, self.center_y, self.center_z):
            sb.setEnabled(manual)
        if self.mesh is None:
            return
        if idx == 0:
            c = self.mesh.centroid
        elif idx == 1:
            c = self.mesh.bounds.mean(axis=0)
        else:
            return
        self.center_x.blockSignals(True)
        self.center_y.blockSignals(True)
        self.center_z.blockSignals(True)
        self.center_x.setValue(float(c[0]))
        self.center_y.setValue(float(c[1]))
        self.center_z.setValue(float(c[2]))
        self.center_x.blockSignals(False)
        self.center_y.blockSignals(False)
        self.center_z.blockSignals(False)
        self.viewer.set_rotation_center(c)

    def on_center_value_edited(self, _value: float):
        c = np.array([self.center_x.value(), self.center_y.value(), self.center_z.value()])
        self.viewer.set_rotation_center(c)

    def on_load_mesh(self):
        path = self.stl_path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "错误", "请先选择一个有效的 STL 文件")
            return
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return

        self.mesh = mesh
        self.stl_path = path

        watertight = mesh.is_watertight
        info = (
            f"顶点数: {len(mesh.vertices)}    面片数: {len(mesh.faces)}\n"
            f"水密 (watertight): {'是' if watertight else '否 - 光线求交结果可能不准确!'}\n"
            f"包围盒尺寸 (mm): {np.round(mesh.extents, 3).tolist()}\n"
            f"质心 centroid (mm): {np.round(mesh.centroid, 4).tolist()}"
        )
        self.mesh_info_label.setText(info)
        if not watertight:
            self.log("警告: STL 非水密, 路径长度积分可能出错. 建议在 CAD 里检查实体导出设置.")

        self.viewer.set_mesh(mesh, mesh.centroid)
        self.on_center_mode_changed(self.center_mode_group.checkedId())
        self.log(f"已加载模型: {path}")

    def _gather_common_params(self):
        if self.mesh is None:
            raise ValueError("请先加载 STL 模型")
        out_dir = self.out_dir_edit.text().strip()
        if not out_dir:
            raise ValueError("请先选择输出目录")

        rotation_center = np.array([self.center_x.value(), self.center_y.value(), self.center_z.value()])
        sod = self.sod_spin.value()
        sdd = sod * self.mag_spin.value()
        pixel_pitch = self.pixel_pitch_spin.value()
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        mu = self.mu_spin.value()
        device = self.device_combo.currentText()
        scheme_name = self.scheme_combo.currentText()
        n_views = self.n_views_spin.value()

        return dict(
            out_dir=out_dir,
            rotation_center=rotation_center,
            sod=sod,
            sdd=sdd,
            pixel_pitch=pixel_pitch,
            rows=rows,
            cols=cols,
            mu=mu,
            device=device,
            scheme_name=scheme_name,
            n_views=n_views,
        )

    def _build_views(self, p: dict):
        scheme_fn = geom.SAMPLING_SCHEMES[p["scheme_name"]]
        directions = scheme_fn(p["n_views"])
        self._last_directions = directions
        return geom.build_all_views(directions, p["rotation_center"], p["sod"], p["sdd"])

    def on_preview_projection(self):
        try:
            p = self._gather_common_params()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        idx = self.preview_direction_combo.currentIndex()
        try:
            if idx == 0:
                views = self._build_views(p)
                direction = self._last_directions[0]
                scheme_label = p["scheme_name"]
                view = views[0]
            else:
                direction = AXIS_PREVIEW_DIRECTIONS[idx]
                scheme_label = self.preview_direction_combo.currentText()
                view = geom.build_view_geometry(direction, p["rotation_center"], p["sod"], p["sdd"])

            projector = rt.GpuProjector(self.mesh.vertices, self.mesh.faces, p["device"])
            path = projector.trace_views([view], p["rows"], p["cols"], p["pixel_pitch"])
            image = path[0] * p["mu"]
        except Exception:
            QtWidgets.QMessageBox.critical(self, "预览失败", traceback.format_exc())
            return

        meta = dict(
            scheme_name=scheme_label,
            view_index=0,
            direction=np.round(direction, 4).tolist(),
            sod=p["sod"],
            sdd=p["sdd"],
            magnification=p["sdd"] / p["sod"],
            pixel_pitch=p["pixel_pitch"],
            mu=p["mu"],
            rows=p["rows"],
            cols=p["cols"],
        )
        self.gallery.add_preview(image, meta)
        self.log(f"生成单张投影预览 (方案: {scheme_label}, 方向: {meta['direction']})")

    def on_compute_shape_metrics(self):
        if self.mesh is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载 STL 模型")
            return
        try:
            p = self._gather_common_params()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        selected = {rank for rank, cb in self.metric_checkboxes.items() if cb.isChecked()}
        if not selected:
            QtWidgets.QMessageBox.warning(self, "错误", "请至少勾选一个参数")
            return

        gen_params = dict(p)
        gen_params["rotation_center"] = gen_params["rotation_center"].tolist()
        gen_params["stl_path"] = self.stl_path

        self.compute_metrics_btn.setEnabled(False)
        self.metrics_progress.setRange(0, len(selected))
        self.metrics_progress.setValue(0)
        self.metrics_result_box.setPlainText("计算中...")

        self.metrics_thread = ShapeMetricsThread(
            self.mesh,
            selected,
            p["n_views"],
            self.voxel_target_spin.value(),
            self.percentile_spin.value(),
            p["out_dir"],
            gen_params,
        )
        self.metrics_thread.progress.connect(self.on_metrics_progress)
        self.metrics_thread.done.connect(self.on_metrics_done)
        self.metrics_thread.failed.connect(self.on_metrics_failed)
        self.metrics_thread.start()

    def on_metrics_progress(self, done: int, total: int, label: str):
        self.metrics_progress.setValue(done)
        self.log(f"计算形态参数: {label} ({done}/{total})")

    def on_metrics_done(self, results: dict, out_path: str):
        self.compute_metrics_btn.setEnabled(True)
        self.metrics_result_box.setPlainText(json.dumps(results, indent=2, ensure_ascii=False, default=str))
        self.log(f"形态参数计算完成, 已导出到: {out_path}")

    def on_metrics_failed(self, err: str):
        self.compute_metrics_btn.setEnabled(True)
        self.metrics_result_box.setPlainText("计算失败:\n" + err)
        self.log("形态参数计算失败:\n" + err)
        QtWidgets.QMessageBox.critical(self, "计算失败", err)

    def on_run_full_scan(self):
        try:
            p = self._gather_common_params()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        views = self._build_views(p)
        params = dict(p)
        params["views"] = views
        params["vertices"] = self.mesh.vertices
        params["faces"] = self.mesh.faces
        params["stl_path"] = self.stl_path
        params["chunk_size"] = max(1, min(25, len(views)))

        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.progress_bar.setRange(0, len(views))
        self.progress_bar.setValue(0)
        self.log(f"开始生成 {len(views)} 个投影 (方案: {p['scheme_name']}, 设备: {p['device']})...")

        self.scan_thread = ScanThread(params)
        self.scan_thread.progress.connect(self.on_scan_progress)
        self.scan_thread.done.connect(self.on_scan_done)
        self.scan_thread.failed.connect(self.on_scan_failed)
        self.scan_thread.start()

    def on_scan_progress(self, done: int, total: int):
        self.progress_bar.setValue(done)

    def on_scan_done(self, out_dir: str):
        self.log(f"完成! 数据集已保存到: {out_dir}")
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)

    def on_scan_failed(self, err: str):
        self.log("生成失败:\n" + err)
        QtWidgets.QMessageBox.critical(self, "生成失败", err)
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)

    def on_preview_n_sequence(self):
        try:
            values = self._parse_seq_table()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return
        self.seq_preview_label.setText(f"共 {len(values)} 个 N 值: {values}")

    def on_run_sequence(self):
        if self.mesh is None:
            QtWidgets.QMessageBox.warning(self, "错误", "请先加载 STL 模型")
            return
        try:
            p = self._gather_common_params()
            n_values = self._parse_seq_table()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "参数错误", str(exc))
            return

        self.seq_run_btn.setEnabled(False)
        self.seq_stop_btn.setEnabled(True)
        self.run_btn.setEnabled(False)
        self.preview_btn.setEnabled(False)
        self.seq_outer_progress.setRange(0, len(n_values))
        self.seq_outer_progress.setValue(0)
        self.seq_inner_progress.setValue(0)
        self.log(f"开始批量序列生成: {len(n_values)} 个 N 值 {n_values}")

        self.seq_thread = SequenceThread(
            p, n_values, self.mesh.vertices, self.mesh.faces, self.stl_path,
            self.seq_skip_existing.isChecked(),
        )
        self.seq_thread.outer_progress.connect(self.on_seq_outer_progress)
        self.seq_thread.inner_progress.connect(self.on_seq_inner_progress)
        self.seq_thread.n_done.connect(self.on_seq_n_done)
        self.seq_thread.all_done.connect(self.on_seq_all_done)
        self.seq_thread.failed.connect(self.on_seq_failed)
        self.seq_thread.start()

    def on_stop_sequence(self):
        if self.seq_thread is not None:
            self.seq_thread.request_cancel()
            self.log("已请求停止序列生成 (当前 N 跑完后停止, 不会中断到一半)")
            self.seq_stop_btn.setEnabled(False)

    def on_seq_outer_progress(self, index: int, total: int, n: int):
        self.seq_outer_progress.setValue(index)
        self.seq_inner_progress.setValue(0)
        self.log(f"序列进度 {index + 1}/{total}: 开始生成 N={n}")

    def on_seq_inner_progress(self, done: int, total: int):
        self.seq_inner_progress.setRange(0, total)
        self.seq_inner_progress.setValue(done)

    def on_seq_n_done(self, n: int, status: str, folder: str):
        self.log(f"N={n} -> {status} ({folder})")

    def on_seq_all_done(self, summary_path: str):
        self.seq_outer_progress.setValue(self.seq_outer_progress.maximum())
        self.seq_run_btn.setEnabled(True)
        self.seq_stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.log(f"序列生成全部完成, 汇总见: {summary_path}")

    def on_seq_failed(self, err: str):
        self.seq_run_btn.setEnabled(True)
        self.seq_stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        self.preview_btn.setEnabled(True)
        self.log("序列生成失败:\n" + err)
        QtWidgets.QMessageBox.critical(self, "序列生成失败", err)


def run_selftest() -> int:
    """Packaged-exe smoke test: actually exercise the GPU/Warp raytracing pipeline,
    not just "the window opened". Used by build_exe.py right after packaging.
    """
    import tempfile

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(FLAT_QSS)
    win = MainWindow()
    try:
        mesh = trimesh.creation.box(extents=[10.0, 8.0, 6.0])
        stl_path = os.path.join(tempfile.gettempdir(), "ct4pi_selftest.stl")
        mesh.export(stl_path)

        win.stl_path_edit.setText(stl_path)
        win.out_dir_edit.setText(os.path.join(tempfile.gettempdir(), "ct4pi_selftest_out"))
        win.on_load_mesh()
        if win.mesh is None:
            raise RuntimeError("mesh failed to load")

        win.rows_spin.setValue(16)
        win.cols_spin.setValue(16)
        win.on_preview_projection()
        if win.gallery._count != 1:
            raise RuntimeError(f"expected 1 preview thumbnail, got {win.gallery._count}")

        device = win.device_combo.currentText()
        print(f"selftest ok: mesh loaded, GPU/Warp preview generated on device={device}")
        return 0
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        win.close()


def main():
    if "--selftest" in sys.argv[1:]:
        sys.exit(run_selftest())

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(FLAT_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
