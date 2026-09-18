"""Standalone CT reconstruction viewer.

Load an already-reconstructed volume (.npy) directly, or load a projection dataset
(projections.npy + astra_vectors.npy + metadata.json, as exported by projector_gui.py's
generation/sequence/batch-experiment features) and run an ASTRA reconstruction on it --
then inspect the result as an interactive 3D isosurface (rotate/zoom, adjustable
threshold) plus scrubbable orthogonal slices for interior detail, and (optionally)
compare it against a reference STL via SSIM.

Note: the SSIM comparison needs GPU ground-truth occupancy testing, which needs
warp-lang -- so this exe is no longer the "no Warp" lightweight build it started as.
Everything else (reconstruction itself, isosurface extraction, slice viewing) still
works without ever touching Warp if you don't load a reference STL.
"""

from __future__ import annotations

import os
import sys

# See projector_gui.py for why: a --windowed PyInstaller build has sys.stdout/stderr
# set to None (no console), which crashes any library that unconditionally prints.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import json
import time
import traceback

import matplotlib
import numpy as np
import trimesh
from PySide6 import QtCore, QtWidgets
from skimage.filters import threshold_otsu

import reconstruction as recon
from mesh_viewer import MeshViewer3D
from theme import FLAT_QSS
from volume_view import SliceViewer, extract_isosurface

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

THRESHOLD_STEPS = 1000


def available_devices() -> list[str]:
    try:
        import warp as wp

        wp.init()
        devices = ["cpu"]
        if wp.is_cuda_available():
            for d in wp.get_cuda_devices():
                devices.append(str(d.alias))
        return devices
    except Exception:
        return ["cpu"]


class ReconstructThread(QtCore.QThread):
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, dataset_dir: str, algorithm: str, iterations: int, n_voxels: int, vol_half: float):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.algorithm = algorithm
        self.iterations = iterations
        self.n_voxels = n_voxels
        self.vol_half = vol_half

    def run(self):
        try:
            sino = np.load(os.path.join(self.dataset_dir, "projections.npy"))
            astra_vectors = np.load(os.path.join(self.dataset_dir, "astra_vectors.npy"))
            with open(os.path.join(self.dataset_dir, "metadata.json"), encoding="utf-8") as f:
                meta = json.load(f)
            rows, cols = meta["detector_rows"], meta["detector_cols"]

            t0 = time.time()
            rec = recon.reconstruct(
                sino, astra_vectors, rows, cols, self.vol_half, self.n_voxels,
                algorithm=self.algorithm, iterations=self.iterations,
            )
            elapsed = time.time() - t0

            try:
                import astra
                gpu_info = astra.get_gpu_info()
            except Exception:
                gpu_info = "unknown"

            self.done.emit({
                "volume": rec,
                "elapsed_s": elapsed,
                "n_views": int(sino.shape[0]),
                "rows": rows,
                "cols": cols,
                "mu": meta.get("mu_per_mm"),
                "rotation_center": meta.get("rotation_center_mm"),
                "stl_path": meta.get("stl_path"),
                "algorithm": self.algorithm,
                "iterations": self.iterations,
                "n_voxels": self.n_voxels,
                "vol_half": self.vol_half,
                "gpu_info": gpu_info,
                "dataset_dir": self.dataset_dir,
            })
        except Exception:
            self.failed.emit(traceback.format_exc())


class SSIMThread(QtCore.QThread):
    done = QtCore.Signal(float, float)  # ssim, elapsed_s
    failed = QtCore.Signal(str)

    def __init__(self, ref_vertices, ref_faces, volume, vol_half, n_voxels, rotation_center, mu, device):
        super().__init__()
        self.ref_vertices = ref_vertices
        self.ref_faces = ref_faces
        self.volume = volume
        self.vol_half = vol_half
        self.n_voxels = n_voxels
        self.rotation_center = rotation_center
        self.mu = mu
        self.device = device

    def run(self):
        try:
            t0 = time.time()
            ground_truth = recon.voxelize_ground_truth(
                self.ref_vertices, self.ref_faces, self.vol_half, self.n_voxels,
                self.rotation_center, self.device,
            )
            score = recon.compute_ssim(ground_truth, self.volume, self.mu)
            elapsed = time.time() - t0
            self.done.emit(float(score), elapsed)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Reconstruction Viewer")
        self.resize(1520, 900)
        self.volume: np.ndarray | None = None
        self.voxel_pitch: float = 1.0
        self.recon_info: dict = {}
        self.reference_mesh: trimesh.Trimesh | None = None
        self.reference_path: str | None = None
        self.recon_thread: ReconstructThread | None = None
        self.ssim_thread: SSIMThread | None = None
        self._build_ui()
        self.statusBar().showMessage("就绪")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        root.addWidget(self._build_sidebar(), 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.viewer3d = MeshViewer3D(show_grid=False)
        self.slice_viewer = SliceViewer()
        splitter.addWidget(self.viewer3d)
        splitter.addWidget(self.slice_viewer)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 480])
        root.addWidget(splitter, 1)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFixedWidth(380)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        load_group = QtWidgets.QGroupBox("导入重构后的三维模型 (.npy)")
        load_layout = QtWidgets.QFormLayout(load_group)
        self.raw_pitch_spin = QtWidgets.QDoubleSpinBox()
        self.raw_pitch_spin.setRange(1.0e-6, 1.0e6)
        self.raw_pitch_spin.setDecimals(6)
        self.raw_pitch_spin.setValue(1.0)
        load_layout.addRow("体素物理尺寸 (mm/voxel):", self.raw_pitch_spin)
        self.raw_mu_spin = QtWidgets.QDoubleSpinBox()
        self.raw_mu_spin.setRange(1.0e-6, 1.0e3)
        self.raw_mu_spin.setDecimals(6)
        self.raw_mu_spin.setValue(0.05)
        load_layout.addRow("线性吸收系数 mu (仅SSIM用):", self.raw_mu_spin)
        load_btn = QtWidgets.QPushButton("浏览并加载...")
        load_btn.clicked.connect(self.on_load_volume_file)
        load_layout.addRow(load_btn)
        layout.addWidget(load_group)

        recon_group = QtWidgets.QGroupBox("导入投影/切片数据集进行重建")
        recon_layout = QtWidgets.QFormLayout(recon_group)

        self.dataset_edit = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("浏览...")
        browse_btn.clicked.connect(self.on_browse_dataset)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.dataset_edit)
        row.addWidget(browse_btn)
        recon_layout.addRow("数据集文件夹:", self._wrap(row))

        self.algo_combo = QtWidgets.QComboBox()
        self.algo_combo.addItems(list(recon.ALGORITHMS.keys()))
        recon_layout.addRow("重建算法:", self.algo_combo)

        self.iters_spin = QtWidgets.QSpinBox()
        self.iters_spin.setRange(1, 10000)
        self.iters_spin.setValue(100)
        recon_layout.addRow("迭代次数 (仅迭代算法):", self.iters_spin)

        self.voxels_spin = QtWidgets.QSpinBox()
        self.voxels_spin.setRange(16, 512)
        self.voxels_spin.setValue(128)
        recon_layout.addRow("重建体素分辨率:", self.voxels_spin)

        self.vol_half_spin = QtWidgets.QDoubleSpinBox()
        self.vol_half_spin.setRange(0.001, 1.0e6)
        self.vol_half_spin.setDecimals(4)
        self.vol_half_spin.setValue(10.0)
        recon_layout.addRow("体积半宽 (mm):", self.vol_half_spin)

        self.recon_btn = QtWidgets.QPushButton("运行重建")
        self.recon_btn.setObjectName("primaryAction")
        self.recon_btn.clicked.connect(self.on_run_reconstruct)
        recon_layout.addRow(self.recon_btn)
        layout.addWidget(recon_group)

        self.recon_info_label = QtWidgets.QLabel("尚无重建信息")
        self.recon_info_label.setWordWrap(True)
        self.recon_info_label.setObjectName("popupInfo")
        layout.addWidget(self.recon_info_label)

        ref_group = QtWidgets.QGroupBox("参考模型 (用于计算 SSIM)")
        ref_layout = QtWidgets.QFormLayout(ref_group)
        self.reference_path_edit = QtWidgets.QLineEdit()
        self.reference_path_edit.setReadOnly(True)
        ref_browse_btn = QtWidgets.QPushButton("浏览...")
        ref_browse_btn.clicked.connect(self.on_browse_reference_stl)
        ref_row = QtWidgets.QHBoxLayout()
        ref_row.addWidget(self.reference_path_edit)
        ref_row.addWidget(ref_browse_btn)
        ref_layout.addRow("参考 STL:", self._wrap(ref_row))

        self.ssim_device_combo = QtWidgets.QComboBox()
        self.ssim_device_combo.addItems(available_devices())
        if self.ssim_device_combo.count() > 1:
            self.ssim_device_combo.setCurrentIndex(self.ssim_device_combo.count() - 1)
        ref_layout.addRow("计算设备:", self.ssim_device_combo)

        self.ssim_label = QtWidgets.QLabel("未计算")
        ref_layout.addRow("SSIM:", self.ssim_label)
        layout.addWidget(ref_group)

        disp_group = QtWidgets.QGroupBox("等值面显示设置")
        disp_layout = QtWidgets.QFormLayout(disp_group)
        self.threshold_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.threshold_slider.setRange(1, THRESHOLD_STEPS - 1)
        self.threshold_slider.setValue(THRESHOLD_STEPS // 2)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)
        disp_layout.addRow("阈值:", self.threshold_slider)
        self.threshold_label = QtWidgets.QLabel("-")
        self.threshold_label.setWordWrap(True)
        disp_layout.addRow("", self.threshold_label)
        layout.addWidget(disp_group)

        export_btn = QtWidgets.QPushButton("导出当前等值面为 STL")
        export_btn.clicked.connect(self.on_export_stl)
        layout.addWidget(export_btn)

        self.status_label = QtWidgets.QLabel("尚未加载数据")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch(1)
        panel.setWidget(content)
        return panel

    @staticmethod
    def _wrap(inner_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(inner_layout)
        return w

    def log(self, msg: str):
        self.statusBar().showMessage(msg, 8000)

    # -------------------------------------------------------------- slots
    def on_load_volume_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择体积文件 (.npy)", "", "NumPy Array (*.npy)")
        if not path:
            return
        try:
            volume = np.load(path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        if volume.ndim != 3:
            QtWidgets.QMessageBox.warning(self, "错误", f"期望一个3维体积数组, 收到 shape={volume.shape}")
            return
        if not (volume.shape[0] == volume.shape[1] == volume.shape[2]):
            self.log("警告: 体积不是立方体, SSIM/等值面尺寸换算假设立方体, 可能不准确")

        self.voxel_pitch = self.raw_pitch_spin.value()
        self.recon_info = {
            "mu": self.raw_mu_spin.value(),
            "rotation_center": [0.0, 0.0, 0.0],
            "vol_half": self.voxel_pitch * volume.shape[0] / 2.0,
            "n_voxels": volume.shape[0],
        }
        self.recon_info_label.setText(
            "直接加载的体积文件, 没有重建过程信息\n"
            f"假定体素尺寸 {self.voxel_pitch:g} mm/voxel, 居中于原点 (与本项目重建坐标约定一致)"
        )
        self._set_volume(volume, f"已加载: {path}")

    def on_browse_dataset(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择包含 projections.npy 的数据集文件夹")
        if folder:
            self.dataset_edit.setText(folder)

    def on_run_reconstruct(self):
        dataset_dir = self.dataset_edit.text().strip()
        if not dataset_dir or not os.path.isfile(os.path.join(dataset_dir, "projections.npy")):
            QtWidgets.QMessageBox.warning(self, "错误", "请先选一个包含 projections.npy 的数据集文件夹")
            return

        self.recon_btn.setEnabled(False)
        self.status_label.setText("重建中...")
        self.log("开始重建...")

        self.recon_thread = ReconstructThread(
            dataset_dir, self.algo_combo.currentText(), self.iters_spin.value(),
            self.voxels_spin.value(), self.vol_half_spin.value(),
        )
        self.recon_thread.done.connect(self._on_recon_done)
        self.recon_thread.failed.connect(self.on_recon_failed)
        self.recon_thread.start()

    def _on_recon_done(self, info: dict):
        self.recon_btn.setEnabled(True)
        self.log(f"重建完成, 耗时 {info['elapsed_s']:.2f} 秒")

        self.recon_info = info
        self.voxel_pitch = 2.0 * info["vol_half"] / info["n_voxels"]

        self.recon_info_label.setText(
            f"数据集: {info['dataset_dir']}\n"
            f"算法: {info['algorithm']}    迭代次数: {info['iterations']}\n"
            f"体素分辨率: {info['n_voxels']}³    体积半宽: {info['vol_half']:.4f} mm\n"
            f"投影角度数: {info['n_views']}    探测器: {info['rows']} x {info['cols']}\n"
            f"重建耗时: {info['elapsed_s']:.3f} 秒\n"
            f"计算设备: {info['gpu_info']}"
        )

        self._set_volume(info["volume"], f"重建完成: {info['dataset_dir']}")

        stl_path = info.get("stl_path")
        if self.reference_mesh is None and stl_path and os.path.isfile(stl_path):
            self.log(f"自动加载参考STL (来自数据集元数据): {stl_path}")
            self._load_reference_stl(stl_path)
        else:
            self._maybe_compute_ssim()

    def on_recon_failed(self, err: str):
        self.recon_btn.setEnabled(True)
        self.status_label.setText("重建失败, 详情见下方弹窗")
        QtWidgets.QMessageBox.critical(self, "重建失败", err)

    def on_browse_reference_stl(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择参考 STL 文件", "", "STL Files (*.stl)")
        if path:
            self._load_reference_stl(path)

    def _load_reference_stl(self, path: str):
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "加载失败", str(exc))
            return
        self.reference_mesh = mesh
        self.reference_path = path
        self.reference_path_edit.setText(path)
        self.log(f"已加载参考STL: {path}")
        self._maybe_compute_ssim()

    def _maybe_compute_ssim(self):
        if self.volume is None or self.reference_mesh is None:
            return
        info = self.recon_info
        mu = info.get("mu")
        rotation_center = info.get("rotation_center")
        vol_half = info.get("vol_half")
        n_voxels = info.get("n_voxels")
        if mu is None or rotation_center is None or vol_half is None or n_voxels is None:
            self.ssim_label.setText("无法计算 (缺少 mu/旋转中心/体积尺寸信息)")
            return

        self.ssim_label.setText("计算中...")
        self.ssim_thread = SSIMThread(
            self.reference_mesh.vertices, self.reference_mesh.faces, self.volume,
            float(vol_half), int(n_voxels), np.asarray(rotation_center, dtype=float),
            float(mu), self.ssim_device_combo.currentText(),
        )
        self.ssim_thread.done.connect(self.on_ssim_done)
        self.ssim_thread.failed.connect(self.on_ssim_failed)
        self.ssim_thread.start()

    def on_ssim_done(self, score: float, elapsed: float):
        self.ssim_label.setText(f"{score:.4f}  (耗时 {elapsed:.2f} 秒)")
        self.log(f"SSIM = {score:.4f}")

    def on_ssim_failed(self, err: str):
        self.ssim_label.setText("计算失败, 详情见日志")
        self.log("SSIM计算失败:\n" + err)
        QtWidgets.QMessageBox.critical(self, "SSIM计算失败", err)

    def _set_volume(self, volume: np.ndarray, status_text: str):
        self.volume = volume
        self.slice_viewer.set_volume(volume)
        self.status_label.setText(
            f"{status_text}\n体积形状: {volume.shape}, 值范围: [{volume.min():.4f}, {volume.max():.4f}]"
        )

        lo, hi = float(volume.min()), float(volume.max())
        default_level = self._default_threshold(volume, lo, hi)
        tick = int(np.clip((default_level - lo) / (hi - lo), 0.001, 0.999) * THRESHOLD_STEPS) if hi > lo else THRESHOLD_STEPS // 2
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(tick)
        self.threshold_slider.blockSignals(False)
        self.on_threshold_changed()

    @staticmethod
    def _default_threshold(volume: np.ndarray, lo: float, hi: float) -> float:
        """Otsu's method: pick the level that best separates two populations (e.g.
        background vs. object) instead of a naive min/max midpoint, which a handful of
        extreme outlier voxels (common at reconstruction boundaries/artifacts) can pull
        far away from where the actual object sits.
        """
        try:
            return float(threshold_otsu(volume))
        except Exception:
            return lo + 0.5 * (hi - lo)

    def _current_level(self) -> float:
        lo, hi = float(self.volume.min()), float(self.volume.max())
        frac = self.threshold_slider.value() / THRESHOLD_STEPS
        return lo + frac * (hi - lo)

    def on_threshold_changed(self):
        if self.volume is None:
            return
        lo, hi = float(self.volume.min()), float(self.volume.max())
        level = self._current_level()
        self.threshold_label.setText(f"阈值 = {level:.4f}  (体积值范围 {lo:.4f} ~ {hi:.4f})")

        mesh = extract_isosurface(self.volume, level, self.voxel_pitch)
        if mesh is None or len(mesh.faces) == 0:
            self.viewer3d.mesh = None
            self.viewer3d.redraw()
            return
        self.viewer3d.set_mesh(mesh, mesh.centroid)

    def on_export_stl(self):
        if self.volume is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存为 STL", "", "STL Files (*.stl)")
        if not path:
            return
        mesh = extract_isosurface(self.volume, self._current_level(), self.voxel_pitch)
        if mesh is None:
            QtWidgets.QMessageBox.warning(self, "错误", "当前阈值下没有等值面, 无法导出")
            return
        mesh.export(path)
        self.status_label.setText(f"已导出等值面: {path}")
        self.log(f"已导出: {path}")


def run_selftest() -> int:
    """Packaged-exe smoke test: load a synthetic volume, extract an isosurface, run a
    tiny real ASTRA reconstruction from a synthetic dataset, and run a tiny real SSIM
    comparison against a synthetic reference mesh (exercises the Warp-dependent path
    too, now that this app needs it again).
    """
    import tempfile

    import astra
    import geometry as geom

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(FLAT_QSS)
    win = MainWindow()
    try:
        # isosurface extraction on a synthetic sphere volume
        n = 32
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
        c = n / 2
        volume = np.sqrt((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2).astype(np.float32)
        volume = (n / 2) - volume  # positive inside a sphere, negative outside
        win._set_volume(volume, "selftest synthetic sphere")
        if win.viewer3d.mesh is None:
            raise RuntimeError("isosurface extraction produced no mesh")

        # tiny real ASTRA reconstruction from a synthetic dataset
        gpu_info = astra.get_gpu_info()
        rows = cols = 16
        n_views = 20
        directions = geom.fibonacci_sphere_directions(n_views)
        rotation_center = np.zeros(3)
        views = geom.build_all_views(directions, rotation_center, sod=100.0, sdd=200.0)
        astra_vectors = geom.to_astra_cone_vec(views, rotation_center, pixel_pitch=0.5)
        sino = np.random.default_rng(0).random((n_views, rows, cols)).astype(np.float32)
        rec = recon.reconstruct(sino, astra_vectors, rows, cols, vol_half=8.0, n_voxels=16, iterations=3)
        if rec.shape != (16, 16, 16):
            raise RuntimeError(f"unexpected reconstruction shape {rec.shape}")

        # tiny real SSIM comparison against a synthetic reference mesh (needs Warp)
        ref_mesh = trimesh.creation.icosphere(subdivisions=1, radius=5.0)
        ground_truth = recon.voxelize_ground_truth(
            ref_mesh.vertices, ref_mesh.faces, vol_half=8.0, n_voxels=16,
            center=np.zeros(3), device="cpu",
        )
        score = recon.compute_ssim(ground_truth, rec, mu=0.05)
        if not (0.0 <= score <= 1.0):
            raise RuntimeError(f"unexpected SSIM value {score}")

        print(
            f"selftest ok: isosurface extraction works; ASTRA reconstruction ran ok "
            f"(GPU: {gpu_info}); SSIM against synthetic reference = {score:.4f}"
        )
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
