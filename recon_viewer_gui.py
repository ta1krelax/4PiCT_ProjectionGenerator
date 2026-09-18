"""Standalone CT reconstruction viewer.

Load an already-reconstructed volume (.npy) directly, or load a projection dataset
(projections.npy + astra_vectors.npy + metadata.json, as exported by projector_gui.py's
generation/sequence/batch-experiment features) and run an ASTRA reconstruction on it --
then inspect the result as an interactive 3D isosurface (rotate/zoom, adjustable
threshold) plus scrubbable orthogonal slices for interior detail.

Deliberately does NOT import warp-lang (no GPU raytracing / STL slicing here): this is
the reconstruction+viewing half of the pipeline, split out from projector_gui.py so this
exe doesn't have to bundle Warp's ~350MB JIT toolchain just to look at a volume.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import matplotlib
import numpy as np
from PySide6 import QtCore, QtWidgets

import reconstruction as recon
from mesh_viewer import MeshViewer3D
from theme import FLAT_QSS
from volume_view import SliceViewer, extract_isosurface

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

THRESHOLD_STEPS = 1000


class ReconstructThread(QtCore.QThread):
    done = QtCore.Signal(np.ndarray)
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
            rec = recon.reconstruct(
                sino, astra_vectors, rows, cols, self.vol_half, self.n_voxels,
                algorithm=self.algorithm, iterations=self.iterations,
            )
            self.done.emit(rec)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CT Reconstruction Viewer")
        self.resize(1320, 880)
        self.volume: np.ndarray | None = None
        self.recon_thread: ReconstructThread | None = None
        self._build_ui()
        self.statusBar().showMessage("就绪")

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        root.addWidget(self._build_sidebar(), 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.viewer3d = MeshViewer3D()
        self.slice_viewer = SliceViewer()
        splitter.addWidget(self.viewer3d)
        splitter.addWidget(self.slice_viewer)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QScrollArea()
        panel.setWidgetResizable(True)
        panel.setFixedWidth(380)
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)

        load_group = QtWidgets.QGroupBox("导入重构后的三维模型 (.npy)")
        load_layout = QtWidgets.QVBoxLayout(load_group)
        load_btn = QtWidgets.QPushButton("浏览并加载...")
        load_btn.clicked.connect(self.on_load_volume_file)
        load_layout.addWidget(load_btn)
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
        self.recon_thread.done.connect(lambda vol: self._on_recon_done(vol, dataset_dir))
        self.recon_thread.failed.connect(self.on_recon_failed)
        self.recon_thread.start()

    def _on_recon_done(self, volume: np.ndarray, dataset_dir: str):
        self.recon_btn.setEnabled(True)
        self.log("重建完成")
        self._set_volume(volume, f"重建完成: {dataset_dir}")

    def on_recon_failed(self, err: str):
        self.recon_btn.setEnabled(True)
        self.status_label.setText("重建失败, 详情见下方弹窗")
        QtWidgets.QMessageBox.critical(self, "重建失败", err)

    def _set_volume(self, volume: np.ndarray, status_text: str):
        self.volume = volume
        self.slice_viewer.set_volume(volume)
        self.status_label.setText(
            f"{status_text}\n体积形状: {volume.shape}, 值范围: [{volume.min():.4f}, {volume.max():.4f}]"
        )
        self.on_threshold_changed()

    def on_threshold_changed(self):
        if self.volume is None:
            return
        lo, hi = float(self.volume.min()), float(self.volume.max())
        frac = self.threshold_slider.value() / THRESHOLD_STEPS
        level = lo + frac * (hi - lo)
        self.threshold_label.setText(f"阈值 = {level:.4f}  (体积值范围 {lo:.4f} ~ {hi:.4f})")

        mesh = extract_isosurface(self.volume, level)
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
        lo, hi = float(self.volume.min()), float(self.volume.max())
        frac = self.threshold_slider.value() / THRESHOLD_STEPS
        level = lo + frac * (hi - lo)
        mesh = extract_isosurface(self.volume, level)
        if mesh is None:
            QtWidgets.QMessageBox.warning(self, "错误", "当前阈值下没有等值面, 无法导出")
            return
        mesh.export(path)
        self.status_label.setText(f"已导出等值面: {path}")
        self.log(f"已导出: {path}")


def run_selftest() -> int:
    """Packaged-exe smoke test: load a synthetic volume, extract an isosurface, and run
    a tiny real ASTRA reconstruction from a synthetic dataset -- not just window-opens.
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

        print(f"selftest ok: isosurface extraction works; ASTRA reconstruction ran ok (GPU: {gpu_info})")
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
