"""Isosurface extraction + a slider-driven orthogonal slice viewer for 3D volumes."""

from __future__ import annotations

import numpy as np
import trimesh
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtWidgets
from skimage import measure


def extract_isosurface(volume: np.ndarray, level: float, voxel_pitch: float = 1.0) -> trimesh.Trimesh | None:
    """Marching-cubes isosurface at `level`, scaled to physical units and centered on
    the origin (matching this project's reconstruction-volume convention: vol_geom is
    always symmetric about 0). Returns None if `level` is outside the volume's actual
    value range (marching_cubes raises in that case).
    """
    lo, hi = float(volume.min()), float(volume.max())
    if not (lo < level < hi):
        return None
    verts, faces, _normals, _values = measure.marching_cubes(volume, level=level)
    half_extent = np.asarray(volume.shape, dtype=np.float64) * voxel_pitch / 2.0
    verts = verts * voxel_pitch - half_extent
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


class SliceViewer(QtWidgets.QWidget):
    """Three orthogonal mid-slices (or wherever the sliders point), for looking at
    interior detail an isosurface alone would hide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume: np.ndarray | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(6, 2.4))
        self.canvas = FigureCanvas(self.fig)
        self.axes = self.fig.subplots(1, 3)
        layout.addWidget(self.canvas)

        self.sliders: dict[str, QtWidgets.QSlider] = {}
        self.slider_labels: dict[str, QtWidgets.QLabel] = {}
        form = QtWidgets.QFormLayout()
        for axis_name in ("X", "Y", "Z"):
            row = QtWidgets.QHBoxLayout()
            sl = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            sl.valueChanged.connect(self._redraw)
            lbl = QtWidgets.QLabel("0")
            lbl.setFixedWidth(40)
            row.addWidget(sl)
            row.addWidget(lbl)
            self.sliders[axis_name] = sl
            self.slider_labels[axis_name] = lbl
            form.addRow(f"{axis_name} 切片:", self._wrap(row))
        layout.addLayout(form)

    @staticmethod
    def _wrap(inner_layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        w.setLayout(inner_layout)
        return w

    def set_volume(self, volume: np.ndarray):
        self.volume = volume
        nx, ny, nz = volume.shape
        for name, n in zip(("X", "Y", "Z"), (nx, ny, nz)):
            sl = self.sliders[name]
            sl.blockSignals(True)
            sl.setRange(0, max(n - 1, 0))
            sl.setValue(n // 2)
            sl.blockSignals(False)
        self._redraw()

    def _redraw(self):
        if self.volume is None:
            return
        v = self.volume
        idx = {name: self.sliders[name].value() for name in ("X", "Y", "Z")}
        for name, i in idx.items():
            self.slider_labels[name].setText(str(i))

        imgs = [v[idx["X"], :, :], v[:, idx["Y"], :], v[:, :, idx["Z"]]]
        titles = [f"X = {idx['X']}", f"Y = {idx['Y']}", f"Z = {idx['Z']}"]
        for ax, img, title in zip(self.axes, imgs, titles):
            ax.clear()
            ax.imshow(img, cmap="gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        self.fig.tight_layout()
        self.canvas.draw_idle()
