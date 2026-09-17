"""Interactive 3D mesh preview.

Mouse-drag rotates the OBJECT itself (the mesh vertex data), not the camera --
matplotlib's built-in mouse rotation/pan/zoom is disabled, so the world axes
(ticks, grid, labels) never move; only the plotted part spins in place.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PySide6 import QtWidgets

MAX_PREVIEW_FACES = 4000
ROTATE_SENSITIVITY = 0.008
ZOOM_STEP = 1.15
ZOOM_MIN, ZOOM_MAX = 0.05, 50.0


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class MeshViewer3D(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.disable_mouse_rotation()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.mesh = None
        self.rotation_center = np.zeros(3)
        self.view_rotation = np.eye(3)
        self.zoom = 1.0

        self._base_center = np.zeros(3)
        self._base_radius = 1.0
        self._sample_idx = None
        self._drag_last = None

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        self._init_axes()

    def _init_axes(self):
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("左键拖拽旋转模型 / 滚轮缩放 (坐标轴固定不动)")

    def set_mesh(self, mesh, rotation_center):
        self.mesh = mesh
        self.rotation_center = np.asarray(rotation_center, dtype=float)
        self.view_rotation = np.eye(3)
        self.zoom = 1.0

        bounds = mesh.bounds
        self._base_center = bounds.mean(axis=0)
        self._base_radius = float(np.max(bounds[1] - bounds[0]) / 2.0 * 1.1) + 1e-9

        faces = mesh.faces
        if len(faces) > MAX_PREVIEW_FACES:
            self._sample_idx = np.random.default_rng(0).choice(len(faces), MAX_PREVIEW_FACES, replace=False)
        else:
            self._sample_idx = None

        self.redraw()

    def set_rotation_center(self, rotation_center):
        if self.mesh is None:
            return
        self.rotation_center = np.asarray(rotation_center, dtype=float)
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self._init_axes()

        if self.mesh is not None:
            faces = self.mesh.faces if self._sample_idx is None else self.mesh.faces[self._sample_idx]
            verts = self.mesh.vertices
            rotated = (verts - self.rotation_center) @ self.view_rotation.T + self.rotation_center
            tris = rotated[faces]

            coll = Poly3DCollection(tris, alpha=0.65, facecolor="#7fa8d9", edgecolor="#33475b", linewidths=0.15)
            self.ax.add_collection3d(coll)
            self.ax.scatter(*self.rotation_center, color="#e2574c", s=45, depthshade=False)

            c = self._base_center
            r = self._base_radius / max(self.zoom, 1e-3)
            self.ax.set_xlim(c[0] - r, c[0] + r)
            self.ax.set_ylim(c[1] - r, c[1] + r)
            self.ax.set_zlim(c[2] - r, c[2] + r)

        self.canvas.draw_idle()

    # ------------------------------------------------------------- mouse
    def _on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._drag_last = (event.x, event.y)

    def _on_release(self, event):
        self._drag_last = None

    def _on_motion(self, event):
        if self._drag_last is None or self.mesh is None or event.x is None or event.y is None:
            return
        dx = event.x - self._drag_last[0]
        dy = event.y - self._drag_last[1]
        self._drag_last = (event.x, event.y)

        step = _rot_z(dx * ROTATE_SENSITIVITY) @ _rot_x(-dy * ROTATE_SENSITIVITY)
        self.view_rotation = step @ self.view_rotation
        self.redraw()

    def _on_scroll(self, event):
        if self.mesh is None or event.inaxes != self.ax:
            return
        factor = ZOOM_STEP if event.button == "up" else (1.0 / ZOOM_STEP)
        self.zoom = float(np.clip(self.zoom * factor, ZOOM_MIN, ZOOM_MAX))
        self.redraw()
