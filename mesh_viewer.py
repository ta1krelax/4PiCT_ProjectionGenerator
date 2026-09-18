"""Interactive 3D mesh preview.

Mouse-drag rotates the OBJECT itself (the mesh vertex data), not the camera --
matplotlib's built-in mouse rotation/pan/zoom is disabled. Two display modes:

- show_grid=True (default; used for the STL slicing preview, where absolute
  coordinates matter -- rotation center, SOD etc. are all defined in this frame):
  the world axes (ticks, grid, labels) stay fixed in place while the plotted part
  spins, so the fixed CT-scan frame stays legible.
- show_grid=False (used for the reconstruction viewer, where only shape matters,
  not absolute coordinates): no grid/ticks/labels at all, just a small XYZ
  orientation triad in the bottom-left corner that rotates along with the model
  (a conventional CAD-viewer-style gizmo), so the view stays uncluttered.
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

GIZMO_AXES = (
    ("X", "#e2574c", np.array([1.0, 0.0, 0.0])),
    ("Y", "#2f9e44", np.array([0.0, 1.0, 0.0])),
    ("Z", "#3b82f6", np.array([0.0, 0.0, 1.0])),
)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class MeshViewer3D(QtWidgets.QWidget):
    def __init__(self, parent=None, show_grid: bool = True):
        super().__init__(parent)
        self.show_grid = show_grid

        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.disable_mouse_rotation()

        self.gizmo_ax = None
        if not self.show_grid:
            self.gizmo_ax = self.fig.add_axes([0.02, 0.02, 0.16, 0.16], projection="3d")
            self.gizmo_ax.disable_mouse_rotation()
            self.gizmo_ax.set_box_aspect((1, 1, 1))

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
        if self.show_grid:
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.set_title("左键拖拽旋转模型 / 滚轮缩放 (坐标轴固定不动)")
        else:
            self.ax.set_axis_off()
            self.ax.set_title("左键拖拽旋转 / 滚轮缩放", fontsize=10)

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
            if self.show_grid:
                self.ax.scatter(*self.rotation_center, color="#e2574c", s=45, depthshade=False)

            c = self._base_center
            r = self._base_radius / max(self.zoom, 1e-3)
            self.ax.set_xlim(c[0] - r, c[0] + r)
            self.ax.set_ylim(c[1] - r, c[1] + r)
            self.ax.set_zlim(c[2] - r, c[2] + r)

        if not self.show_grid:
            self._draw_gizmo()

        self.canvas.draw_idle()

    def _draw_gizmo(self):
        ax = self.gizmo_ax
        ax.clear()
        ax.set_axis_off()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        for label, color, direction in GIZMO_AXES:
            tip = self.view_rotation @ direction
            ax.plot([0, tip[0]], [0, tip[1]], [0, tip[2]], color=color, linewidth=2.2)
            ax.text(tip[0] * 1.3, tip[1] * 1.3, tip[2] * 1.3, label, color=color, fontsize=8, ha="center", va="center")

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
