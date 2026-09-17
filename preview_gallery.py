"""Right-hand preview gallery (PowerPoint-slide-panel style) + full-size popup viewer."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

THUMB_SIZE = 220


def _to_qpixmap(image: np.ndarray) -> QtGui.QPixmap:
    lo, hi = float(image.min()), float(image.max())
    if hi - lo < 1e-12:
        scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        scaled = ((image - lo) / (hi - lo) * 255.0).astype(np.uint8)
    scaled = np.ascontiguousarray(scaled)
    h, w = scaled.shape
    qimg = QtGui.QImage(scaled.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return QtGui.QPixmap.fromImage(qimg.copy())


class ClickableThumb(QtWidgets.QFrame):
    doubleClicked = QtCore.Signal()

    def __init__(self, pixmap: QtGui.QPixmap, caption: str, parent=None):
        super().__init__(parent)
        self.setObjectName("thumbCard")
        self.setCursor(QtCore.Qt.PointingHandCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        img_label = QtWidgets.QLabel()
        img_label.setPixmap(pixmap.scaled(THUMB_SIZE, THUMB_SIZE, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        img_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(img_label)

        cap_label = QtWidgets.QLabel(caption)
        cap_label.setWordWrap(True)
        cap_label.setAlignment(QtCore.Qt.AlignCenter)
        cap_label.setObjectName("thumbCaption")
        layout.addWidget(cap_label)

    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


class PreviewPopup(QtWidgets.QDialog):
    def __init__(self, image: np.ndarray, meta: dict, parent=None):
        super().__init__(parent)
        self.image = image
        self.setWindowTitle(meta.get("title", "投影预览"))
        self.resize(720, 640)

        layout = QtWidgets.QVBoxLayout(self)

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        im = self.ax.imshow(image, cmap="gray")
        self.ax.set_title(meta.get("title", ""))
        self.fig.colorbar(im, ax=self.ax, fraction=0.046, pad=0.04, label="mu * path length")
        layout.addWidget(self.canvas)

        self.info_label = QtWidgets.QLabel(self._static_info_text(meta))
        self.info_label.setObjectName("popupInfo")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self._static_text = self._static_info_text(meta)
        self.info_label.setText(self._static_text + "\n(将鼠标移到图像上查看像素信息)")

        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.canvas.mpl_connect("axes_leave_event", self._on_leave)

    @staticmethod
    def _static_info_text(meta: dict) -> str:
        return (
            f"采样方案: {meta.get('scheme_name', '-')}   视角索引: {meta.get('view_index', '-')}\n"
            f"方向向量: {meta.get('direction', '-')}\n"
            f"SOD: {meta.get('sod', 0):.3f} mm   SDD: {meta.get('sdd', 0):.3f} mm   "
            f"放大倍率: {meta.get('magnification', 0):.3f}x\n"
            f"像素尺寸: {meta.get('pixel_pitch', 0):.4f} mm/px   分辨率: {meta.get('rows', 0)} x {meta.get('cols', 0)}   "
            f"mu: {meta.get('mu', 0):.5f} /mm"
        )

    def _on_hover(self, event):
        if event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        rows, cols = self.image.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return
        value = float(self.image[row, col])
        transmission = float(np.exp(-value))
        self.info_label.setText(
            self._static_text + f"\n像素 (row={row}, col={col})   线积分值={value:.4f}   透过率={transmission * 100:.2f}%"
        )

    def _on_leave(self, event):
        self.info_label.setText(self._static_text + "\n(将鼠标移到图像上查看像素信息)")


class PreviewGallery(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._count = 0

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        title = QtWidgets.QLabel("预览记录")
        title.setObjectName("galleryTitle")
        outer.addWidget(title)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        content = QtWidgets.QWidget()
        self.list_layout = QtWidgets.QVBoxLayout(content)
        self.list_layout.setAlignment(QtCore.Qt.AlignTop)
        self.list_layout.addStretch(1)
        self.scroll.setWidget(content)

        outer.addWidget(self.scroll)

    def add_preview(self, image: np.ndarray, meta: dict):
        self._count += 1
        meta = dict(meta)
        meta.setdefault("view_index", 0)
        meta["title"] = f"预览 #{self._count}  ({meta.get('scheme_name', '')})"

        pixmap = _to_qpixmap(image)
        caption = f"#{self._count}  {meta.get('scheme_name', '')}  view {meta.get('view_index', 0)}"
        thumb = ClickableThumb(pixmap, caption)
        thumb.doubleClicked.connect(lambda: self._open_popup(image, meta))

        self.list_layout.insertWidget(self.list_layout.count() - 1, thumb)
        QtCore.QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def _open_popup(self, image: np.ndarray, meta: dict):
        popup = PreviewPopup(image, meta, parent=self)
        popup.setWindowFlag(QtCore.Qt.Window, True)
        popup.show()
        self._popups = getattr(self, "_popups", [])
        self._popups.append(popup)
