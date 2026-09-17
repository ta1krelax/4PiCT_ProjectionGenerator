"""Flat, PowerPoint-ribbon-inspired QSS theme."""

FLAT_QSS = """
QWidget {
    background: #f4f6f9;
    color: #202634;
    font-size: 13px;
}

QMainWindow {
    background: #eef1f6;
}

QToolButton#ribbonTab {
    background: transparent;
    border: none;
    border-bottom: 3px solid transparent;
    padding: 8px 16px;
    font-weight: 600;
    color: #4a5568;
}
QToolButton#ribbonTab:hover {
    background: #e4e9f2;
}
QToolButton#ribbonTab:checked {
    color: #2f6fed;
    border-bottom: 3px solid #2f6fed;
    background: #e9f0fe;
}

QWidget#ribbonPanel {
    background: #ffffff;
    border-bottom: 1px solid #dbe1ea;
}

QWidget#actionRow {
    background: #ffffff;
    border-bottom: 1px solid #dbe1ea;
}

QPushButton {
    background: #ffffff;
    border: 1px solid #c7cfdb;
    border-radius: 5px;
    padding: 6px 14px;
}
QPushButton:hover {
    background: #eef2fb;
    border-color: #2f6fed;
}
QPushButton:pressed {
    background: #dbe6fb;
}
QPushButton:disabled {
    color: #9aa4b2;
    background: #f0f1f4;
}

QPushButton#primaryAction {
    background: #2f6fed;
    color: white;
    border: none;
    font-weight: 600;
    padding: 8px 18px;
}
QPushButton#primaryAction:hover {
    background: #4c85f2;
}
QPushButton#primaryAction:disabled {
    background: #a9c0f2;
}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #c7cfdb;
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 20px;
}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #2f6fed;
}

QRadioButton {
    padding: 2px;
}

QProgressBar {
    background: #e4e9f2;
    border: none;
    border-radius: 4px;
    height: 14px;
    text-align: center;
}
QProgressBar::chunk {
    background: #2f6fed;
    border-radius: 4px;
}

QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #dbe1ea;
    border-radius: 4px;
    font-family: Consolas, monospace;
}

QScrollArea {
    border: none;
}

QLabel#galleryTitle {
    font-weight: 600;
    padding: 8px 10px 4px 10px;
    color: #4a5568;
}

QFrame#thumbCard {
    background: #ffffff;
    border: 1px solid #dbe1ea;
    border-radius: 6px;
    margin: 4px 8px;
}
QFrame#thumbCard:hover {
    border-color: #2f6fed;
}

QLabel#thumbCaption {
    color: #5b6472;
    font-size: 11px;
}

QLabel#popupInfo {
    background: #f4f6f9;
    border: 1px solid #dbe1ea;
    border-radius: 4px;
    padding: 6px 8px;
    font-family: Consolas, monospace;
    font-size: 12px;
}

QStatusBar {
    background: #eef1f6;
    color: #5b6472;
}

QSplitter::handle {
    background: #dbe1ea;
}
"""
