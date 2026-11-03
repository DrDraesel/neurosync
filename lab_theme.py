"""Qt theme for the lab design system (PyQt6).

Maps design-system/tokens.css to a QPalette + QSS stylesheet so the desktop
apps (NeuroSync) read as the same product family as the web dashboards.

Usage:
    from lab_theme import apply_theme
    apply_theme(app)                  # or apply_theme(app, mode="dark")

Keep every colour here in sync with tokens.css. Do not hard-code colours in
widgets; use lab_theme.C (dict) so a palette change lands everywhere.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import QApplication

C = {
    # canvas
    "bg": "#F7F0E7",
    "surface": "#FFFFFF",
    "surface_2": "#FBF7F1",
    "surface_3": "#F1EBE2",
    "surface_inset": "#ECE6DD",
    # ink
    "ink": "#1A1A1A",
    "ink_2": "#6E6961",
    "ink_3": "#9C968B",
    # lines
    "line": "#E9E2D8",
    "line_strong": "#D9D1C5",
    # accent
    "accent": "#E8541F",
    "accent_600": "#D24715",
    "accent_200": "#F7C9B2",
    "accent_100": "#FBE2D5",
    # semantic
    "ok": "#1F7A4B",
    "ok_bg": "#E4F4E8",
    "warn": "#9A5B00",
    "warn_bg": "#FBEEDA",
    "danger": "#C8341C",
    "danger_bg": "#FBE4DF",
    "info": "#2C5C8F",
    "info_bg": "#E6EFF8",
    # series
    "c_sms": "#E8541F",
    "c_whatsapp": "#3FA96A",
    "c_email": "#5B6478",
    "c_violet": "#7C5CFF",
    "c_teal": "#1F8A8A",
    # rail
    "rail": "#161616",
    "rail_ink": "#A9A39A",
}

FONT_STACK = ["Inter", "Segoe UI Variable Text", "Segoe UI", "Arial"]


def _font(size: int, weight: int = 400) -> QFont:
    f = QFont()
    f.setFamilies(FONT_STACK)
    f.setPointSizeF(size)
    f.setWeight(QFont.Weight(weight))
    return f


def apply_theme(app: QApplication, mode: str = "light") -> None:
    """Apply palette + stylesheet. mode: 'light' (default) | 'dark'."""
    if mode == "dark":
        C.update(
            bg="#121212", surface="#1B1B1B", surface_2="#202020", surface_3="#262626",
            surface_inset="#2B2B2B", ink="#F4EFE7", ink_2="#B9B3A9", ink_3="#8C867C",
            line="#2E2E2E", line_strong="#3A3A3A",
        )

    app.setFont(_font(10))
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(C["bg"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(C["surface"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(C["surface_2"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(C["ink"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(C["ink"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(C["ink"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(C["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(C["surface"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(C["ink"]))
    app.setPalette(pal)

    app.setStyleSheet(f"""
    QWidget {{ background: {C['bg']}; color: {C['ink']};
               font-family: "Inter", "Segoe UI Variable Text", "Segoe UI"; font-size: 10pt; }}
    QMainWindow, QDialog {{ background: {C['bg']}; }}

    QFrame#Card, QGroupBox {{
        background: {C['surface']}; border: 1px solid {C['line']};
        border-radius: 14px; padding: 14px; margin-top: 10px;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px;
                        color: {C['ink_2']}; font-weight: 600; }}

    QLabel#H1 {{ font-size: 20pt; font-weight: 600; letter-spacing: -0.4px; }}
    QLabel#H2 {{ font-size: 12pt; font-weight: 600; }}
    QLabel#Muted {{ color: {C['ink_3']}; font-size: 9pt; }}
    QLabel#Kpi {{ font-size: 24pt; font-weight: 600; }}
    QLabel#BadgeOk {{ background: {C['ok_bg']}; color: {C['ok']}; border-radius: 9px;
                      padding: 2px 8px; font-size: 8.5pt; font-weight: 600; }}
    QLabel#BadgeWarn {{ background: {C['warn_bg']}; color: {C['warn']}; border-radius: 9px;
                        padding: 2px 8px; font-size: 8.5pt; font-weight: 600; }}
    QLabel#BadgeBad {{ background: {C['danger_bg']}; color: {C['danger']}; border-radius: 9px;
                       padding: 2px 8px; font-size: 8.5pt; font-weight: 600; }}

    QPushButton {{
        background: {C['surface']}; border: 1px solid {C['line_strong']};
        border-radius: 10px; padding: 7px 14px; font-weight: 500;
    }}
    QPushButton:hover {{ background: {C['surface_2']}; border-color: {C['ink_3']}; }}
    QPushButton:pressed {{ background: {C['surface_3']}; }}
    QPushButton:disabled {{ color: {C['ink_3']}; border-color: {C['line']}; }}
    QPushButton#Primary {{ background: {C['accent']}; border-color: {C['accent']}; color: #fff; }}
    QPushButton#Primary:hover {{ background: {C['accent_600']}; border-color: {C['accent_600']}; }}
    QPushButton#Danger {{ background: {C['danger_bg']}; border-color: transparent; color: {C['danger']}; }}

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
        background: {C['surface']}; border: 1px solid {C['line_strong']};
        border-radius: 10px; padding: 6px 10px; selection-background-color: {C['accent']};
    }}
    QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {C['accent']}; }}

    QTabWidget::pane {{ border: 1px solid {C['line']}; border-radius: 12px; background: {C['surface']}; }}
    QTabBar::tab {{ background: transparent; color: {C['ink_2']}; padding: 8px 16px;
                    border-radius: 9px; margin: 2px; }}
    QTabBar::tab:selected {{ background: {C['surface']}; color: {C['ink']}; font-weight: 600; }}

    QHeaderView::section {{ background: {C['surface_2']}; color: {C['ink_2']};
                            border: 0; border-bottom: 1px solid {C['line']}; padding: 7px 10px; }}
    QTableView, QTableWidget {{ background: {C['surface']}; border: 1px solid {C['line']};
                                border-radius: 12px; gridline-color: {C['line']};
                                alternate-background-color: {C['surface_2']}; }}
    QTableView::item:selected {{ background: {C['accent_100']}; color: {C['ink']}; }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
    QScrollBar::handle:vertical {{ background: {C['line_strong']}; border-radius: 5px; min-height: 30px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QToolTip {{ background: {C['surface']}; color: {C['ink']};
                border: 1px solid {C['line_strong']}; padding: 6px; border-radius: 8px; }}
    QStatusBar {{ color: {C['ink_3']}; }}
    """)
