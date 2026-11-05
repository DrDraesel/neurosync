"""qEEG 5 x 5 grid panel: 25 schematic topographic heads over a z colour ramp.

This is the display half of the qEEG sheet described in ``docs/QEEG-GRID-SPEC.md``.
The numbers come from ``qeeg.compute_metrics`` / ``qeeg.grid``; this module never
computes a metric itself and never guesses a channel value. Head positions and
interpolation come from ``topomap`` (schematic 10-20 layout, multiquadric
interpolation), so the grid is exactly as schematic as the single live map.

HONESTY CONTRACT
----------------
Every sheet carries the reference block returned by ``qeeg.grid``: kind,
label and n. The label is always printed in the panel, in the exported PNG and
in the patient report. A reference is called a *local norm* only when
``kind == 'norm'`` (a local file with >= 20 subjects); no other reference is
ever described as normative. Nothing here is called NeuroGuide, nothing here
is a diagnosis, and the words normative database appear only in the fixed
disclaimer "not a normative clinical database".

Compute policy: one full computation per open/refresh (Refresh button, tab
activation, band/row change). There is no timer in this module; the host app
must not call :meth:`QEEGPanel.refresh` from its 1 s analysis timer. Results
are cached by ``(session_key, band_set, reference)`` so re-opening the tab does
not repeat the Welch/coherence work.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                         QPixmap)
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QSizePolicy, QStackedWidget,
                             QToolButton, QVBoxLayout, QWidget)

import qeeg
import topomap

# Row order is the reference-sheet order, top to bottom. Keep these keys in the
# same spelling as qeeg.ROWS; display labels are separate.
ROW_LABELS = OrderedDict([
    ('absolute', 'Absolute power'),
    ('relative', 'Relative power'),
    ('asymmetry', 'Amplitude asymmetry'),
    ('coherence', 'Coherence'),
    ('phase', 'Phase lag'),
])
BAND_LABELS = OrderedDict([
    ('Delta', 'Delta'),
    ('Theta', 'Theta'),
    ('Alpha', 'Alpha'),
    ('Beta', 'Beta'),
    ('HighBeta', 'High Beta'),
])
ROW_UNITS = {'absolute': 'µV²', 'relative': '%', 'asymmetry': '%',
             'coherence': '0–1', 'phase': '°'}
ROW_KEYS = tuple(ROW_LABELS)
BAND_KEYS = tuple(BAND_LABELS)

NOT_NORMATIVE = 'not a normative clinical database'
REPORT_NOTE = 'Not a normative clinical database; educational display only.'
EMPTY_NO_DATA = ('No recording yet. Connect a headset (or the simulator) and wait a few '
                 'seconds — the qEEG sheet needs at least 3 scalp channels.')
EMPTY_FEW_CHANNELS = ('The qEEG grid needs at least 3 scalp channels; this device exposes '
                      '{count}. Nothing is interpolated from fewer than 3 sites.')
EMPTY_NO_SUBJECT = 'Set a subject before adding a qEEG grid to the patient report.'

GUIDE_TEXT = """Five bands = five "speeds". Delta (1–4 Hz) restorative sleep; Theta (4–8) drowsy / shifting
to sleep — the classic ADHD marker when excessive; Alpha (8–13) calm and relaxed;
Beta (13–30) focused thinking; High Beta (30–45) stress response.
Colours: blue = below the reference, green = at it, yellow→red = above it.
Absolute power = how much energy is in that band at that site.
Relative power = that band's share of the total (which speed dominates).
Amplitude asymmetry = left vs right difference at homologous sites.
Coherence = how similarly two sites behave in that band (shared information).
Phase lag = the timing offset between those sites (how late or early the signal arrives).
Read clusters, not single heads: a single coloured head is weak evidence, a region that
repeats across rows and bands is the pattern worth discussing.
This app compares against a reference you select (session, your saved baseline, or a local
norm file). It is NOT an age-matched clinical normative database, and nothing here is a diagnosis."""

# Reference choices: display text -> internal kind. "Load norm file…" is a
# separate action that fills self.norm_reference; selecting it without a file
# shows an honest empty state instead of silently falling back.
REFERENCE_CHOICES = (
    ('This session', 'session'),
    ('Saved baseline', 'baseline'),
    ('Load norm file…', 'norm'),
    ('Raw', 'none'),
)


def _finite(value):
    """float(value) or None; bools are never measurements."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _fmt(value, places=2):
    number = _finite(value)
    return '—' if number is None else f'{number:.{places}f}'


def describe_reference(reference):
    """One always-printed line with kind, label, n and the disclaimer.

    Only ``kind == 'norm'`` may be described as a local norm; every other kind
    gets a plain description of the comparison. The words "NeuroGuide" and
    "normative" (outside the fixed disclaimer) never appear here.
    """
    ref = reference if isinstance(reference, dict) else {}
    kind = str(ref.get('kind') or 'none')
    label = str(ref.get('label') or '').strip()
    n = ref.get('n')
    if kind == 'session':
        text = "kind: session · self-referenced vs this session's channel mean"
    elif kind == 'baseline':
        if label.startswith('vs this subject'):
            text = f'kind: baseline · {label}'
        else:
            text = 'kind: baseline · vs this subject\'s saved baseline'
            if label:
                text += f' ({label})'
    elif kind == 'norm':
        if label.startswith('vs local norm'):
            text = f'kind: norm · {label}'
        else:
            text = 'kind: norm · vs local norm'
            if n not in (None, ''):
                text += f' n={n}'
            if label:
                text += f', age {label}'
    else:
        text = 'kind: none · raw values — no reference selected'
    if kind != 'norm' and n not in (None, '', 0):
        text += f' · n={n}'
    return f'Reference — {text} · {NOT_NORMATIVE}'


def _cell_entries(entries):
    """(channel/value) pairs from a grid cell map.

    qeeg.grid returns per-channel entries for every row. If a future version
    returns an asymmetry/coherence pair key such as ``F3/F4``, the pair value is
    placed on both named sites for the picture and the tooltip keeps the pair
    name; no new coordinate is invented for a midpoint.
    """
    pairs = []
    for name, entry in (entries or {}).items():
        value = _finite(entry.get('value')) if isinstance(entry, dict) else _finite(entry)
        if value is None:
            continue
        text = str(name)
        if '/' in text:
            for part in text.split('/'):
                part = part.strip()
                if part:
                    pairs.append((part, value, text))
        else:
            pairs.append((text, value, text))
    return pairs


class ZColorBar(QWidget):
    """Vertical z colour bar, +3 at the top and −3 at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.limits = (-3.0, 3.0)
        self.setMinimumWidth(84)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setToolTip('z colour scale: blue = below the reference, green = at it, '
                        'yellow→red = above it. Fixed −3 … +3.')

    def set_limits(self, limits):
        try:
            low, high = float(limits[0]), float(limits[1])
        except (TypeError, ValueError, IndexError):
            low, high = -3.0, 3.0
        if not (np.isfinite(low) and np.isfinite(high)) or high <= low:
            low, high = -3.0, 3.0
        self.limits = (low, high)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FFFFFF'))
        left, top = 6, 22
        width = 20
        height = max(60, self.height() - 46)
        low, high = self.limits
        steps = 96
        for index in range(steps):
            z = high - (high - low) * index / (steps - 1)
            colour = QColor(qeeg.color_for_z(z, self.limits))
            y = top + height * index / steps
            painter.fillRect(QRectF(left, y, width, height / steps + 1.0), colour)
        painter.setPen(QPen(QColor('#6E6961'), 1))
        painter.drawRect(QRectF(left, top, width, height))
        painter.setFont(QFont('Segoe UI', 7))
        painter.setPen(QColor('#1A1A1A'))
        for tick in (3, 2, 1, 0, -1, -2, -3):
            if tick > high or tick < low:
                continue
            y = top + height * (high - tick) / (high - low)
            label = f'+{tick}' if tick > 0 else ('0' if tick == 0 else f'−{abs(tick)}')
            painter.drawText(QRectF(left + width + 4, y - 7, 40, 14),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             label)
        painter.setFont(QFont('Segoe UI', 8, QFont.Weight.DemiBold))
        painter.drawText(QRectF(0, 2, self.width(), 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, 'z')
        painter.end()


class TopoCell(QWidget):
    """One miniature schematic head, coloured by z (or neutral for raw values)."""

    def __init__(self, row, band, parent=None):
        super().__init__(parent)
        self.row = row
        self.band = band
        self.entries = {}
        self.reference_kind = 'none'
        self.limits = (-3.0, 3.0)
        self.field = None
        self._image = None
        self.setFixedSize(104, 104)
        self.setToolTip('')

    # ------------------------------------------------------------------ data
    def set_entry(self, entries, reference_kind='session', limits=(-3.0, 3.0)):
        self.entries = entries if isinstance(entries, dict) else {}
        self.reference_kind = reference_kind
        self.limits = tuple(limits) if limits else (-3.0, 3.0)
        self.field = self._interpolate()
        self._image = self._build_image()
        self.setToolTip(self._tooltip())
        self.update()

    def _interpolate(self):
        pairs = _cell_entries(self.entries)
        if len(pairs) < 3:
            return None
        names, values = [], []
        seen = set()
        for name, value, _key in pairs:
            if name in seen or name not in topomap.ELECTRODE_POSITIONS:
                continue
            seen.add(name)
            names.append(name)
            values.append(value)
        if len(names) < 3:
            return None
        try:
            return topomap.interpolate_topomap(values, names, grid_n=56)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            return None

    def _build_image(self):
        if self.field is None or self.reference_kind == 'none':
            return None
        grid = np.asarray(self.field, dtype=float)
        size = grid.shape[0]
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        low, high = self.limits
        for iy in range(size):
            for ix in range(size):
                z = grid[iy, ix]
                if not np.isfinite(z):
                    continue
                if z < low:
                    z = low
                elif z > high:
                    z = high
                image.setPixelColor(ix, iy, QColor(qeeg.color_for_z(float(z), self.limits)))
        return image

    def _tooltip(self):
        label = f'{ROW_LABELS.get(self.row, self.row)} · {BAND_LABELS.get(self.band, self.band)}'
        unit = ROW_UNITS.get(self.row, '')
        lines = [label, 'Schematic head — engineering interpolation, not source localisation.']
        zs = []
        for name, entry in (self.entries or {}).items():
            if not isinstance(entry, dict):
                continue
            value = _finite(entry.get('value'))
            z = _finite(entry.get('z'))
            if z is not None:
                zs.append(z)
            lines.append(f'{name}: raw {_fmt(value, 3)} {unit}'.rstrip()
                         + (f' · z {z:+.2f}' if z is not None else ' · z —'))
        if zs:
            lines.insert(1, f'this head: z {min(zs):+.2f} … {max(zs):+.2f}')
        elif self.reference_kind == 'none':
            lines.insert(1, 'raw values — no reference selected, so no z colour')
        else:
            lines.insert(1, 'no z is available for this head (reference could not score it)')
        return '\n'.join(lines)

    # --------------------------------------------------------------- painting
    def _head_rect(self):
        side = max(10.0, min(self.width(), self.height()) - 24.0)
        return QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FFFFFF'))
        head = self._head_rect()
        if self.field is None:
            painter.setPen(QPen(QColor('#C9C2B6'), 1.2, Qt.PenStyle.DashLine))
            painter.setBrush(QColor('#FBF7F1'))
            painter.drawEllipse(head)
            painter.setPen(QColor('#6E6961'))
            painter.setFont(QFont('Segoe UI', 8))
            text = 'no data' if not self.entries else 'needs 3 sites'
            painter.drawText(head, Qt.AlignmentFlag.AlignCenter, text)
        elif self.reference_kind == 'none':
            painter.setPen(QPen(QColor('#6E6961'), 1.2))
            painter.setBrush(QColor('#F1EBE2'))
            painter.drawEllipse(head)
            values = [_finite(entry.get('value')) for entry in self.entries.values()
                      if isinstance(entry, dict)]
            values = [value for value in values if value is not None]
            if values:
                painter.setFont(QFont('Segoe UI', 7))
                painter.setPen(QColor('#1A1A1A'))
                painter.drawText(head, Qt.AlignmentFlag.AlignCenter,
                                 f'{np.mean(values):.2f}'.rstrip('0').rstrip('.'))
        else:
            path = QPainterPath()
            path.addEllipse(head)
            painter.save()
            painter.setClipPath(path)
            if self._image is not None:
                painter.drawImage(head, self._image)
            else:
                painter.fillRect(head, QColor('#F5F1EA'))
            painter.restore()
            if self._image is None:
                painter.setPen(QColor('#6E6961'))
                painter.setFont(QFont('Segoe UI', 7))
                painter.drawText(head, Qt.AlignmentFlag.AlignCenter, 'no z')
        # Shared schematic outline: head circle, nose, ears and chin line.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor('#7C7468'), 1.1))
        cx, cy, half = head.center().x(), head.center().y(), head.width() / 2.0
        points = [self._map(px, py, cx, cy, half) for px, py in zip(*topomap.head_outline(121))]
        path = QPainterPath()
        path.moveTo(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.drawPath(path)
        painter.drawPolyline([self._map(-0.22, 0.97, cx, cy, half),
                              self._map(0.0, 1.14, cx, cy, half),
                              self._map(0.22, 0.97, cx, cy, half)])
        for side in (-1, 1):
            painter.drawPolyline([self._map(side * 1.0, -0.22, cx, cy, half),
                                  self._map(side * 1.12, -0.12, cx, cy, half),
                                  self._map(side * 1.12, 0.12, cx, cy, half),
                                  self._map(side * 1.0, 0.22, cx, cy, half)])
        painter.drawLine(self._map(-0.62, -1.06, cx, cy, half),
                         self._map(0.62, -1.06, cx, cy, half))
        painter.end()

    @staticmethod
    def _map(x, y, cx, cy, half):
        return QPointF(cx + x * half, cy - y * half)


class QEEGSheet(QFrame):
    """The printable sheet: title, both axis label sets, 25 heads, colour bar, footer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('QEEGSheet')
        self.setStyleSheet('#QEEGSheet { background:#FFFFFF; }')
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor('#FFFFFF'))
        self.setPalette(palette)
        self.setMinimumWidth(760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 12)
        layout.setSpacing(6)
        self.title = QLabel('qEEG grid · z-scored band summary · 5 rows × 5 bands')
        self.title.setStyleSheet('font-size:17px; font-weight:700; color:#1A1A1A; background:#FFFFFF;')
        layout.addWidget(self.title)
        self.subtitle = QLabel('No recording yet.')
        self.subtitle.setStyleSheet('color:#6E6961; font-size:10px; background:#FFFFFF;')
        self.subtitle.setWordWrap(True)
        layout.addWidget(self.subtitle)

        self.grid_host = QWidget()
        self.grid_host.setStyleSheet('background:#FFFFFF;')
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(1)
        self.grid.setVerticalSpacing(1)
        self.cells = {}
        self.row_labels = {}
        self.band_labels = {}
        corner = QLabel('')
        self.grid.addWidget(corner, 0, 0)
        for column, band in enumerate(BAND_KEYS, start=1):
            label = QLabel(BAND_LABELS[band])
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('font-weight:600; color:#1A1A1A; font-size:11px; '
                                'background:#FFFFFF;')
            self.band_labels[band] = label
            self.grid.addWidget(label, 0, column)
        row_tips = {
            'absolute': 'How much energy is in that band at each site (µV²).',
            'relative': "That band's share of the 1–45 Hz total at each site (%).",
            'asymmetry': 'Left minus right homologous-pair difference (%), mirrored onto both '
                         'sites for the picture; the tooltip names the pair.',
            'coherence': "Each electrode's mean magnitude-squared coherence over its pairs "
                         "(0–1, documented in qeeg.py).",
            'phase': "Each electrode's mean signed phase offset over its pairs (degrees).",
        }
        for index, row in enumerate(ROW_KEYS, start=1):
            label = QLabel(ROW_LABELS[row])
            label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            label.setWordWrap(True)
            label.setMinimumWidth(104)
            label.setMaximumWidth(120)
            label.setToolTip(row_tips.get(row, ROW_LABELS[row]))
            label.setStyleSheet('font-size:11px; color:#1A1A1A; background:#FFFFFF;')
            self.row_labels[row] = label
            self.grid.addWidget(label, index, 0)
            for column, band in enumerate(BAND_KEYS, start=1):
                cell = TopoCell(row, band)
                self.cells[(row, band)] = cell
                self.grid.addWidget(cell, index, column)
        self.color_bar = ZColorBar()
        self.grid.addWidget(self.color_bar, 1, len(BAND_KEYS) + 1,
                            len(ROW_KEYS), 1)
        self.grid_host.setMaximumWidth(1080)
        layout.addWidget(self.grid_host, 0, Qt.AlignmentFlag.AlignHCenter)

        self.sheet_reference_label = QLabel('Reference — not computed yet.')
        self.sheet_reference_label.setWordWrap(True)
        self.sheet_reference_label.setMaximumWidth(1070)
        self.sheet_reference_label.setStyleSheet('font-size:10px; color:#1A1A1A; '
                                                  'background:#FFFFFF;')
        layout.addWidget(self.sheet_reference_label)
        self.note = QLabel(
            'Schematic 10-20 head outlines and engineering interpolation only — not source '
            'localisation, not a clinical brain map. Reading method after Dr. Trish Leigh '
            '(Leigh Brain & Spine), educational only. ' + REPORT_NOTE)
        self.note.setWordWrap(True)
        self.note.setMaximumWidth(1070)
        self.note.setStyleSheet('font-size:9px; color:#6E6961; background:#FFFFFF;')
        layout.addWidget(self.note)

    def visible_cells(self):
        return [cell for cell in self.cells.values() if cell.isVisible()]


class QEEGPanel(QWidget):
    """Toolbar + sheet + "How to read this" panel.

    ``data_provider`` returns ``None`` or a dict with ``samples`` (n, ch in µV),
    ``fs``, ``channels`` and ``session_key``. ``baseline_provider`` returns
    ``None`` or ``{'metrics': ..., 'label': ..., 'n': ...}`` for the current
    subject's saved baseline. Both are called only from :meth:`refresh`.
    """

    def __init__(self, data_provider=None, baseline_provider=None, parent=None):
        super().__init__(parent)
        self.data_provider = data_provider
        self.baseline_provider = baseline_provider
        self._cache = {}
        self._grid = None
        self._metrics = None
        self._reference = {'kind': 'none', 'label': '', 'n': 0}
        self._notice = ''
        self.norm_reference = None
        self.norm_path = ''
        self.last_error = ''
        self._build_ui()

    # ------------------------------------------------------------------- UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(7)

        header = QHBoxLayout()
        title = QLabel('qEEG grid · 5 rows × 5 bands · z-scored schematic heads')
        title.setStyleSheet('font-size:13px; font-weight:700;')
        header.addWidget(title)
        header.addStretch()
        self.guide_toggle = QToolButton()
        self.guide_toggle.setText('How to read this')
        self.guide_toggle.setCheckable(True)
        self.guide_toggle.setChecked(False)
        self.guide_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.guide_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        header.addWidget(self.guide_toggle)
        layout.addLayout(header)

        guide_host = QWidget()
        guide_layout = QVBoxLayout(guide_host)
        guide_layout.setContentsMargins(0, 0, 0, 0)
        self.guide_text = QLabel(GUIDE_TEXT)
        self.guide_text.setWordWrap(True)
        self.guide_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.guide_text.setStyleSheet('background:#FBF7F1; border:1px solid #E9E2D8; '
                                      'padding:9px; font-size:10px; color:#1A1A1A;')
        guide_layout.addWidget(self.guide_text)
        self.guide_panel = guide_host
        self.guide_panel.setVisible(False)
        layout.addWidget(self.guide_panel)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel('Reference'))
        self.reference_combo = QComboBox()
        self.reference_combo.addItems([text for text, _kind in REFERENCE_CHOICES])
        self.reference_combo.setToolTip(
            'What the z colours compare against. "This session" is self-referenced: the mean '
            'and SD of the channels in this recording, not an age-matched database.')
        toolbar.addWidget(self.reference_combo)
        condition_label = QLabel('Condition *')
        condition_label.setToolTip('Required: eyes closed / eyes open / other. The condition is '
                                   'stored with the sheet and printed with its reference.')
        toolbar.addWidget(condition_label)
        self.condition_combo = QComboBox()
        self.condition_combo.addItems(['eyes closed', 'eyes open', 'other'])
        self.condition_combo.setToolTip('Required: the measurement condition. Eyes closed and '
                                        'eyes open are not comparable with each other.')
        toolbar.addWidget(self.condition_combo)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.setToolTip('Recompute this sheet once (not on the 1 s analysis timer).')
        toolbar.addWidget(self.refresh_button)
        toolbar.addStretch()
        self.export_button = QPushButton('Export sheet PNG…')
        self.report_button = QPushButton('Add to patient report')
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.report_button)
        layout.addLayout(toolbar)

        chips = QHBoxLayout()
        chips.addWidget(QLabel('Bands'))
        self.band_checks = OrderedDict()
        for band in BAND_KEYS:
            box = QCheckBox(BAND_LABELS[band])
            box.setChecked(True)
            box.setToolTip(f'Show the {BAND_LABELS[band]} column.')
            self.band_checks[band] = box
            chips.addWidget(box)
        chips.addSpacing(12)
        chips.addWidget(QLabel('Rows'))
        self.row_checks = OrderedDict()
        for row in ROW_KEYS:
            box = QCheckBox(ROW_LABELS[row])
            box.setChecked(True)
            box.setToolTip(f'Show the {ROW_LABELS[row]} row.')
            self.row_checks[row] = box
            chips.addWidget(box)
        chips.addStretch()
        layout.addLayout(chips)

        self.stack = QStackedWidget()
        empty_page = QWidget()
        empty_layout = QVBoxLayout(empty_page)
        empty_layout.addStretch()
        self.empty_label = QLabel(EMPTY_NO_DATA)
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setStyleSheet('color:#6E6961; font-size:12px; padding:28px;')
        empty_layout.addWidget(self.empty_label)
        empty_layout.addStretch()
        self.stack.addWidget(empty_page)
        self.empty_page = empty_page

        self.sheet = QEEGSheet()
        self.sheet.setMaximumWidth(1220)
        scroller = QScrollArea()
        scroller.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        scroller.setWidgetResizable(True)
        scroller.setWidget(self.sheet)
        self.stack.addWidget(scroller)
        self.sheet_page = scroller
        layout.addWidget(self.stack, 1)

        self.reference_label = QLabel('Reference — not computed yet.')
        self.reference_label.setWordWrap(True)
        self.reference_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.reference_label.setStyleSheet('background:#FBF7F1; border:1px solid #E9E2D8; '
                                           'padding:7px; font-size:10px; color:#1A1A1A;')
        layout.addWidget(self.reference_label)
        self.status_label = QLabel('Open this tab or press Refresh to compute the sheet once.')
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet('color:#6E6961; font-size:10px;')
        layout.addWidget(self.status_label)

        self.guide_toggle.toggled.connect(self._toggle_guide)
        self.refresh_button.clicked.connect(lambda: self.refresh(force=True))
        self.report_button.clicked.connect(self.add_to_report_clicked)
        self.export_button.clicked.connect(self.export_clicked)
        self.reference_combo.currentIndexChanged.connect(self._reference_changed)
        self.reference_combo.activated.connect(self._reference_activated)
        for box in list(self.band_checks.values()) + list(self.row_checks.values()):
            box.toggled.connect(self._filters_changed)
        self.condition_combo.currentIndexChanged.connect(self._condition_changed)

    # --------------------------------------------------------------- helpers
    def _toggle_guide(self):
        shown = self.guide_toggle.isChecked()
        self.guide_panel.setVisible(shown)
        self.guide_toggle.setArrowType(Qt.ArrowType.DownArrow if shown
                                       else Qt.ArrowType.RightArrow)

    def reference_kind(self):
        index = self.reference_combo.currentIndex()
        return REFERENCE_CHOICES[max(0, min(index, len(REFERENCE_CHOICES) - 1))][1]

    def condition(self):
        return self.condition_combo.currentText()

    def selected_bands(self):
        return tuple(band for band, box in self.band_checks.items() if box.isChecked())

    def selected_rows(self):
        return tuple(row for row, box in self.row_checks.items() if box.isChecked())

    def _reference_changed(self, _index=None):
        """Selection changed: refresh from cache (norm file needs the dialog)."""
        if self.reference_kind() == 'norm' and self.norm_reference is None:
            self.norm_reference = None
            self._notice = ('No local norm file loaded. Choose "Load norm file…" and pick a '
                            'reference JSON with >= 20 subjects; until then the sheet stays raw.')
        self.refresh(force=False)

    def _reference_activated(self, _index):
        if self.reference_kind() == 'norm':
            self.load_norm_file()

    def _condition_changed(self, _index=None):
        # A condition change changes the reference key (and the printed line);
        # no recomputation of the metrics themselves is required.
        self.refresh(force=False)

    def _filters_changed(self, _checked=None):
        self.refresh(force=False)

    def load_norm_file(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, 'Load a local norm reference (JSON)', self.norm_path or '',
            'Norm reference (*.json);;All files (*)')
        if not path:
            return False
        try:
            self.norm_reference = qeeg.load_reference(path)
            self.norm_path = path
            self._notice = ''
        except Exception as exc:              # bad file: say so, never fake it
            self.norm_reference = None
            self.last_error = str(exc)
            self._notice = f'Norm file not usable: {exc}'
            self.status_label.setText(self._notice)
            return False
        self.refresh(force=True)
        return True

    # ---------------------------------------------------------------- compute
    def _resolve_reference(self, metrics):
        kind = self.reference_kind()
        if kind == 'session':
            return qeeg.reference_from_session(metrics), ''
        if kind == 'baseline':
            try:
                base = self.baseline_provider() if self.baseline_provider else None
            except Exception:
                base = None
            if not base or not base.get('metrics'):
                return (self._raw_reference('no saved baseline for this subject'),
                        'Saved baseline selected but none is available — showing raw values.')
            if base.get('condition') and base.get('condition') != self.condition():
                return (self._raw_reference('baseline condition does not match'),
                        f'The saved baseline is for {base.get("condition")!r} but the current '
                        f'condition is {self.condition()!r}. Conditions are not comparable, so '
                        'the baseline is not used — showing raw values.')
            ref = qeeg.reference_from_baseline(base['metrics'], base.get('label') or 'saved baseline',
                                               base.get('n'))
            return ref, ''
        if kind == 'norm':
            if self.norm_reference is None:
                return (self._raw_reference('no local norm file loaded'),
                        self._notice or 'No local norm file is loaded — showing raw values.')
            return self.norm_reference, ''
        return self._raw_reference('raw values'), ''

    @staticmethod
    def _raw_reference(reason):
        return {'kind': 'none', 'label': reason, 'n': 0}

    def _cache_key(self, data):
        return (str(data.get('session_key', '')), self.selected_bands(),
                self.reference_kind(), getattr(self, 'norm_path', ''),
                self.condition(), tuple(data.get('channels') or ()))

    def refresh(self, force=False):
        """Compute (or reuse) the grid and draw it. Never call from a timer."""
        if self.data_provider is None:
            self._show_empty(EMPTY_NO_DATA)
            return
        try:
            data = self.data_provider()
        except Exception as exc:
            self._show_empty(f'Could not read the recording buffer: {exc}')
            return
        if not data:
            self._show_empty(EMPTY_NO_DATA)
            return
        channels = tuple(data.get('channels') or ())
        if len(channels) < 3:
            self._show_empty(EMPTY_FEW_CHANNELS.format(count=len(channels)))
            return
        samples = np.asarray(data.get('samples'), dtype=float)
        if samples.ndim != 2 or samples.shape[0] < 64 or samples.shape[1] != len(channels):
            self._show_empty(EMPTY_NO_DATA)
            return
        fs = float(data.get('fs') or 250.0)
        key = self._cache_key(data)
        cached = None if force else self._cache.get(key)
        if cached is None:
            try:
                metrics = qeeg.compute_metrics(samples, fs, list(channels))
            except Exception as exc:
                self.last_error = str(exc)
                self._show_empty(f'The qEEG grid could not be computed: {exc}')
                return
            if len(metrics.get('usable_channels') or []) < 3:
                detail = ' '.join(str(note) for note in (metrics.get('notes') or []))
                self._show_empty('No usable scalp channels for the grid. ' + detail)
                return
            try:
                reference, notice = self._resolve_reference(metrics)
                grid = qeeg.grid(metrics, reference)
            except Exception as exc:
                # A broken/unreadable reference must not hide the measured grid:
                # fall back to raw values and say so.
                self.last_error = str(exc)
                reference = self._raw_reference(f'reference not usable: {exc}')
                notice = (f'The selected reference could not be built ({exc}) — showing raw '
                          'values.')
                grid = qeeg.grid(metrics, reference)
            cached = {'metrics': metrics, 'grid': grid, 'notice': notice}
            self._cache[key] = cached
            if len(self._cache) > 12:                 # bounded: one slot per session/ref
                for stale in list(self._cache)[:-12]:
                    self._cache.pop(stale, None)
        self._metrics = cached['metrics']
        self._grid = cached['grid']
        self._reference = self._grid.get('reference') or {'kind': 'none', 'label': '', 'n': 0}
        self._notice = cached.get('notice') or ''
        self._render(data)

    def _render(self, data):
        grid = self._grid or {}
        rows = grid.get('rows') or {}
        ref = self._reference or {}
        kind = str(ref.get('kind') or 'none')
        limits = (grid.get('limits') or {}).get('z') or [-3, 3]
        metrics = self._metrics or {}
        channel_count = len(metrics.get('channels') or [])
        self.sheet.color_bar.set_limits(limits)
        self.sheet.subtitle.setText(
            f'Condition: {self.condition()} · {channel_count} scalp '
            f'channels · {self._fmt_window()} of data · z scale fixed −3 … +3')
        for row in ROW_KEYS:
            row_data = rows.get(row) or {}
            for band in BAND_KEYS:
                entries = row_data.get(band) or {}
                self.sheet.cells[(row, band)].set_entry(entries, kind, limits)
        for band, box in self.band_checks.items():
            self.sheet.band_labels[band].setVisible(box.isChecked())
            for row in ROW_KEYS:
                self.sheet.cells[(row, band)].setVisible(box.isChecked() and
                                                         self.row_checks[row].isChecked())
        for row, box in self.row_checks.items():
            self.sheet.row_labels[row].setVisible(box.isChecked())
        line = describe_reference(ref)
        self.reference_label.setText(line)
        self.sheet.sheet_reference_label.setText(line)
        self.sheet.sheet_reference_label.setToolTip(
            'kind / label / n exactly as returned by qeeg.grid — the reference is printed as '
            'measured, never upgraded to a licensed clinical norm set.')
        self.stack.setCurrentWidget(self.sheet_page)
        notes = [str(note) for note in (grid.get('notes') or [])
                 if not str(note).startswith('Reference:')]   # footer already prints it
        status = f'Computed {self._fmt_window()} of data · reference n={ref.get("n", "—")}'
        if notes:
            status += ' · ' + ' · '.join(str(note) for note in notes)
        if self._notice:
            status += ' · ' + self._notice
        self.status_label.setText(status)

    def _fmt_window(self):
        if not self._metrics:
            return '—'
        seconds = _finite(self._metrics.get('window_s'))
        if seconds is None and self._metrics.get('n_samples') and self._metrics.get('fs'):
            seconds = self._metrics['n_samples'] / self._metrics['fs']
        return f'{seconds:.1f} s' if seconds is not None else '—'

    def _show_empty(self, message):
        self.empty_label.setText(message)
        self.stack.setCurrentWidget(self.empty_page)
        self.status_label.setText(message)
        self._grid = None                 # never let a stale sheet be exported
        self._metrics = None
        self.reference_label.setText('Reference — not computed (no data). ' + NOT_NORMATIVE)
        self.sheet.sheet_reference_label.setText('Reference — not computed (no data). '
                                                 + NOT_NORMATIVE)

    # ----------------------------------------------------------------- export
    def sheet_pixmap(self, scale=2.0):
        """Rendered sheet at ``scale`` device pixels per logical pixel, white background.

        The live sheet is sized by its scroll area, so it is rendered at that
        width rather than being resized (a QScrollArea would override the
        resize and the right-hand columns would be clipped).
        """
        self.sheet.layout().activate()
        width = max(self.sheet.width(), 1080)
        # Render to the layout's own content height: the live sheet stretches to
        # fill the scroll area, which would otherwise export a tall white band.
        try:
            content_height = self.sheet.layout().totalMinimumSize().height()
        except Exception:
            content_height = self.sheet.height()
        height = max(int(content_height), 640)
        pixmap = QPixmap(int(width * scale), int(height * scale))
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.scale(scale, scale)
        self.sheet.render(painter, QPoint(0, 0))
        painter.end()
        return pixmap

    def sheet_png_bytes(self):
        from PyQt6.QtCore import QBuffer, QIODevice
        pixmap = self.sheet_pixmap()
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        if not pixmap.save(buffer, 'PNG'):
            raise RuntimeError('Qt could not encode the qEEG sheet as PNG')
        return bytes(buffer.data())

    def export_sheet_png(self, path):
        """Write the full sheet PNG and return the path. Raises on failure."""
        from pathlib import Path
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not self.sheet_pixmap().save(str(target), 'PNG'):
            raise RuntimeError(f'Could not write the qEEG sheet PNG to {target}')
        return target

    def export_clicked(self):
        from datetime import datetime
        if not self.has_grid():
            self.status_label.setText('Nothing to export yet — open this tab with a recording '
                                      'running, then press Refresh.')
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, 'Export the qEEG sheet as PNG',
            f'qeeg_grid_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 'PNG image (*.png)')
        if not path:
            return
        try:
            written = self.export_sheet_png(path)
        except (OSError, RuntimeError) as exc:
            self.status_label.setText(f'Export failed: {exc}')
            return
        self.status_label.setText(f'Sheet written: {written}')

    def add_to_report_clicked(self):
        handler = getattr(self, 'report_handler', None)
        if handler is None:
            self.status_label.setText('No patient report is open in this window.')
            return
        handler()

    def has_grid(self):
        """True when a computed grid is currently displayed (for the report button)."""
        return self._grid is not None

    def report_metadata(self):
        ref = self._reference or {}
        return {
            'reference_line': describe_reference(ref),
            'reference_kind': ref.get('kind'),
            'reference_label': ref.get('label'),
            'reference_n': ref.get('n'),
            'condition': self.condition(),
            'bands': list(self.selected_bands()),
            'rows': list(self.selected_rows()),
            'channels': list((self._metrics or {}).get('channels') or []),
            'note': REPORT_NOTE,
        }

    def describe_reference(self):
        return describe_reference(self._reference)
