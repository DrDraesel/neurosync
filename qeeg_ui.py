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

Reading aids (``docs/QEEG-NORMS-SPEC.md``): under the grid a theta/beta card
(per-channel ratio, channel mean, teaching note, flag only when the mean is
>= 3.0 and both bands are usable) and a region roll-up from ``regions``
(``band × region`` mean z, flagged at |z| >= 2 with >= 2 sites). A visible
raw-trace reminder sits above the grid. The **Compare EC/EO…** button pairs
two saved sessions of the same subject, refuses different devices or reference
kinds with a visible message, and draws both sheets plus an ``EO − EC`` strip.
A norm file is loaded through ``norms.load`` when available; its source label,
n, age band and any device-scope warning are printed verbatim, and an imported
file is never called local.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                         QPixmap)
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QSizePolicy, QStackedWidget,
                             QToolButton, QVBoxLayout, QWidget)

import qeeg
import topomap

# Optional wave-2 modules (docs/QEEG-NORMS-SPEC.md). The panel must keep
# working without them: the region table hides itself and the norm picker
# falls back to the plain qeeg reference format.
try:                                        # pragma: no cover - import shape
    import norms
except Exception:                           # noqa: BLE001 - optional module
    norms = None
try:                                        # pragma: no cover - import shape
    import regions
except Exception:                           # noqa: BLE001 - optional module
    regions = None

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
RAW_TRACE_REMINDER = ('Check the raw traces before trusting a map — artifact looks like a region.')
TBR_NOTE = ('A theta/beta ratio near 3:1 is the pattern Dr. Leigh names for ADHD — one number '
            'is not a diagnosis; read it with the rest of the sheet.')
TBR_FLAG_MEAN = 3.0
REGION_FLAG_Z = 2.0
REGION_MIN_SITES = 2
NORM_SCOPE_WARNING = 'device scope mismatch'
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


def theta_beta_values(metrics):
    """Per-channel Theta/Beta ratios, the channel mean and a usability flag.

    The ratio is ``Theta absolute / Beta absolute`` for each usable channel,
    exactly the numerator/denominator of ``qeeg.ratios``; the mean is the mean
    of those per-channel ratios (not the ratio of the means). ``usable`` is
    True only when a ratio exists for every usable channel, because the 3:1
    flag is only meaningful when both bands are usable everywhere.

    Returns ``(ratios: dict, mean: float|None, usable: bool)``.
    """
    metrics = metrics if isinstance(metrics, dict) else {}
    bands = metrics.get('bands') or {}
    channels = list(metrics.get('usable_channels') or ())
    theta = (bands.get('Theta') or {}).get('absolute') or {}
    beta = (bands.get('Beta') or {}).get('absolute') or {}
    ratios = OrderedDict()
    for name in channels:
        theta_value = _finite(theta.get(name))
        beta_value = _finite(beta.get(name))
        if theta_value is None or beta_value is None or beta_value <= 0:
            continue
        ratios[name] = theta_value / beta_value
    mean = float(np.mean(list(ratios.values()))) if ratios else None
    usable = bool(channels) and len(ratios) == len(channels)
    return ratios, mean, usable


def tbr_is_flagged(mean, usable):
    """True only when both bands are usable and the channel mean is >= 3.0."""
    number = _finite(mean)
    return bool(usable) and number is not None and number >= TBR_FLAG_MEAN


def _region_display(name):
    return str(name).replace('_', ' ')


def _region_cell(entry):
    """(mean z, n_sites, flag) from one region_table cell, whatever its shape."""
    if isinstance(entry, dict):
        value = None
        for key in ('mean_z', 'z', 'mean', 'value'):
            value = _finite(entry.get(key))
            if value is not None:
                break
        sites = entry.get('n_sites')
        if sites is None:
            sites = entry.get('sites')
        if isinstance(sites, (list, tuple, set)):
            sites = len(sites)
        try:
            sites = int(sites) if sites is not None else None
        except (TypeError, ValueError):
            sites = None
        flag = entry.get('flag')
        if flag is None:
            flag = entry.get('flagged')
        if isinstance(flag, str):
            flag = flag.strip().lower() not in ('', 'none', 'false', '0')
        return value, sites, (bool(flag) if flag is not None else None)
    value = _finite(entry)
    return value, None, None


def normalise_region_table(table):
    """``{(band, region): (mean_z, n_sites|None, flag|None)}`` from region_table().

    ``regions.region_table`` is specified to return, per band, the mean z of
    each of the ten regions plus a flag when |z| >= 2 with at least two sites.
    The exact container is not pinned down by the spec, so this accepts a
    ``{band: {region: cell}}`` mapping, a ``{(band, region): cell}`` mapping or
    a list of records with ``band``/``region`` keys. Nothing is invented: a
    region with no site count reported keeps ``None`` and is never flagged
    here (the flag stays whatever the module returned).
    """
    cells = OrderedDict()
    if table is None:
        return cells
    if isinstance(table, dict):
        for outer, inner in table.items():
            if isinstance(inner, dict) and not any(
                    key in inner for key in ('mean_z', 'z', 'mean', 'value', 'flag')):
                band = str(outer)
                for region, entry in inner.items():
                    value, sites, flag = _region_cell(entry)
                    if value is None:
                        continue
                    cells[(band, str(region))] = (value, sites, flag)
            else:
                if isinstance(outer, tuple) and len(outer) == 2:
                    band, region = str(outer[0]), str(outer[1])
                    value, sites, flag = _region_cell(inner)
                    if value is not None:
                        cells[(band, region)] = (value, sites, flag)
                elif isinstance(outer, str) and isinstance(inner, tuple) \
                        and len(inner) == 2 and isinstance(inner[0], str):
                    value, sites, flag = _region_cell(inner[1])
                    if value is not None:
                        cells[(str(inner[0]), outer)] = (value, sites, flag)
    elif isinstance(table, (list, tuple)):
        for record in table:
            if not isinstance(record, dict):
                continue
            band = record.get('band') or record.get('row_band')
            region = record.get('region') or record.get('name')
            if band is None or region is None:
                continue
            value, sites, flag = _region_cell(record)
            if value is None:
                continue
            cells[(str(band), str(region))] = (value, sites, flag)
    return cells


def region_flag(cell):
    """True when the module flagged the cell or |z| >= 2 with a known site count >= 2."""
    value, sites, flag = cell
    if flag is not None:
        return bool(flag)
    return (_finite(value) is not None and abs(value) >= REGION_FLAG_Z
            and sites is not None and sites >= REGION_MIN_SITES)


def region_columns(cells):
    """Region names ordered by their largest |z| (strongest first)."""
    strongest = {}
    for (_band, region), (value, _sites, _flag) in cells.items():
        magnitude = abs(value)
        if magnitude > strongest.get(region, -1.0):
            strongest[region] = magnitude
    return [name for name in sorted(strongest, key=lambda name: (-strongest[name], name))]


def region_band_order(cells):
    """Bands present in the table, in the sheet's left-to-right band order."""
    present = {band for band, _region in cells}
    ordered = [band for band in BAND_KEYS if band in present]
    ordered += sorted(band for band in present if band not in BAND_KEYS)
    return ordered


def describe_reference(reference):
    """One always-printed line with kind, label, n and the disclaimer.

    Only ``kind == 'norm'`` may be described as a local norm; every other kind
    gets a plain description of the comparison. An imported reference file
    (``source`` other than ``local``) is named by its source, never relabelled
    as local. The words "NeuroGuide" and "normative" (outside the fixed
    disclaimer) never appear here unless a source string itself says so.
    """
    ref = reference if isinstance(reference, dict) else {}
    kind = str(ref.get('kind') or 'none')
    label = str(ref.get('label') or '').strip()
    n = ref.get('n')
    source = str(ref.get('source') or '').strip()
    age_band = str(ref.get('age_band') or '').strip()
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
        if source and source.lower() != 'local':
            text = f'kind: norm · vs imported reference ({source})'
            if n not in (None, ''):
                text += f', n={n}'
            if age_band:
                text += f', age {age_band}'
        else:
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
    if kind not in ('norm',) and n not in (None, '', 0):
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
        self.setFixedSize(96, 96)
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


class ThetaBetaCard(QFrame):
    """Theta/Beta strip: one ratio per channel, the channel mean and the teaching note.

    The note is the source teaching verbatim and is always printed; the flag is
    a single amber line that appears only when the channel mean is >= 3.0 and
    both bands are usable for every channel. Nothing here is a diagnosis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('ThetaBetaCard')
        self.setStyleSheet('#ThetaBetaCard { background:#FFFFFF; border-top:1px solid #E9E2D8; }')
        self.ratios = OrderedDict()
        self.mean_value = None
        self.usable = False
        self.flagged = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 2)
        layout.setSpacing(4)
        head = QHBoxLayout()
        title = QLabel('Theta / Beta · θ absolute power ÷ β absolute power, per channel')
        title.setStyleSheet('font-size:12px; font-weight:700; color:#1A1A1A; background:#FFFFFF;')
        head.addWidget(title)
        head.addStretch()
        self.mean_label = QLabel('Channel mean θ/β —')
        self.mean_label.setStyleSheet('font-size:12px; font-weight:700; color:#1A1A1A; '
                                      'background:#FFFFFF;')
        head.addWidget(self.mean_label)
        layout.addLayout(head)
        self.channel_host = QWidget()
        self.channel_host.setStyleSheet('background:#FFFFFF;')
        self.channel_grid = QGridLayout(self.channel_host)
        self.channel_grid.setContentsMargins(0, 2, 0, 2)
        self.channel_grid.setHorizontalSpacing(14)
        self.channel_grid.setVerticalSpacing(1)
        self.channel_labels = OrderedDict()
        layout.addWidget(self.channel_host)
        self.flag_label = QLabel('')
        self.flag_label.setWordWrap(True)
        self.flag_label.setStyleSheet('background:#FFF3D6; color:#7A4B00; border:1px solid #E0B25A; '
                                      'padding:4px 7px; font-size:10px; font-weight:600;')
        self.flag_label.setVisible(False)
        layout.addWidget(self.flag_label)
        self.note = QLabel(TBR_NOTE)
        self.note.setWordWrap(True)
        self.note.setStyleSheet('font-size:10px; color:#6E6961; background:#FFFFFF;')
        layout.addWidget(self.note)

    def set_empty(self, message=''):
        self.ratios = OrderedDict()
        self.mean_value = None
        self.usable = False
        self.flagged = False
        for name, label in self.channel_labels.items():
            label.setText(f'{name} —')
        self.mean_label.setText('Channel mean θ/β —')
        self.flag_label.setVisible(False)

    def set_metrics(self, metrics):
        ratios, mean, usable = theta_beta_values(metrics)
        self.ratios, self.mean_value, self.usable = ratios, mean, usable
        self.flagged = tbr_is_flagged(mean, usable)
        names = list((metrics or {}).get('usable_channels') or ()) or list(ratios)
        for stale in [name for name in self.channel_labels if name not in names]:
            label = self.channel_labels.pop(stale)
            label.setParent(None)
            label.deleteLater()
        for name in names:
            if name not in self.channel_labels:
                label = QLabel(f'{name} —')
                label.setStyleSheet('font-size:10px; color:#1A1A1A; background:#FFFFFF;')
                self.channel_labels[name] = label
                index = len(self.channel_labels) - 1
                self.channel_grid.addWidget(label, index // 7, index % 7)
        for name, label in self.channel_labels.items():
            value = ratios.get(name)
            label.setText(f'{name} —' if value is None else f'{name} {value:.2f}')
        self.mean_label.setText('Channel mean θ/β —' if mean is None
                                else f'Channel mean θ/β {mean:.2f}')
        if self.flagged:
            self.flag_label.setText(
                f'θ/β mean {mean:.2f} ≥ {TBR_FLAG_MEAN:.1f} — the pattern Dr. Leigh describes, '
                'not a diagnosis. Read it with the rest of the sheet.')
        self.flag_label.setVisible(self.flagged)
        return self.flagged


class RegionTable(QWidget):
    """Compact band × region table of mean z, flagged at |z| >= 2 with >= 2 sites.

    The table is built from ``regions.region_table(grid)``. When that module is
    not installed the widget hides itself and says so in the patient-report
    metadata — it never fills a region from neighbours or invents a site.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cells_map = OrderedDict()
        self.region_order = []
        self.band_order = []
        self.cell_labels = OrderedDict()
        self.flag_count = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(4)
        head = QHBoxLayout()
        self.title = QLabel('Region roll-up · mean z by region (|z| ≥ 2 with ≥ 2 sites flagged)')
        self.title.setStyleSheet('font-size:12px; font-weight:700; color:#1A1A1A; background:#FFFFFF;')
        head.addWidget(self.title)
        head.addStretch()
        self.flag_label = QLabel('')
        self.flag_label.setStyleSheet('font-size:10px; color:#7A4B00; background:#FFFFFF;')
        head.addWidget(self.flag_label)
        layout.addLayout(head)
        self.table_host = QWidget()
        self.table_host.setStyleSheet('background:#FFFFFF;')
        self.grid = QGridLayout(self.table_host)
        self.grid.setContentsMargins(0, 2, 0, 2)
        self.grid.setHorizontalSpacing(3)
        self.grid.setVerticalSpacing(2)
        layout.addWidget(self.table_host)
        self.note = QLabel('Regions with fewer than two available sites are omitted, never '
                           'filled from neighbours. Sorted by |z|.')
        self.note.setStyleSheet('font-size:9px; color:#6E6961; background:#FFFFFF;')
        layout.addWidget(self.note)

    def _clear(self):
        self.cell_labels = OrderedDict()
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def set_table(self, table, bands=None, row=None):
        """Draw ``table``; returns False (and hides) when there is nothing usable."""
        cells = normalise_region_table(table)
        if bands is not None:
            keep = {str(band) for band in bands}
            cells = OrderedDict((key, value) for key, value in cells.items() if key[0] in keep)
        self.cells_map = cells
        self._clear()
        if row:
            self.title.setText(f'Region roll-up · mean z · row: {ROW_LABELS.get(row, row)} '
                               '(|z| ≥ 2 with ≥ 2 sites flagged)')
        if not cells:
            self.region_order, self.band_order, self.flag_count = [], [], 0
            self.setVisible(False)
            return False
        self.setVisible(True)
        self.region_order = region_columns(cells)
        self.band_order = region_band_order(cells)
        for column, region in enumerate(self.region_order, start=1):
            label = QLabel(_region_display(region))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet('font-size:9px; font-weight:600; color:#1A1A1A; background:#FFFFFF;')
            self.grid.addWidget(label, 0, column)
        self.flag_count = 0
        for row, band in enumerate(self.band_order, start=1):
            band_label = QLabel(BAND_LABELS.get(band, band))
            band_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            band_label.setStyleSheet('font-size:10px; color:#6E6961; background:#FFFFFF;')
            band_label.setMinimumWidth(58)
            self.grid.addWidget(band_label, row, 0)
            for column, region in enumerate(self.region_order, start=1):
                cell = cells.get((band, region))
                text, style, flagged = '—', 'font-size:10px; color:#A9A29A; background:#FFFFFF;', False
                tooltip = f'{BAND_LABELS.get(band, band)} · {_region_display(region)}: no value'
                if cell is not None:
                    value, sites, _flag = cell
                    flagged = region_flag(cell)
                    self.flag_count += 1 if flagged else 0
                    text = f'{value:+.2f}'
                    colour = qeeg.color_for_z(max(-3.0, min(3.0, value)), (-3.0, 3.0))
                    style = (f'font-size:10px; color:#1A1A1A; background:{colour}; '
                             + ('font-weight:800; border:2px solid #7A4B00;' if flagged
                                else 'border:1px solid #FFFFFF;'))
                    site_text = '' if sites is None else f' · {sites} sites'
                    tooltip = (f'{BAND_LABELS.get(band, band)} · {_region_display(region)}: '
                               f'mean z {value:+.2f}{site_text}'
                               + (' · FLAGGED (|z| ≥ 2 with ≥ 2 sites)' if flagged else ''))
                label = QLabel(text)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                label.setMinimumWidth(66)
                label.setStyleSheet(style)
                label.setToolTip(tooltip)
                self.cell_labels[(band, region)] = label
                self.grid.addWidget(label, row, column)
        self.flag_label.setText(f'{self.flag_count} flagged cell(s)' if self.flag_count
                                else 'no flagged region')
        return True



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
        self.title = QLabel('qEEG grid · z-scored band summary · 5 rows × 5 bands  (scroll for all rows · Export for the full sheet)')
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
        # Reading aids under the grid: theta/beta strip, then the region roll-up.
        self.tbr_card = ThetaBetaCard()
        layout.addWidget(self.tbr_card)
        self.region_table = RegionTable()
        self.region_table.setVisible(False)
        layout.addWidget(self.region_table)
        self.note = QLabel(
            'Schematic 10-20 head outlines and engineering interpolation only — not source '
            'localisation, not a clinical brain map. Reading method after Dr. Trish Leigh '
            '(Leigh Brain & Spine), educational only. ' + REPORT_NOTE)
        self.note.setWordWrap(True)
        self.note.setMaximumWidth(1070)
        self.note.setStyleSheet('font-size:9px; color:#6E6961; background:#FFFFFF;')
        layout.addWidget(self.note)

    def apply(self, grid, reference=None, condition='', channel_count=None,
              window_text='', bands=None, rows=None, metrics=None):
        """Draw a computed grid into this sheet (single view and EC/EO view).

        ``bands``/``rows`` are iterables of keys to show; ``None`` means all.
        The reference line is printed from the grid's own reference block, so a
        paired sheet cannot borrow the other session's reference.
        """
        grid = grid if isinstance(grid, dict) else {}
        rows_data = grid.get('rows') or {}
        ref = reference if reference is not None else (grid.get('reference') or {})
        ref = ref if isinstance(ref, dict) else {}
        kind = str(ref.get('kind') or 'none')
        limits = (grid.get('limits') or {}).get('z') or [-3, 3]
        self.color_bar.set_limits(limits)
        count = channel_count
        if count is None:
            count = len(grid.get('channels') or [])
        shown_bands = tuple(bands) if bands is not None else BAND_KEYS
        shown_rows = tuple(rows) if rows is not None else ROW_KEYS
        subtitle = f'Condition: {condition or "not recorded"} · {count} scalp channels'
        if window_text:
            subtitle += f' · {window_text}'
        subtitle += ' · z scale fixed −3 … +3'
        self.subtitle.setText(subtitle)
        for row in ROW_KEYS:
            row_data = rows_data.get(row) or {}
            for band in BAND_KEYS:
                cell = self.cells.get((row, band))
                if cell is None:
                    continue
                cell.set_entry(row_data.get(band) or {}, kind, limits)
                cell.setVisible(band in shown_bands and row in shown_rows)
        for band, label in self.band_labels.items():
            label.setVisible(band in shown_bands)
        for row, label in self.row_labels.items():
            label.setVisible(row in shown_rows)
        line = describe_reference(ref)
        self.sheet_reference_label.setText(line)
        self.sheet_reference_label.setToolTip(
            'kind / label / n exactly as returned by qeeg.grid — the reference is printed as '
            'measured, never upgraded to a licensed clinical norm set.')
        if hasattr(self, 'tbr_card'):
            self.tbr_card.set_metrics(metrics or {})
        return line

    def visible_cells(self):
        return [cell for cell in self.cells.values() if cell.isVisible()]


class DeltaStrip(QWidget):
    """5 × 5 table of mean ``EO z − EC z`` per row × band on the same colour ramp."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.deltas = {}
        self.bands = BAND_KEYS
        self.rows = ROW_KEYS
        self.limits = (-3.0, 3.0)
        self.setMinimumHeight(170)
        self.setMinimumWidth(760)

    def set_deltas(self, deltas, bands=None, rows=None, limits=(-3.0, 3.0)):
        self.deltas = dict(deltas or {})
        self.bands = tuple(bands or BAND_KEYS)
        self.rows = tuple(rows or ROW_KEYS)
        try:
            self.limits = (float(limits[0]), float(limits[1]))
        except (TypeError, ValueError, IndexError):
            self.limits = (-3.0, 3.0)
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor('#FFFFFF'))
        left, top = 112, 24
        width = max(60, self.width() - left - 12)
        height = max(60, self.height() - top - 22)
        columns = max(1, len(self.bands))
        rows = max(1, len(self.rows))
        cell_w = width / columns
        cell_h = height / rows
        painter.setFont(QFont('Segoe UI', 8, QFont.Weight.DemiBold))
        painter.setPen(QColor('#1A1A1A'))
        for column, band in enumerate(self.bands):
            painter.drawText(QRectF(left + column * cell_w, 2, cell_w, top - 4),
                             Qt.AlignmentFlag.AlignCenter,
                             BAND_LABELS.get(band, band))
        for row_index, row in enumerate(self.rows):
            y = top + row_index * cell_h
            painter.setFont(QFont('Segoe UI', 8))
            painter.setPen(QColor('#1A1A1A'))
            painter.drawText(QRectF(0, y, left - 8, cell_h),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             ROW_LABELS.get(row, row))
            for column, band in enumerate(self.bands):
                value = _finite(self.deltas.get((row, band)))
                x = left + column * cell_w
                if value is None:
                    painter.fillRect(QRectF(x + 1, y + 1, cell_w - 2, cell_h - 2),
                                     QColor('#F1EBE2'))
                    painter.setPen(QColor('#A9A29A'))
                    painter.drawText(QRectF(x, y, cell_w, cell_h),
                                     Qt.AlignmentFlag.AlignCenter, '—')
                else:
                    clamped = max(self.limits[0], min(self.limits[1], value))
                    painter.fillRect(QRectF(x + 1, y + 1, cell_w - 2, cell_h - 2),
                                     QColor(qeeg.color_for_z(clamped, self.limits)))
                    painter.setPen(QColor('#1A1A1A'))
                    painter.drawText(QRectF(x, y, cell_w, cell_h),
                                     Qt.AlignmentFlag.AlignCenter, f'{value:+.2f}')
        painter.setFont(QFont('Segoe UI', 8))
        painter.setPen(QColor('#6E6961'))
        painter.drawText(QRectF(0, self.height() - 18, self.width(), 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         'EO z − EC z · blue = lower with eyes open, red = higher · '
                         'clamped to the −3 … +3 ramp')
        painter.end()


def _condition_text(session):
    session = session or {}
    return str(session.get('condition') or session.get('picked_condition') or '').strip()


def _condition_key(session):
    text = _condition_text(session).lower()
    if text in ('ec', 'eyes closed', 'eyes-closed', 'closed'):
        return 'ec'
    if text in ('eo', 'eyes open', 'eyes-open', 'open'):
        return 'eo'
    return ''


def _device_key(session):
    session = session or {}
    return str(session.get('device') or session.get('device_type')
               or session.get('device_label') or '').strip().lower()


def _recorded_reference_kind(session):
    return str((session or {}).get('reference_kind') or '').strip().lower() or None


def comparison_refusal(a, b):
    """Why these two sessions may not be paired; '' when they may.

    Different devices (or different montages) and different reference kinds are
    refused: an EC/EO difference computed across them would describe the
    hardware or the comparison, not the condition. Conditions must be one eyes
    closed and one eyes open, because the delta is defined as EO − EC.
    """
    reasons = []
    left, right = a or {}, b or {}
    device_left, device_right = _device_key(left), _device_key(right)
    if device_left and device_right and device_left != device_right:
        reasons.append(f'different devices ({left.get("device") or left.get("device_type") or device_left}'
                       f' vs {right.get("device") or right.get("device_type") or device_right}) — '
                       'EC/EO pairing refused: that would compare hardware, not condition')
    channels_left = {str(name) for name in (left.get('channels') or [])}
    channels_right = {str(name) for name in (right.get('channels') or [])}
    if channels_left and channels_right and channels_left != channels_right:
        reasons.append('different montages (' + ', '.join(sorted(channels_left ^ channels_right))
                       + ' differ) — EC/EO pairing refused')
    kind_left, kind_right = _recorded_reference_kind(left), _recorded_reference_kind(right)
    if kind_left and kind_right and kind_left != kind_right:
        reasons.append(f'different reference kinds ({kind_left} vs {kind_right}) — the two sheets '
                       'would not be on the same scale, so they are not paired')
    for side, picked, name in ((left, 'ec', 'eyes-closed'), (right, 'eo', 'eyes-open')):
        stored = str((side or {}).get('condition') or '').strip().lower()
        stored_key = _condition_key({'condition': stored})
        if (side or {}).get('picked_condition') and stored_key and stored_key != picked:
            reasons.append(f'the {name} pick is recorded as {side.get("condition")} — pick the '
                           'session that actually matches the slot')
    cond_left, cond_right = _condition_key(left), _condition_key(right)
    if cond_left and cond_right and cond_left == cond_right:
        reasons.append(f'both sessions are {_condition_text(left)} — pair one eyes-closed with one '
                       'eyes-open session (the strip is EO − EC)')
    elif not (cond_left and cond_right):
        reasons.append('a condition is not recorded for both sessions — the delta strip is '
                       'defined as EO − EC, so pairing needs both conditions')
    return '; '.join(reasons)


def _session_reference(metrics, kind, norm_reference=None):
    if kind == 'norm':
        if not norm_reference:
            raise ValueError('no norm reference file is loaded')
        return norm_reference
    if kind == 'session':
        return qeeg.reference_from_session(metrics)
    if kind in ('none', '', None):
        return None
    raise ValueError(f'a {kind} reference belongs to one recording and cannot be shared by two')


def compare_sessions(a, b, reference_kind='session', norm_reference=None):
    """Two grids plus the EO − EC delta table, or ``{'ok': False, 'message': ...}``."""
    refusal = comparison_refusal(a, b)
    if refusal:
        return {'ok': False, 'message': refusal}
    if reference_kind == 'baseline':
        return {'ok': False, 'message': 'A saved baseline belongs to one recording and cannot be '
                                        'used for both sheets — choose This session, a norm file '
                                        'or Raw for the EC/EO comparison.'}
    ec, eo = (a, b) if _condition_key(a) == 'ec' else (b, a)
    prepared = []
    for session in (ec, eo):
        metrics = session.get('metrics')
        if not isinstance(metrics, dict):
            try:
                metrics = qeeg.compute_metrics(np.asarray(session.get('samples'), dtype=float),
                                               float(session.get('fs') or 250.0),
                                               list(session.get('channels') or []))
            except Exception as exc:                    # noqa: BLE001 - report, never fake
                return {'ok': False, 'message': f'Could not compute the {session.get("condition")} '
                                                f'sheet: {exc}'}
        try:
            reference = _session_reference(metrics, reference_kind, norm_reference)
            grid = qeeg.grid(metrics, reference)
        except Exception as exc:                        # noqa: BLE001
            return {'ok': False, 'message': f'Could not build the {session.get("condition")} '
                                            f'reference: {exc}'}
        prepared.append({'session': session, 'metrics': metrics, 'grid': grid})
    deltas = {}
    limits = (-3.0, 3.0)
    for row in ROW_KEYS:
        for band in BAND_KEYS:
            ec_entries = (((prepared[0]['grid'].get('rows') or {}).get(row) or {}).get(band) or {})
            eo_entries = (((prepared[1]['grid'].get('rows') or {}).get(row) or {}).get(band) or {})
            values = []
            for name, entry in eo_entries.items():
                if not isinstance(entry, dict):
                    continue
                z_eo = _finite(entry.get('z'))
                z_ec = _finite(((ec_entries.get(name) or {}) if isinstance(ec_entries, dict)
                                else {}).get('z'))
                if z_eo is not None and z_ec is not None:
                    values.append(z_eo - z_ec)
            deltas[(row, band)] = float(np.mean(values)) if values else None
    return {
        'ok': True,
        'ec': prepared[0],
        'eo': prepared[1],
        'deltas': deltas,
        'limits': limits,
        'header': (
            f'EC — {ec.get("label") or ec.get("session_id") or "session"} '
            f'({_condition_text(ec)}, {_device_key(ec) or "device not recorded"}) · '
            f'EO — {eo.get("label") or eo.get("session_id") or "session"} '
            f'({_condition_text(eo)}, {_device_key(eo) or "device not recorded"}) · '
            'delta = EO z − EC z, same reference kind for both sheets'),
    }


class ComparisonView(QDialog):
    """Two full sheets side by side with an ``EO − EC`` delta strip underneath."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle('qEEG · eyes closed / eyes open comparison')
        self.resize(1720, 1240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        self.content = QWidget()
        self.content.setStyleSheet('background:#FFFFFF;')
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(10, 8, 10, 8)
        content_layout.setSpacing(8)
        self.header = QLabel(result.get('header') or '')
        self.header.setWordWrap(True)
        self.header.setStyleSheet('font-size:11px; color:#1A1A1A; background:#FFFFFF;')
        content_layout.addWidget(self.header)
        sheets_row = QHBoxLayout()
        sheets_row.setSpacing(10)
        self.sheet_ec = QEEGSheet()
        self.sheet_eo = QEEGSheet()
        for sheet, key, condition in ((self.sheet_ec, 'ec', 'eyes closed'),
                                      (self.sheet_eo, 'eo', 'eyes open')):
            part = result.get(key) or {}
            metrics = part.get('metrics') or {}
            grid = part.get('grid') or {}
            sheet.setMinimumWidth(740)
            sheet.apply(grid, grid.get('reference'), condition,
                        len(metrics.get('usable_channels') or []),
                        metrics=metrics)
            sheets_row.addWidget(sheet, 1)
        content_layout.addLayout(sheets_row)
        self.delta_title = QLabel('EO − EC · mean z difference per row × band '
                                  '(green at 0, blue below, red above)')
        self.delta_title.setStyleSheet('font-size:11px; font-weight:700; color:#1A1A1A; '
                                       'background:#FFFFFF;')
        content_layout.addWidget(self.delta_title)
        self.delta = DeltaStrip()
        self.delta.set_deltas(result.get('deltas') or {}, limits=result.get('limits') or (-3, 3))
        content_layout.addWidget(self.delta)
        self.footer = QLabel('Each sheet keeps its own reference line. The delta is only '
                             'meaningful because both sheets share the same reference kind and '
                             'device. ' + REPORT_NOTE)
        self.footer.setWordWrap(True)
        self.footer.setStyleSheet('font-size:9px; color:#6E6961; background:#FFFFFF;')
        content_layout.addWidget(self.footer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.content)
        layout.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.export_button = QPushButton('Export comparison PNG…')
        self.close_button = QPushButton('Close')
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.close_button.clicked.connect(self.close)
        self.export_button.clicked.connect(self._export_clicked)

    def export_png(self, path, scale=1.4):
        """One PNG containing both sheets and the delta strip."""
        from pathlib import Path
        self.content.layout().activate()
        width = max(self.content.size().width(), 1500)
        height = max(self.content.sizeHint().height(), self.content.size().height(), 900)
        pixmap = QPixmap(int(width * scale), int(height * scale))
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.scale(scale, scale)
        self.content.render(painter, QPoint(0, 0))
        painter.end()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not pixmap.save(str(target), 'PNG'):
            raise RuntimeError(f'Could not write the EC/EO comparison PNG to {target}')
        return target

    def _export_clicked(self):
        from datetime import datetime
        path, _filter = QFileDialog.getSaveFileName(
            self, 'Export the EC/EO comparison as PNG',
            f'qeeg_ec_eo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 'PNG image (*.png)')
        if not path:
            return
        try:
            self.export_png(path)
        except (OSError, RuntimeError) as exc:
            self.delta_title.setText(f'Export failed: {exc}')


class CompareDialog(QDialog):
    """Two saved-session pickers (already filtered to one subject by the caller)."""

    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.sessions = list(sessions or [])
        self.setWindowTitle('Compare eyes closed / eyes open')
        layout = QVBoxLayout(self)
        note = QLabel('Pick one eyes-closed and one eyes-open session of the same subject. '
                      'Different devices or reference kinds are refused — the two sheets would '
                      'not be comparable.')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QGridLayout()
        form.addWidget(QLabel('Eyes closed session'), 0, 0)
        self.ec_combo = QComboBox()
        form.addWidget(self.ec_combo, 0, 1)
        form.addWidget(QLabel('Eyes open session'), 1, 0)
        self.eo_combo = QComboBox()
        form.addWidget(self.eo_combo, 1, 1)
        layout.addLayout(form)
        for session in self.sessions:
            text = (f'{session.get("started_at") or session.get("label") or "session"} · '
                    f'{session.get("condition") or "condition not recorded"} · '
                    f'{session.get("device") or session.get("device_label") or "device not recorded"}')
            self.ec_combo.addItem(text)
            self.eo_combo.addItem(text)
        for index, session in enumerate(self.sessions):
            if _condition_key(session) == 'eo':
                self.eo_combo.setCurrentIndex(index)
                break
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QPushButton('Cancel')
        self.ok_button = QPushButton('Compare')
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.ok_button)
        layout.addLayout(buttons)
        self.cancel_button.clicked.connect(self.reject)
        self.ok_button.clicked.connect(self.accept)

    def selection(self):
        """The two picks, each tagged with the slot's condition (user-asserted)."""
        if not self.sessions:
            return None, None
        left = dict(self.sessions[self.ec_combo.currentIndex()])
        right = dict(self.sessions[self.eo_combo.currentIndex()])
        left['picked_condition'] = 'eyes closed'
        right['picked_condition'] = 'eyes open'
        return left, right



class QEEGPanel(QWidget):
    """Toolbar + sheet + "How to read this" panel.

    ``data_provider`` returns ``None`` or a dict with ``samples`` (n, ch in µV),
    ``fs``, ``channels`` and ``session_key``. ``baseline_provider`` returns
    ``None`` or ``{'metrics': ..., 'label': ..., 'n': ...}`` for the current
    subject's saved baseline. ``sessions_provider`` returns the saved sessions
    of the current subject for the EC/EO pickers; ``session_loader`` turns one
    of those entries into ``{'samples','fs','channels','label','condition',
    'device','reference_kind'}``. All of them are called only from an explicit
    refresh / button press, never from a timer.
    """

    raw_trace_requested = pyqtSignal()

    def __init__(self, data_provider=None, baseline_provider=None, parent=None,
                 sessions_provider=None, session_loader=None):
        super().__init__(parent)
        self.data_provider = data_provider
        self.baseline_provider = baseline_provider
        self.sessions_provider = sessions_provider
        self.session_loader = session_loader
        self._cache = {}
        self._grid = None
        self._metrics = None
        self._reference = {'kind': 'none', 'label': '', 'n': 0}
        self._notice = ''
        self.norm_reference = None
        self.norm_meta = None
        self.norm_path = ''
        self.norm_scope_warning = ''
        self.last_error = ''
        self.region_cells = OrderedDict()
        self._region_notice = ''
        self._region_reason = ''
        self._region_row = 'absolute'
        self.comparison_result = None
        self.comparison_view = None
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
        self.compare_button = QPushButton('Compare EC/EO…')
        self.compare_button.setToolTip('Pair one eyes-closed and one eyes-open saved session of '
                                       'this subject; both sheets plus an EO − EC strip.')
        self.export_button = QPushButton('Export sheet PNG…')
        self.report_button = QPushButton('Add to patient report')
        toolbar.addWidget(self.compare_button)
        toolbar.addWidget(self.export_button)
        toolbar.addWidget(self.report_button)
        layout.addLayout(toolbar)

        # Reference-file provenance: source, label, n and age band are printed
        # verbatim; an amber style marks a device-scope mismatch.
        self.norm_label = QLabel('')
        self.norm_label.setWordWrap(True)
        self.norm_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.norm_label.setStyleSheet('background:#FBF7F1; border:1px solid #E9E2D8; '
                                      'padding:6px; font-size:10px; color:#1A1A1A;')
        self.norm_label.setVisible(False)
        layout.addWidget(self.norm_label)

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

        # Visible, one line, above the grid — deliberately not a tooltip.
        reminder_row = QHBoxLayout()
        reminder_row.setSpacing(6)
        self.raw_reminder = QLabel(RAW_TRACE_REMINDER)
        self.raw_reminder.setWordWrap(True)
        self.raw_reminder.setStyleSheet('background:#FFF3D6; color:#7A4B00; border:1px solid '
                                        '#E0B25A; padding:5px 8px; font-size:11px; font-weight:600;')
        reminder_row.addWidget(self.raw_reminder, 1)
        self.raw_link = QPushButton('Open raw traces')
        self.raw_link.setFlat(True)
        self.raw_link.setStyleSheet('color:#0B4A75; text-decoration:underline; border:none; '
                                    'font-size:11px;')
        self.raw_link.setToolTip('Switch to the Live analysis tab, where the unfiltered raw '
                                 'traces are drawn.')
        reminder_row.addWidget(self.raw_link)
        layout.addLayout(reminder_row)

        self.compare_label = QLabel('')
        self.compare_label.setWordWrap(True)
        self.compare_label.setStyleSheet('background:#FFF3D6; color:#7A4B00; border:1px solid '
                                         '#E0B25A; padding:5px 8px; font-size:10px;')
        self.compare_label.setVisible(False)
        layout.addWidget(self.compare_label)

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
        self.compare_button.clicked.connect(self.open_compare)
        self.raw_link.clicked.connect(lambda _checked=False: self.raw_trace_requested.emit())
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

    def load_norm_file(self, path=None):
        """Load a norm file (``norms.load`` when available) and print its provenance.

        ``path`` may be supplied by a caller/test; otherwise a file dialog is
        shown. The source string is printed verbatim, never relabelled: an
        imported licensed export is shown as that export, and only a local file
        is called a local norm.
        """
        if path is None:
            path, _filter = QFileDialog.getOpenFileName(
                self, 'Load a reference norm file (JSON)', self.norm_path or '',
                'Norm reference (*.json);;All files (*)')
            if not path:
                return False
        try:
            reference, meta = self._read_norm_file(path)
        except Exception as exc:              # bad file: say so, never fake it
            self.norm_reference = None
            self.norm_meta = None
            self.norm_path = ''
            self.norm_scope_warning = ''
            self.last_error = str(exc)
            self._notice = f'Norm file not usable: {exc}'
            self.norm_label.setText(f'Reference file not usable — {exc}')
            self.norm_label.setStyleSheet(
                'background:#FFF3D6; color:#7A4B00; border:1px solid #E0B25A; padding:6px; '
                'font-size:10px;')
            self.norm_label.setVisible(True)
            self.status_label.setText(self._notice)
            return False
        self.norm_reference = reference
        self.norm_meta = meta
        self.norm_path = str(path)
        self._notice = ''
        index = next((position for position, (_text, kind) in enumerate(REFERENCE_CHOICES)
                      if kind == 'norm'), 2)
        blocked = self.reference_combo.blockSignals(True)
        self.reference_combo.setCurrentIndex(index)
        self.reference_combo.blockSignals(blocked)
        self.refresh(force=True)
        return True

    @staticmethod
    def _norm_field(loaded, name, default=None):
        if isinstance(loaded, dict):
            return loaded.get(name, default)
        return getattr(loaded, name, default)

    def _read_norm_file(self, path):
        """(reference, metadata) from a NormFile (norms.load) or a qeeg reference JSON."""
        source_path = str(path)
        if norms is not None and hasattr(norms, 'load'):
            try:
                loaded = norms.load(source_path)
            except Exception as norm_error:
                try:
                    reference = qeeg.load_reference(source_path)
                except Exception:
                    raise norm_error
                return reference, {
                    'path': source_path,
                    'source': str(reference.get('source') or 'local'),
                    'label': str(reference.get('label') or ''),
                    'n': reference.get('n'),
                    'age_band': str(reference.get('age_band') or ''),
                    'condition': '', 'device_scope': 'any',
                }
            return self._reference_from_norm(loaded, source_path)
        reference = qeeg.load_reference(source_path)
        return reference, {
            'path': source_path,
            'source': str(reference.get('source') or 'local'),
            'label': str(reference.get('label') or ''),
            'n': reference.get('n'),
            'age_band': str(reference.get('age_band') or ''),
            'condition': '', 'device_scope': 'any',
        }

    def _reference_from_norm(self, loaded, path):
        """Convert a NormFile (or reference dict) into a qeeg reference + metadata."""
        from pathlib import Path
        source = str(self._norm_field(loaded, 'source', '') or 'local')
        label = str(self._norm_field(loaded, 'label', '') or Path(path).stem)
        n = self._norm_field(loaded, 'n')
        age_band = str(self._norm_field(loaded, 'age_band', '') or '')
        condition = str(self._norm_field(loaded, 'condition', '') or '')
        scope = str(self._norm_field(loaded, 'device_scope', 'any') or 'any')
        if isinstance(loaded, dict) and isinstance(loaded.get('mean'), dict) \
                and isinstance(loaded.get('sd'), dict) and 'kind' in loaded:
            reference = dict(loaded)
        elif norms is not None and hasattr(norms, 'to_reference'):
            # Let the norms module do the flattening: it owns the validation.
            reference = norms.to_reference(loaded)
        else:
            rows = self._norm_field(loaded, 'rows', {}) or {}
            mean, sd = {}, {}
            for row, bands in rows.items():
                mean[str(row)], sd[str(row)] = {}, {}
                for band, channels in (bands or {}).items():
                    mean[str(row)][str(band)], sd[str(row)][str(band)] = {}, {}
                    for name, stats in (channels or {}).items():
                        if isinstance(stats, dict):
                            centre = _finite(stats.get('mean', stats.get('value')))
                            spread = _finite(stats.get('sd'))
                        elif isinstance(stats, (list, tuple)) and len(stats) >= 2:
                            centre, spread = _finite(stats[0]), _finite(stats[1])
                        else:
                            centre, spread = _finite(stats), None
                        if centre is None:
                            continue
                        mean[str(row)][str(band)][str(name)] = centre
                        sd[str(row)][str(band)][str(name)] = 0.0 if spread is None else spread
            if not mean:
                raise ValueError('the norm file has no per-channel mean values')
            reference = {
                'schema': qeeg.REFERENCE_SCHEMA,
                'kind': 'norm',
                'label': label,
                'n': n,
                'age_band': age_band or None,
                'bands': [str(band) for band in next(iter(rows.values()) or {})] if rows else [],
                'rows': [str(row) for row in rows],
                'mean': mean,
                'sd': sd,
                'source': source,
            }
        reference = dict(reference)
        reference['kind'] = 'norm'
        reference.setdefault('schema', qeeg.REFERENCE_SCHEMA)
        if not reference.get('n') and n:
            reference['n'] = n
        if not reference.get('age_band') and age_band:
            reference['age_band'] = age_band
        if not reference.get('label'):
            reference['label'] = label
        try:
            clean = qeeg._validated_reference(reference)
        except Exception as exc:
            raise ValueError(f'{path}: {exc}') from exc
        metadata = {'path': str(path), 'source': source, 'label': label,
                    'n': clean.get('n'), 'age_band': clean.get('age_band') or '',
                    'condition': condition, 'device_scope': scope}
        return clean, metadata

    def _update_norm_label(self, channel_count=None):
        meta = self.norm_meta
        if not isinstance(meta, dict) or not meta:
            self.norm_scope_warning = ''
            self.norm_label.setVisible(False)
            return
        scope = str(meta.get('device_scope') or 'any')
        count = channel_count
        if count is None:
            count = len((self._metrics or {}).get('channels') or [])
        warning = ''
        digits = ''.join(character for character in scope if character.isdigit())
        if digits and count and int(digits) != int(count):
            warning = f'{NORM_SCOPE_WARNING} ({scope} norm on {count}ch recording)'
        self.norm_scope_warning = warning
        text = (f'Reference file — source: {meta.get("source")} · label: {meta.get("label")} · '
                f'n={meta.get("n")} · age {meta.get("age_band") or "not recorded"}')
        if meta.get('condition'):
            text += f' · {meta["condition"]}'
        if scope:
            text += f' · device scope {scope}'
        text += f' · {meta.get("path")}'
        if warning:
            text += f' · {warning}'
        self.norm_label.setText(text)
        self.norm_label.setStyleSheet(
            'background:#FFF3D6; color:#7A4B00; border:1px solid #E0B25A; padding:6px; '
            'font-size:10px; font-weight:600;' if warning else
            'background:#FBF7F1; border:1px solid #E9E2D8; padding:6px; font-size:10px; '
            'color:#1A1A1A;')
        self.norm_label.setVisible(True)

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
        metrics = self._metrics or {}
        ref = self._reference or {}
        channel_count = len(metrics.get('channels') or [])
        ref_for_line = dict(ref)
        if str(ref.get('kind')) == 'norm' and isinstance(self.norm_meta, dict):
            ref_for_line['source'] = self.norm_meta.get('source')
            ref_for_line['age_band'] = self.norm_meta.get('age_band')
            if self.norm_meta.get('n') not in (None, ''):
                ref_for_line['n'] = self.norm_meta.get('n')
        line = self.sheet.apply(grid, ref_for_line, self.condition(), channel_count,
                                self._fmt_window(), self.selected_bands(),
                                self.selected_rows(), metrics=metrics)
        self.reference_label.setText(line)
        self._update_region_table(grid)
        self._update_norm_label(channel_count)
        self.stack.setCurrentWidget(self.sheet_page)
        notes = [str(note) for note in (grid.get('notes') or [])
                 if not str(note).startswith('Reference:')]   # footer already prints it
        imported_norm = (str(ref.get('kind')) == 'norm' and isinstance(self.norm_meta, dict)
                         and str(self.norm_meta.get('source') or '').lower()
                         not in ('', 'local'))
        if imported_norm:
            # qeeg's generic norm note says "local norm file"; an imported file is
            # named by its own source above, so that sentence would mislabel it.
            notes = [note for note in notes if 'local norm file' not in note]
        status = f'Computed {self._fmt_window()} of data · reference n={ref.get("n", "—")}'
        if notes:
            status += ' · ' + ' · '.join(str(note) for note in notes)
        if self._region_notice:
            status += ' · ' + self._region_notice
        if self._notice:
            status += ' · ' + self._notice
        self.status_label.setText(status)

    def _update_region_table(self, grid):
        """Draw (or cleanly hide) the region roll-up for the current grid."""
        table_widget = self.sheet.region_table
        self._region_notice = ''
        if regions is None or not hasattr(regions, 'region_table'):
            table_widget.setVisible(False)
            self.region_cells = OrderedDict()
            self._region_reason = 'regions module not available'
            return False
        try:
            row = next(name for name in self.selected_rows() if name in ROW_KEYS)
        except StopIteration:
            table_widget.setVisible(False)
            self.region_cells = OrderedDict()
            self._region_reason = 'no grid row is selected'
            return False
        self._region_row = row
        try:
            table = regions.region_table(grid, row=row)
        except TypeError:
            table = regions.region_table(grid)
        except Exception as exc:              # noqa: BLE001 - optional module
            self.last_error = str(exc)
            self._region_notice = f'Region roll-up unavailable: {exc}'
            table_widget.setVisible(False)
            self.region_cells = OrderedDict()
            self._region_reason = f'region table failed: {exc}'
            return False
        shown = table_widget.set_table(table, bands=self.selected_bands(), row=row)
        self.region_cells = table_widget.cells_map if shown else OrderedDict()
        self._region_reason = '' if shown else 'no region has two or more available sites'
        return shown

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
        self._region_notice = ''
        self._region_reason = ''
        self.region_cells = OrderedDict()
        self.sheet.region_table.setVisible(False)
        self.sheet.tbr_card.set_empty()
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

    # ------------------------------------------------------------ EC / EO view
    def _show_compare_message(self, message):
        self.compare_label.setText(message or '')
        self.compare_label.setVisible(bool(message))

    def _resolve_session(self, entry):
        data = dict(entry or {})
        if self.session_loader is not None:
            loaded = self.session_loader(entry)
            if loaded:
                data.update(loaded)
        return data

    def compare_sessions(self, a, b):
        """Pair two saved sessions; a refusal is written to the visible label."""
        self._show_compare_message('')
        try:
            left, right = self._resolve_session(a), self._resolve_session(b)
        except Exception as exc:              # noqa: BLE001 - report, never fake
            self._show_compare_message(f'Could not read the saved sessions: {exc}')
            return None
        result = compare_sessions(left, right, self.reference_kind(), self.norm_reference)
        if not result.get('ok'):
            self._show_compare_message('EC/EO comparison refused — ' + str(result.get('message')))
            return None
        self.comparison_result = result
        view = ComparisonView(result, self)
        self.comparison_view = view
        view.show()
        return result

    def open_compare(self):
        """Compare EC/EO… button: two pickers over this subject's saved sessions."""
        self._show_compare_message('')
        if self.sessions_provider is None:
            self._show_compare_message('EC/EO comparison needs the saved-session store, which is '
                                       'not connected to this panel.')
            return None
        try:
            sessions = list(self.sessions_provider() or [])
        except Exception as exc:              # noqa: BLE001
            self._show_compare_message(f'Could not list the saved sessions: {exc}')
            return None
        if len(sessions) < 2:
            self._show_compare_message('Two saved sessions of this subject (one eyes closed, one '
                                       'eyes open) are needed for the EC/EO comparison.')
            return None
        dialog = CompareDialog(sessions, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        left, right = dialog.selection()
        if left is None or right is None:
            return None
        return self.compare_sessions(left, right)

    # ---------------------------------------------------------------- report
    def reference_line(self):
        ref = dict(self._reference or {})
        if str(ref.get('kind')) == 'norm' and isinstance(self.norm_meta, dict):
            ref['source'] = self.norm_meta.get('source')
            ref['age_band'] = self.norm_meta.get('age_band')
            if self.norm_meta.get('n') not in (None, ''):
                ref['n'] = self.norm_meta.get('n')
        return describe_reference(ref)

    def theta_beta_metadata(self):
        card = self.sheet.tbr_card
        return {
            'mean': None if card.mean_value is None else float(card.mean_value),
            'flag': bool(card.flagged),
            'usable': bool(card.usable),
            'channels': {str(name): float(value) for name, value in card.ratios.items()},
            'note': TBR_NOTE,
            'flag_rule': f'channel mean >= {TBR_FLAG_MEAN:.1f} and both bands usable',
        }

    def region_metadata(self):
        cells = self.region_cells or {}
        if not cells:
            reason = self._region_reason or ('regions module not available' if regions is None
                                             else 'no region has two or more available sites')
            return {'available': False, 'reason': reason, 'bands': [], 'regions': [],
                    'values': {}, 'flags': []}
        values, flags = {}, []
        for (band, region), (value, sites, _flag) in cells.items():
            values.setdefault(str(band), {})[str(region)] = float(value)
            if region_flag((value, sites, _flag)):
                flags.append({'band': str(band), 'region': str(region), 'z': float(value),
                              'n_sites': sites})
        return {
            'available': True,
            'row': self._region_row,
            'bands': [str(band) for band in self.sheet.region_table.band_order],
            'regions': [str(region) for region in self.sheet.region_table.region_order],
            'values': values,
            'flags': flags,
            'rule': 'mean z per region; flag when |z| >= 2 with >= 2 sites',
        }

    def report_metadata(self):
        ref = self._reference or {}
        return {
            'reference_line': self.reference_line(),
            'reference_kind': ref.get('kind'),
            'reference_label': ref.get('label'),
            'reference_n': ref.get('n'),
            'reference_source': (self.norm_meta or {}).get('source') if self.norm_meta else None,
            'reference_age_band': ((self.norm_meta or {}).get('age_band')
                                   if self.norm_meta else None),
            'norm_scope_warning': self.norm_scope_warning or None,
            'condition': self.condition(),
            'bands': list(self.selected_bands()),
            'rows': list(self.selected_rows()),
            'channels': list((self._metrics or {}).get('channels') or []),
            'theta_beta': self.theta_beta_metadata(),
            'regions': self.region_metadata(),
            'raw_reminder': RAW_TRACE_REMINDER,
            'note': REPORT_NOTE,
        }

    def describe_reference(self):
        return self.reference_line()
