"""Second-device live window — two headsets at once, each with its own heat map.

Why: NeuroSync's main window drives ONE headset.  When a BrainBit and a
DragonEEG are both powered, you want both streams live SIMULTANEOUSLY — e.g.
one on the subject, one on a comparison montage — and you want the band-power
heat map (topomap), the predominant-rhythm read-out and a savable session for
EACH device.  This window is a second, independent live pipeline:

* its own device picker / scan / connect (BLE via engine.EEGClient) or the
  clearly-labelled simulator (``simulated`` argument),
* its own 60 s rolling buffer, 5 s window analysis, contact check and topomap,
* save to the same per-subject permanent storage as the main window,
* head-map PNG export and a local AI-doctor request for this device's numbers.

Honesty rules are identical to the main window: engineering-grade descriptive
numbers only — no diagnosis, no mental-state claims; simulated data is
labelled SIMULATED everywhere.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QBuffer, QIODevice, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout,
                             QWidget, QHeaderView)
import pyqtgraph as pg

import ai_doctor
import devices
import storage
import topomap
from eeg_analysis import BANDS, CHANNELS, COLORS, analyze_window
from engine import EEGClient
from simfeed import SimulatedEEGClient, SIM_NOTE

BUFFER_SECONDS = 60
DEFAULT_FAMILY = 'dragon'


def grab_png(widget) -> bytes:
    pixmap = widget.grab()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, 'PNG')
    return bytes(buffer.data())


def render_topomap_png(field, band, channels, footer) -> bytes:
    """Off-screen presentation render of one band's head map (PNG bytes)."""
    plot = pg.PlotWidget()
    plot.setBackground('#0e1c26')
    plot.setAspectLocked(True)
    plot.setRange(xRange=(-1.32, 1.32), yRange=(-1.32, 1.32), padding=0)
    plot.getPlotItem().setTitle(
        f'NeuroSync · {band} relative power (%) · schematic head map — second device',
        size='14pt', color='#e3edf4')
    try:
        cmap = pg.colormap.get('turbo')
    except Exception:
        cmap = pg.colormap.get('jet')
    image = pg.ImageItem()
    image.setLookupTable(cmap.getLookupTable(nPts=256))
    plot.addItem(image)
    image.setImage(field[::-1, :], autoLevels=True, axisOrder='row-major')
    from PyQt6.QtCore import QRectF
    image.setRect(QRectF(-1, -1, 2, 2))
    outline_x, outline_y = topomap.head_outline(181)
    plot.plot(outline_x, outline_y, pen=pg.mkPen('#6f8ea0', width=1.6))
    plot.plot([-.22, 0, .22], [.97, 1.14, .97], pen=pg.mkPen('#6f8ea0', width=1.6))
    plot.plot([-1, -1.12, -1.12, -1], [-.22, -.12, .12, .22], pen=pg.mkPen('#6f8ea0', width=1.6))
    plot.plot([1, 1.12, 1.12, 1], [-.22, -.12, .12, .22], pen=pg.mkPen('#6f8ea0', width=1.6))
    positions = [topomap.ELECTRODE_POSITIONS[name] for name in channels
                 if name in topomap.ELECTRODE_POSITIONS]
    if positions:
        spots = pg.ScatterPlotItem(size=9, brush=pg.mkBrush('#e3edf4'), pen=pg.mkPen('#20323e'))
        spots.setData(pos=np.asarray(positions, dtype=float))
        plot.addItem(spots)
        for name in channels:
            if name in topomap.ELECTRODE_POSITIONS:
                x, y = topomap.ELECTRODE_POSITIONS[name]
                label = pg.TextItem(name, color='#cde6f2', anchor=(0.5, 0.5))
                label.setPos(x * 1.16, y * 1.16)
                plot.addItem(label)
    footer_item = pg.TextItem(footer, color='#9aadba', anchor=(0, 1))
    footer_item.setPos(-1.30, -1.30)
    plot.addItem(footer_item)
    plot.resize(1200, 1100)
    png = grab_png(plot)
    plot.setParent(None)
    plot.deleteLater()
    return png


class SubjectEntryDialog(QDialog):
    """Minimal subject entry for the second window (name + phone)."""

    def __init__(self, parent=None, name='', phone=''):
        super().__init__(parent)
        self.setWindowTitle('Second device — subject')
        form = QFormLayout(self)
        self.name_edit = QLineEdit(name)
        self.phone_edit = QLineEdit(phone)
        form.addRow('Name *', self.name_edit)
        form.addRow('Phone *', self.phone_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def subject(self):
        return {'name': self.name_edit.text().strip(),
                'phone': self.phone_edit.text().strip()}


class AIWorker(QThread):
    note_ready = pyqtSignal(dict)

    def __init__(self, payload, model, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.model = model

    def run(self):
        self.note_ready.emit(ai_doctor.ask(self.payload, model=self.model))


class SecondDeviceWindow(QDialog):
    def __init__(self, parent=None, simulated=None):
        super().__init__(parent)
        self.setWindowTitle('NeuroSync · second device (independent live stream)')
        self.resize(1180, 860)
        self.simulated = simulated
        self.channels = tuple(CHANNELS)
        self.fs = devices.CLASSIC.fs
        self.samples = []
        self.packet_numbers = []
        self.contact = None
        self.last_report = None
        self.device_metadata = {}
        self.transport_metrics = {}
        self.subject = None
        self.ai_note = None
        self.ai_worker = None
        self.client = None
        self.last_received = None
        self._build_ui()
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self.analyze_now)
        self.analysis_timer.start(1000)
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_buttons)
        self.status_timer.start(1500)
        if self.simulated:
            self.status_label.setText('Simulator armed — press Connect to start the '
                                      'clearly-labelled SIMULATED stream for this window.')
        self.update_buttons()

    # ------------------------------------------------------------------ layout
    def _build_ui(self):
        layout = QVBoxLayout(self)
        heading = QLabel('SECOND DEVICE · independent stream · own heat map')
        heading.setStyleSheet('font-size:16px; font-weight:700; color:#8be6ce; padding:4px;')
        layout.addWidget(heading)

        bar = QHBoxLayout()
        self.scan_button = QPushButton('Scan devices')
        self.device_combo = QComboBox(); self.device_combo.setMinimumWidth(300)
        self.device_combo.addItem('No scan yet — press "Scan devices"', None)
        self.connect_button = QPushButton('Connect')
        self.disconnect_button = QPushButton('Disconnect')
        self.contact_button = QPushButton('Check electrodes')
        self.save_button = QPushButton('Save session (this device)')
        for widget in (self.scan_button, self.device_combo, self.connect_button,
                       self.disconnect_button, self.contact_button, self.save_button):
            bar.addWidget(widget)
        bar.addStretch()
        layout.addLayout(bar)

        subject_row = QHBoxLayout()
        self.subject_button = QPushButton('Set subject…')
        self.subject_label = QLabel('Subject: not set (required before saving)')
        self.subject_label.setStyleSheet('color:#f0c47b;')
        subject_row.addWidget(self.subject_button); subject_row.addWidget(self.subject_label)
        subject_row.addStretch()
        subject_row.addWidget(QLabel('Head map band'))
        self.map_band_select = QComboBox(); self.map_band_select.addItems(list(BANDS))
        subject_row.addWidget(self.map_band_select)
        layout.addLayout(subject_row)

        self.status_label = QLabel('Disconnected. This window is independent from the main '
                                   'one: scan/connect a DIFFERENT headset here to run both '
                                   'at the same time.')
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet('background:#182730; padding:7px; color:#b9cad7;')
        layout.addWidget(self.status_label)
        self.sim_note = QLabel(SIM_NOTE)
        self.sim_note.setWordWrap(True)
        self.sim_note.setStyleSheet('color:#f0c47b; font-size:11px; padding:4px;')
        self.sim_note.setVisible(bool(self.simulated))
        layout.addWidget(self.sim_note)

        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)

        left = QWidget(); left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel('BAND POWER · latest 5 s · µV² and % of 1–45 Hz'))
        self.bars_plot = pg.PlotWidget()
        self.bars_plot.setLabel('left', 'power', units='µV²')
        self.bars_plot.setMinimumHeight(210)
        names = list(BANDS)
        self.bars_plot.getAxis('bottom').setTicks([[(i, n) for i, n in enumerate(names)]])
        self.bars_absolute = pg.BarGraphItem(x=[], height=[], width=.6)
        self.bars_relative = pg.BarGraphItem(x=[], height=[], width=.6, brush='#3f6379')
        self.bars_plot.addItem(self.bars_relative)
        self.bars_plot.addItem(self.bars_absolute)
        left_layout.addWidget(self.bars_plot)

        left_layout.addWidget(QLabel('BRAIN MAP · second device · schematic layout (%)'))
        self.map_plot = pg.PlotWidget()
        self.map_plot.setAspectLocked(True)
        self.map_plot.setMinimumHeight(330)
        self.map_plot.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25), padding=0)
        self.map_image = pg.ImageItem()
        try:
            self.map_cmap = pg.colormap.get('turbo')
        except Exception:
            self.map_cmap = pg.colormap.get('jet')
        self.map_image.setLookupTable(self.map_cmap.getLookupTable(nPts=256))
        self.map_plot.addItem(self.map_image)
        outline_x, outline_y = topomap.head_outline(181)
        self.map_plot.plot(outline_x, outline_y, pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_electrodes = pg.ScatterPlotItem(size=7, brush=pg.mkBrush('#e3edf4'),
                                                 pen=pg.mkPen('#20323e'))
        self.map_plot.addItem(self.map_electrodes)
        left_layout.addWidget(self.map_plot)
        map_row = QHBoxLayout()
        self.map_status = QLabel('Waiting for a valid five-second window.')
        self.map_status.setWordWrap(True)
        self.map_status.setStyleSheet('color:#9aadba; font-size:11px;')
        self.map_export_button = QPushButton('Export head map PNG…')
        map_row.addWidget(self.map_status, 1); map_row.addWidget(self.map_export_button)
        left_layout.addLayout(map_row)
        self.map_export_button.clicked.connect(self.export_head_map)
        left_layout.addStretch()
        split.addWidget(left)

        right = QWidget(); right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel('LATEST 5 SECONDS · this device'))
        self.predominant = QLabel('PREDOMINANT RHYTHM · —')
        self.predominant.setWordWrap(True)
        self.predominant.setStyleSheet('font-size:15px; padding:9px; background:#17262f;')
        right_layout.addWidget(self.predominant)
        self.conclusion = QLabel('Waiting for five seconds of continuous EEG.')
        self.conclusion.setWordWrap(True)
        self.conclusion.setStyleSheet('font-size:13px; padding:9px; background:#20323e;')
        right_layout.addWidget(self.conclusion)
        self.quality_label = QLabel('Signal quality unverified.')
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet('color:#f0c47b;')
        right_layout.addWidget(self.quality_label)
        self.channel_table = QTableWidget(0, 4)
        self.channel_table.setHorizontalHeaderLabels(['Channel', 'Power µV²', 'Relative %', 'Dominant'])
        self.channel_table.verticalHeader().hide()
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.channel_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.channel_table, 1)
        self.contact_label = QLabel('ELECTRODE CONTACT · not checked')
        self.contact_label.setWordWrap(True)
        right_layout.addWidget(self.contact_label)
        right_layout.addWidget(QLabel('AI DOCTOR · this device’s numbers · local model'))
        ai_row = QHBoxLayout()
        self.ai_button = QPushButton('Ask AI doctor')
        self.ai_auto = QCheckBox('auto every 60 s')
        ai_row.addWidget(self.ai_button); ai_row.addWidget(self.ai_auto); ai_row.addStretch()
        right_layout.addLayout(ai_row)
        self.ai_text = QLabel('Ask for an assistive description of THIS device’s latest measured '
                              'numbers. Same rules as the main window: derived values only, local '
                              'model, assistive description — not a diagnosis.')
        self.ai_text.setWordWrap(True)
        self.ai_text.setStyleSheet('background:#182730; padding:9px; font-size:11px;')
        right_layout.addWidget(self.ai_text)
        right_layout.addStretch()
        split.addWidget(right)
        split.setSizes([700, 420])

        footer = QLabel('Same honesty rules as the main window: descriptive engineering numbers, '
                        'schematic maps, no diagnosis, nothing uploaded. Raw EEG never leaves '
                        'this PC.')
        footer.setWordWrap(True)
        footer.setStyleSheet('color:#9aadba; padding:4px;')
        layout.addWidget(footer)

        # signals
        self.scan_button.clicked.connect(self.scan_devices)
        self.connect_button.clicked.connect(self.connect_device)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.contact_button.clicked.connect(self.request_contact)
        self.save_button.clicked.connect(self.save_session)
        self.subject_button.clicked.connect(self.set_subject)
        self.map_band_select.currentIndexChanged.connect(lambda _: self.update_topomap())
        self.ai_button.clicked.connect(self.run_ai_doctor)

    # ------------------------------------------------------------------ client
    def _install_client(self, **kwargs):
        options = dict(auto_contact=True)
        options.update(kwargs)
        if self.simulated:
            self.client = SimulatedEEGClient(family=self.simulated,
                                             auto_contact=options.get('auto_contact', True))
        else:
            self.client = EEGClient(**options)
        self.client.batch_received.connect(self.handle_batch)
        self.client.status_changed.connect(self.handle_status)
        self.client.device_info.connect(self.handle_device)
        self.client.devices_found.connect(self.handle_devices)
        self.client.contact_received.connect(self.handle_contact)
        self.client.stream_reset.connect(self.reset_window)
        self.client.metrics_received.connect(self.handle_metrics)
        self.client.finished.connect(self.update_buttons)

    def scan_devices(self):
        if self.simulated:
            self.connect_device()
            return
        if self.client is not None and self.client.isRunning():
            self.status_label.setText('A scan or stream is already running in this window.')
            return
        self._install_client(selection_required=True, selection_timeout=600.)
        self.reset_window('Scanning for devices (second window)…')
        self.client.start()
        self.update_buttons()

    def connect_device(self):
        if self.client is not None and self.client.isRunning():
            if getattr(self.client, 'mode', None) == 'awaiting_selection':
                data = self.device_combo.currentData()
                address = data.get('address') if isinstance(data, dict) else None
                if address:
                    self.client.select_device(address)
                else:
                    self.status_label.setText('Select a device in this window’s list first.')
            return
        if self.simulated:
            self._install_client()
        else:
            data = self.device_combo.currentData()
            address = data.get('address') if isinstance(data, dict) else None
            if not address:
                self.status_label.setText('Scan first, then choose the second device here.')
                return
            self._install_client(target_address=address, selection_required=False)
        self.reset_window('Connecting (second device)…')
        self.client.start()
        self.update_buttons()

    def disconnect_device(self):
        if self.client:
            self.client.stop()
        self.contact = None
        self.reset_window('Second device disconnected.')
        self.update_buttons()

    def request_contact(self):
        if self.client and self.client.isRunning():
            self.client.check_contact()

    def update_buttons(self):
        running = self.client is not None and self.client.isRunning()
        self.connect_button.setEnabled(not running)
        self.disconnect_button.setEnabled(running)
        self.contact_button.setEnabled(running)
        self.scan_button.setEnabled(not running and not self.simulated)
        enough = len(self.samples) >= self.fs * 5
        self.save_button.setEnabled(enough)
        busy = self.ai_worker is not None and self.ai_worker.isRunning()
        self.ai_button.setEnabled(not busy and self.last_report is not None)

    # ------------------------------------------------------------- device data
    def handle_devices(self, described):
        self.device_combo.clear()
        if not described:
            self.device_combo.addItem('No devices found (second window)', None)
            return
        for item in described:
            channels = len(item.get('channels') or [])
            label = (f"{item['device_label']} · {item['address']} · {channels} channels"
                     f"{' · pairing required' if item.get('pairing_required') else ''}")
            self.device_combo.addItem(label, item)

    def handle_status(self, text):
        self.status_label.setText(text)

    def handle_device(self, info):
        self.apply_device_info(info)

    def apply_device_info(self, info):
        channels = tuple(info.get('channels') or self.channels)
        fs = int(info.get('fs') or self.fs)
        if channels != self.channels or fs != self.fs:
            self.channels = channels; self.fs = fs
            self.samples = []; self.packet_numbers = []
        self.device_metadata = info
        self.status_label.setText(
            f"SECOND DEVICE: {info.get('device_label', 'device')} · {fs} Hz · "
            f"{len(channels)} channels · {info.get('channel_source', '')}")

    def handle_batch(self, batch):
        channels = tuple(batch.get('channels') or self.channels)
        if channels != self.channels:
            self.channels = channels
            self.samples = []; self.packet_numbers = []
        limit = self.fs * BUFFER_SECONDS
        self.samples.extend(batch['samples_uv'])
        self.packet_numbers.extend(batch['packnums'])
        if len(self.samples) > limit:
            del self.samples[:len(self.samples) - limit]
            del self.packet_numbers[:len(self.packet_numbers) - limit]
        self.last_received = batch['arrived_monotonic']

    def handle_metrics(self, metrics):
        self.transport_metrics = metrics

    def handle_contact(self, contact):
        self.contact = contact
        ohms = contact.get('ohms') or {}
        text = ' · '.join(f'{name}: ' + (f'{value / 1000:,.0f} kΩ' if value else 'open')
                          for name, value in list(ohms.items())[:8])
        self.contact_label.setText('ELECTRODE CONTACT · ' + (text or 'no values'))

    def reset_window(self, reason):
        self.samples = []; self.packet_numbers = []
        self.last_received = None; self.last_report = None
        self.conclusion.setText(reason)
        self.predominant.setText('PREDOMINANT RHYTHM · —')
        self.channel_table.setRowCount(0)
        self.bars_absolute.setOpts(x=[], height=[])
        self.bars_relative.setOpts(x=[], height=[])
        self.map_image.clear()
        self.map_status.setText('Waiting for a valid five-second window (second device).')
        self.update_buttons()

    # ----------------------------------------------------------------- analysis
    def analyze_now(self):
        if len(self.samples) < self.fs * 5:
            self.update_buttons()
            return
        age = None if self.last_received is None else time.monotonic() - self.last_received
        contact_age = None if self.contact is None else time.monotonic() - self.contact['measured_monotonic']
        try:
            samples = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        except ValueError:
            return
        notch = 60
        report = analyze_window(samples, fs=self.fs, notch_hz=notch,
                                contact_ohms=self.contact['ohms'] if self.contact else None,
                                contact_age_s=contact_age,
                                continuous=age is not None and age <= .5,
                                channels=self.channels)
        self.last_report = report
        self.conclusion.setText(report['conclusion'])
        self.quality_label.setText(report['reason'])
        if report.get('dominant'):
            self.predominant.setText(
                f"PREDOMINANT RHYTHM · {str(report['dominant']).upper()}"
                + (f" · {float(report['dominant_pct']):.1f}% of 1–45 Hz"
                   if report.get('dominant_pct') is not None else '')
                + (f" · peak {float(report['peak_hz']):.1f} Hz"
                   if report.get('peak_hz') is not None else ''))
        powers = [report['bands'].get(name, {}).get('power_uv2') for name in BANDS]
        relative = [report['bands'].get(name, {}).get('relative_pct') for name in BANDS]
        xs = list(range(len(BANDS)))
        heights = [value if value is not None else 0. for value in powers]
        scale = max([value for value in relative if value is not None] or [0.]) or 1.
        heights_relative = [(value / scale * (max(heights) or 1.)) if value is not None else 0.
                            for value in relative]
        self.bars_absolute.setOpts(x=xs, height=heights,
                                   brushes=[pg.mkBrush(COLORS[name]) for name in BANDS])
        self.bars_relative.setOpts(x=xs, height=heights_relative, width=.6)
        self._fill_channel_table(report)
        self.update_topomap()
        self.maybe_auto_ai()
        self.update_buttons()

    def _fill_channel_table(self, report):
        channels = list(report.get('channels', {}))
        self.channel_table.setRowCount(len(channels))
        for row, name in enumerate(channels):
            entry = report['channels'][name]
            relative = entry.get('relative_pct') or {}
            powers = entry.get('powers_uv2') or {}
            top_band = max(relative, key=relative.get) if relative else None
            cells = [name,
                     (f"{powers[top_band]:.2f} ({top_band})" if top_band else '—'),
                     (f"{relative[top_band]:.1f}%" if top_band else '—'),
                     (entry.get('dominant') or ('mixed' if relative else 'unreliable'))
                     + ('' if entry.get('valid') else ' · flagged')]
            for column, text in enumerate(cells):
                self.channel_table.setItem(row, column, QTableWidgetItem(str(text)))

    def _topomap_field(self, band, grid_n=140):
        report = self.last_report
        if not report or not report.get('bands', {}).get(band, {}).get('relative_pct'):
            return None
        powers = {name: (report['channels'].get(name, {}).get('relative_pct') or {})
                  for name in self.channels}
        field = topomap.topomap_band(powers, band, list(self.channels), grid_n=grid_n)
        return field if np.isfinite(field).any() else None

    def update_topomap(self):
        band = self.map_band_select.currentText()
        scalp = devices.scalp_channels(self.channels)
        positions = [topomap.ELECTRODE_POSITIONS[name] for name in scalp
                     if name in topomap.ELECTRODE_POSITIONS]
        self.map_electrodes.setData(pos=np.asarray(positions, dtype=float)
                                    if positions else np.zeros((0, 2)))
        field = self._topomap_field(band)
        if field is None:
            self.map_image.clear()
            self.map_status.setText(
                f'No {band} map yet (second device): needs a valid five-second window and at '
                f'least three usable electrodes (currently {len(scalp)} scalp channel(s)).')
            return
        self.map_image.setImage(field[::-1, :], autoLevels=True, axisOrder='row-major')
        from PyQt6.QtCore import QRectF
        self.map_image.setRect(QRectF(-1, -1, 2, 2))
        finite = field[np.isfinite(field)]
        self.map_status.setText(
            f'{band}: {len(scalp)} schematic positions, colour {finite.min():.1f}–'
            f'{finite.max():.1f}%. Approximate layout, not source localisation.')

    def export_head_map(self):
        band = self.map_band_select.currentText()
        field = self._topomap_field(band, grid_n=288)
        if field is None:
            QMessageBox.warning(self, 'Head map export',
                                f'No {band} map yet (second device).')
            return
        default = str(Path.home() / f'head_map_second_{band.lower()}_'
                                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        path, _ = QFileDialog.getSaveFileName(self, 'Save second-device head map PNG',
                                              default, 'PNG image (*.png)')
        if not path:
            return
        png = render_topomap_png(field, band, devices.scalp_channels(self.channels),
                                 'Schematic 10-20 interpolation (second device) — not source '
                                 'localisation. Descriptive research view only.')
        Path(path).write_bytes(png)
        self.status_label.setText(f'Head map PNG written: {path}')

    # --------------------------------------------------------------------- AI
    def maybe_auto_ai(self):
        if not self.ai_auto.isChecked():
            return
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return
        if self.last_report is None:
            return
        self.run_ai_doctor()

    def run_ai_doctor(self):
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return
        if self.last_report is None:
            self.ai_text.setText('No analysis yet in this window — wait for five seconds of EEG.')
            return
        payload = ai_doctor.build_payload(report=self.last_report, stable=None,
                                          transport=self.transport_metrics,
                                          device=self.device_metadata,
                                          contact=self.contact['ohms'] if self.contact else None)
        self.ai_button.setEnabled(False)
        self.ai_worker = AIWorker(payload, ai_doctor.DEFAULT_MODEL, self)
        self.ai_worker.note_ready.connect(self.handle_ai_note)
        self.ai_worker.start()

    def handle_ai_note(self, result):
        self.ai_button.setEnabled(True)
        self.ai_note = result
        if result.get('ok'):
            self.ai_text.setText(result.get('text', '') + '\n\n' + ai_doctor.DISCLAIMER)
        else:
            self.ai_text.setText('AI doctor unavailable: ' + str(result.get('error'))
                                 + '\n\nNo substitute analysis is shown.')
        self.update_buttons()

    # ------------------------------------------------------------------ saving
    def set_subject(self):
        dialog = SubjectEntryDialog(self, **(self.subject or {}))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        subject = dialog.subject()
        if not subject['name'] or not subject['phone']:
            QMessageBox.warning(self, 'Subject', 'Name and phone are required.')
            return
        self.subject = subject
        self.subject_label.setText(f"Subject: {subject['name']} / {subject['phone']}")

    def save_session(self):
        if len(self.samples) < self.fs * 5:
            QMessageBox.warning(self, 'Nothing to save',
                                'Five seconds of continuous EEG are required (second device).')
            return
        if not self.subject or not self.subject.get('name'):
            self.set_subject()
            if not self.subject or not self.subject.get('name'):
                return
        self.analyze_now()
        rows = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        charts = {}
        try:
            charts['band_power_bars_second_device.png'] = grab_png(self.bars_plot)
        except Exception:
            pass
        try:
            charts['head_map_current_second_device.png'] = grab_png(self.map_plot)
        except Exception:
            pass
        for band in BANDS:
            field = self._topomap_field(band, grid_n=288)
            if field is None:
                continue
            try:
                charts[f'head_map_{band.lower()}_second_device.png'] = render_topomap_png(
                    field, band, devices.scalp_channels(self.channels),
                    'Second device — schematic interpolation, not source localisation.')
            except Exception:
                pass
        summary = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'device_type': self.device_metadata.get('device_key', 'unknown'),
            'device_label': self.device_metadata.get('device_label', 'unknown'),
            'device': self.device_metadata,
            'simulated': bool(self.simulated),
            'window': 'second device window',
            'family': self.device_metadata.get('family'),
            'channels': list(self.channels),
            'scalp_channels': list(devices.scalp_channels(self.channels)),
            'poly_channels': list(devices.poly_channels(self.channels)),
            'channel_source': self.device_metadata.get('channel_source', 'unknown'),
            'fs_hz': self.fs,
            'raw_rows': int(rows.shape[0]),
            'duration_s': round(rows.shape[0] / self.fs, 3),
            'notch_hz': 60,
            'analysis': self.last_report,
            'stable_estimate': None,
            'contact': self.contact,
            'transport': self.transport_metrics,
            'ai_doctor': (dict(self.ai_note) if isinstance(self.ai_note, dict) else None),
            'methodology': 'METHODS.md',
            'limits': 'Engineering signal-quality estimates only: not a diagnosis, not a clinical '
                      'measurement, no mental-state, emotion or sleep-stage inference.',
        }
        header = (['sample_index', 'relative_time_s', 'packet_number']
                  + [f'raw_{name}_uV' for name in self.channels])
        data_rows = [[index, round(index / self.fs, 6), number, *values]
                     for index, (number, values) in enumerate(
                         zip(list(self.packet_numbers), rows.tolist()))]
        try:
            folder = storage.save_session(storage.DEFAULT_ROOT, self.subject['name'],
                                          self.subject.get('phone', ''), summary, header,
                                          data_rows, charts=charts)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Save failed', str(exc))
            return
        self.status_label.setText(f'Second-device session saved locally: {folder}')

    def closeEvent(self, event):
        self.analysis_timer.stop()
        self.status_timer.stop()
        if self.client is not None:
            self.client.stop()
        event.accept()
