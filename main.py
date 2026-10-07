"""NeuroSync live EEG research viewer.

One application for the verified BrainBit Classic/Black headset, the BrainBit 2
family and the 21+ channel DragonEEG (NeuroEEG). Real SDK samples only; the
only synthetic source is the built-in simulator (``--simulate``), which is
labelled SIMULATED everywhere it appears and exists to rehearse the live
pipeline — it is never mixed into real recordings. It is an exploratory signal
viewer: not a medical device, not a diagnosis, not a validated brain map and
not a mental-state classifier.

Everything stays on this PC. Recordings and analysis summaries are written
under recordings/subjects/<subject>/<timestamp>/ and nothing is uploaded.
"""
import sys
import threading
import time
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from html import escape as esc_html
from collections import deque
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QMessageBox, QCheckBox, QScrollArea, QGridLayout,
    QDialog, QLineEdit, QDialogButtonBox, QFormLayout, QListWidget, QListWidgetItem,
    QTextBrowser, QFileDialog, QSizePolicy)
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal, QBuffer, QIODevice, QRectF
from PyQt6.QtGui import QColor, QPixmap, QTextCursor
from PyQt6 import sip
import pyqtgraph as pg
from engine import EEGClient
from eeg_analysis import (CHANNELS, BANDS, COLORS, analyze_window, analyze_stable,
                          band_traces, mean_psd_curve)
import ai_doctor
import baseline
import devices
import patient_report
import storage
import topomap
from simfeed import SimulatedEEGClient

ROOT = Path(__file__).resolve().parent
SUBJECT_ROOT = storage.DEFAULT_ROOT
STABLE_MIN_SAMPLES = 3000      # 12 s: six clean 2 s epochs (at 250 Hz)
AI_AUTO_INTERVAL_S = 60.
CHAT_INTERVAL_CHOICES = (15, 30, 60)   # seconds between automatic live-chat updates
CHAT_DEFAULT_INTERVAL_S = 30.
CHAT_HISTORY_MAX = 8                   # previous chat turns kept for continuity
BUFFER_SECONDS = 60            # rolling analysis buffer, also the session raw span
SAVED_ADDRESS = 'DA:BB:A3:A0:73:4E'   # the user's already discovered BrainBit Classic
_ORPHAN_WORKERS = []           # keep detached AI threads alive until they finish

# Distinct plot colours for up to 24 channels; the four Classic channels keep
# their original colours so the verified view looks unchanged.
CHANNEL_PALETTE = [
    '#6adacc', '#f1cc63', '#b899ec', '#f79f79',
    '#7fd1f7', '#9ee493', '#ffd28a', '#c9a7ff',
    '#84e0c0', '#f5a3c7', '#8fb8ff', '#e8d16b',
    '#a0e6e0', '#ffb1a1', '#b6d36a', '#d3a7f0',
    '#6fc3e8', '#f0b178', '#8ee6b4', '#e39bd6',
    '#a8c8ff', '#c8e06f', '#f2c7a0', '#9ad9f0',
]
CLASSIC_COLORS = dict(zip(CHANNELS, CHANNEL_PALETTE[:4]))


class BandAxis(pg.AxisItem):
    """X axis that prints band names instead of indices."""
    def tickStrings(self, values, scale, spacing):
        names = list(BANDS)
        labels = []
        for value in values:
            index = int(round(value))
            labels.append(names[index] if 0 <= index < len(names) and abs(value - index) < 1e-6 else '')
        return labels


def grab_png(widget):
    """PNG bytes of a widget as currently rendered (local file content only)."""
    pixmap = widget.grab()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not pixmap.save(buffer, 'PNG'):
        raise RuntimeError('Qt could not encode the image as PNG')
    return bytes(buffer.data())


def widget_alive(widget) -> bool:
    """False once Qt has deleted the C++ object behind a widget wrapper.

    Qt destroys the C++ widgets of closed windows/dialogs while their Python
    wrappers survive, and every later call on such a wrapper raises RuntimeError
    ('wrapped C/C++ object ... has been deleted'). Refresh paths test this first
    so a redraw after a widget is gone is skipped instead of crashing the window
    (e.g. pressing "Save session" after a plot or dialog was torn down).
    """
    if widget is None:
        return False
    try:
        return not sip.isdeleted(widget)
    except (RuntimeError, TypeError):
        return False


class AIDoctorWorker(QThread):
    """Runs one local-model request off the GUI thread."""
    note_ready = pyqtSignal(dict)

    def __init__(self, payload, model, parent=None):
        super().__init__(parent)
        self.payload = payload
        self.model = model

    def run(self):
        self.note_ready.emit(ai_doctor.ask(self.payload, model=self.model))


class LiveChatWorker(QThread):
    """Streams one local-model live-chat reply off the GUI thread."""
    chunk_ready = pyqtSignal(str)
    finished_run = pyqtSignal(dict)

    def __init__(self, messages, model, parent=None):
        super().__init__(parent)
        self.messages = messages
        self.model = model
        self.stop_requested = False

    def run(self):
        parts, error = [], ''
        try:
            for event in ai_doctor.stream_ask(self.messages, model=self.model):
                if self.stop_requested:
                    break
                kind = event.get('type')
                if kind == 'chunk':
                    text = str(event.get('text') or '')
                    if text:
                        parts.append(text)
                        self.chunk_ready.emit(text)
                elif kind == 'error':
                    error = str(event.get('error') or 'unknown error')
                    break
        except Exception as exc:                     # a worker crash must never escape
            error = f'unexpected stream error: {exc}'
        self.finished_run.emit({'ok': not error and bool(parts),
                                'text': ''.join(parts), 'error': error,
                                'model': self.model})


class SubjectDialog(QDialog):
    """Subject identity for permanent, local-only storage (name + phone)."""
    def __init__(self, parent=None, name='', phone=''):
        super().__init__(parent)
        self.setWindowTitle('Subject for this recording')
        self.setMinimumWidth(430)
        form = QFormLayout(self)
        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText('e.g. Maria Lopez')
        self.phone_edit = QLineEdit(phone)
        self.phone_edit.setPlaceholderText('e.g. +1 555 010 2030')
        form.addRow('Name', self.name_edit)
        form.addRow('Phone', self.phone_edit)
        note = QLabel('Stored only in this folder on this PC '
                      '(recordings/subjects/<name>_<phone>/). Nothing is uploaded. '
                      'The phone number is used only to keep two subjects with the same '
                      'name apart and is written to the local session files as typed.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#9aadba; font-size:11px;')
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def subject(self):
        return {'name': self.name_edit.text().strip(),
                'phone': self.phone_edit.text().strip()}


class BaselineDialog(QDialog):
    """Condition for one baseline capture (the condition changes interpretation)."""
    def __init__(self, parent=None, condition='eyes closed', source_text=''):
        super().__init__(parent)
        self.setWindowTitle('Set baseline — measurement condition')
        self.setMinimumWidth(470)
        form = QFormLayout(self)
        self.condition_box = QComboBox()
        self.condition_box.addItems(list(baseline.CONDITIONS))
        if condition in baseline.CONDITIONS:
            self.condition_box.setCurrentText(condition)
        form.addRow('Condition at capture', self.condition_box)
        if source_text:
            source = QLabel(source_text)
            source.setWordWrap(True)
            source.setStyleSheet('color:#9aadba; font-size:11px;')
            form.addRow(source)
        note = QLabel('The condition is stored with the baseline and shown with every delta: '
                      'eyes-closed and eyes-open records differ, so only measurements taken in the '
                      'same condition should be compared. The baseline is a descriptive EEG pattern '
                      'reference of this subject\'s own numbers — not a diagnosis and not an '
                      'assessment of the person.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#9aadba; font-size:11px;')
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def condition(self):
        return self.condition_box.currentText()


class HistoryDialog(QDialog):
    """Browse locally stored subjects and reopen a saved session summary."""
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root = Path(root)
        self.setWindowTitle('History — saved subjects and sessions (local only)')
        self.resize(1120, 700)
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.addWidget(QLabel('SUBJECTS'))
        self.subject_list = QListWidget()
        self.subject_list.setMinimumWidth(250)
        left.addWidget(self.subject_list)
        layout.addLayout(left)
        middle = QVBoxLayout()
        middle.addWidget(QLabel('SESSIONS (newest first) · ★ = baseline source'))
        self.session_list = QListWidget()
        self.session_list.setMinimumWidth(230)
        middle.addWidget(self.session_list)
        self.baseline_label = QLabel('No baseline for this subject.')
        self.baseline_label.setWordWrap(True)
        self.baseline_label.setStyleSheet('background:#182730; padding:6px; font-size:11px;')
        middle.addWidget(self.baseline_label)
        self.baseline_button = QPushButton('Set as baseline…')
        self.baseline_button.setToolTip('Store the selected session as this subject\'s baseline '
                                        '("predominant-wave profile"); the measurement condition is '
                                        'asked for and stored with it.')
        middle.addWidget(self.baseline_button)
        middle.addStretch()
        layout.addLayout(middle)
        right = QVBoxLayout()
        right.addWidget(QLabel('SUMMARY AND SAVED CHARTS'))
        self.details = QTextBrowser()
        right.addWidget(self.details, 1)
        self.chart_area = QWidget()
        self.chart_row = QHBoxLayout(self.chart_area)
        self.chart_row.setContentsMargins(4, 4, 4, 4)
        self.chart_scroll = QScrollArea()
        self.chart_scroll.setWidgetResizable(True)
        self.chart_scroll.setWidget(self.chart_area)
        self.chart_scroll.setMinimumHeight(240)
        right.addWidget(self.chart_scroll, 1)
        layout.addLayout(right, 1)
        self.subjects = []
        self.sessions = []
        self.baseline_info = {'present': False, 'record': None, 'error': '', 'path': None}
        self.subject_list.currentRowChanged.connect(self.show_sessions)
        self.session_list.currentRowChanged.connect(self.show_session)
        self.baseline_button.clicked.connect(self.set_baseline_from_selected)
        self.reload()

    def reload(self):
        self.subjects = storage.list_subjects(self.root)
        self.subject_list.clear()
        if not self.subjects:
            self.subject_list.addItem('No saved subjects yet.')
            self.details.setPlainText(
                'Nothing stored yet.\n\nSet a subject in the main window, record, and press '
                '"Save session" — sessions land in:\n'
                f'{self.root}\n\nFolder layout: <name>_<phone>/<timestamp>/ with raw.csv, '
                'summary.json and charts/.')
            return
        for item in self.subjects:
            last = (item.get('last_session_at') or '')[:19].replace('T', ' ')
            self.subject_list.addItem(
                f"{item.get('name') or item['key']} · {item.get('phone') or '—'} · "
                f"{item['sessions']} session(s) · {last}")

    def subject_index(self):
        row = self.subject_list.currentRow()
        return row if 0 <= row < len(self.subjects) else None

    def sessions_with_baseline(self):
        """Session rows: (listed_item, summary, report, source, baseline_line, is_baseline_source).

        Each row is compared against the subject's CURRENT baseline (the stored
        numbers of the session, stable estimate preferred), so a session saved
        before a baseline existed still gets a readable comparison.
        """
        self.baseline_info = {'present': False, 'record': None, 'error': '', 'path': None}
        index = self.subject_index()
        if index is None:
            return []
        subject = self.subjects[index]
        name, phone = subject.get('name') or '', subject.get('phone') or ''
        try:
            self.baseline_info = baseline.load_baseline(self.root, name, phone)
        except (OSError, ValueError) as exc:
            self.baseline_info = {'present': False, 'record': None, 'error': str(exc),
                                  'path': None}
        record = self.baseline_info.get('record')
        if self.baseline_info.get('error'):
            self.baseline_label.setText('Baseline file present but not usable: '
                                        + self.baseline_info['error'] + ' Capture it again to replace it.')
        elif record is None:
            self.baseline_label.setText('No baseline set for this subject. Select a session and press '
                                        '"Set as baseline…", or capture one from the live estimate in the '
                                        'main window.')
        else:
            self.baseline_label.setText(baseline.profile_line(record) + ' · '
                                        + baseline.CONDITION_NOTE)
        rows = []
        for item in self.sessions:
            summary = storage.read_json_object(item['path'] / storage.SUMMARY_FILE) or {}
            report, source = baseline.best_estimate(summary)
            line = ''
            if record is not None:
                line = baseline.one_line(baseline.compare(record, report, source=source or 'stable'))
            rows.append({
                'item': item, 'summary': summary, 'report': report, 'source': source,
                'baseline_line': line,
                'is_source': bool(record is not None
                                  and record.get('session_id') == item.get('session_id')),
            })
        return rows

    def show_sessions(self, _row=None):
        self.session_list.clear()
        self.sessions = []
        index = self.subject_index()
        if index is None:
            self.baseline_label.setText('No subject selected.')
            return
        subject = self.subjects[index]
        self.sessions = storage.list_sessions(self.root, subject.get('name') or '',
                                              subject.get('phone') or '')
        rows = self.sessions_with_baseline()
        if not self.sessions:
            self.session_list.addItem('No sessions with a summary.')
            return
        for row in rows:
            item = row['item']
            started = (item.get('started_at') or '')[:19].replace('T', ' ')
            device = item.get('device_label') or item.get('device_type') or '?'
            text = f"{started} · {device}"
            if row['is_source']:
                text += ' · ★ baseline source'
            if row['baseline_line']:
                text += f"\n{row['baseline_line']}"
            self.session_list.addItem(text)

    def _clear_charts(self, message):
        while self.chart_row.count():
            item = self.chart_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None); widget.deleteLater()
        label = QLabel(message)
        label.setWordWrap(True)
        label.setStyleSheet('color:#9aadba;')
        self.chart_row.addWidget(label)
        self.chart_row.addStretch()

    def show_chart(self, path):
        pixmap = QPixmap(str(path))
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(4, 4, 4, 4)
        caption = QLabel(path.name)
        caption.setStyleSheet('color:#9aadba; font-size:10px;')
        box.addWidget(caption)
        image = QLabel()
        if pixmap.isNull():
            image.setText('could not be displayed')
        else:
            image.setPixmap(pixmap.scaledToWidth(300, Qt.TransformationMode.SmoothTransformation))
        box.addWidget(image)
        box.addStretch()
        return holder

    def show_session(self, _row=None):
        row = self.session_list.currentRow()
        if not (0 <= row < len(self.sessions)):
            return
        try:
            loaded = storage.load_session(self.sessions[row]['path'])
        except (OSError, ValueError) as exc:
            self.details.setPlainText(f'Could not read this session: {exc}')
            self._clear_charts('No charts.')
            return
        summary = loaded['summary']
        lines = [f"Session {summary.get('session_id', '?')}",
                 f"Subject: {summary.get('subject_name', '?')} · {summary.get('subject_phone', '?')}",
                 f"Saved: {summary.get('saved_at', '?')}",
                 f"Device: {summary.get('device_label', summary.get('device_type', '?'))}",
                 f"Channels: {len(summary.get('channels') or [])} · fs {summary.get('fs_hz', '?')} Hz",
                 f"Raw rows: {summary.get('raw_rows', '?')} · duration {summary.get('duration_s', '?')} s",
                 f"Notch: {summary.get('notch_hz', '?')} Hz",
                 f"Raw span: {summary.get('raw_span_note', '')}",
                 f"Stored note: {summary.get('storage_note', '')}",
                 '', 'ANALYSIS (from the saved summary):']
        report = summary.get('analysis') or {}
        stable = summary.get('stable_estimate') or {}
        for name in BANDS:
            band = (report.get('bands') or {}).get(name, {})
            power = band.get('power_uv2')
            relative = band.get('relative_pct')
            stable_power = ((stable.get('bands') or {}).get(name) or {}).get('power_uv2')
            line = (f"  {name}: 5 s {power:.2f} uV^2" if power is not None else f"  {name}: 5 s —")
            if relative is not None:
                line += f" ({relative:.1f}% of 1-45 Hz)"
            if stable_power is not None:
                line += f" · stable {stable_power:.2f} uV^2"
            lines.append(line)
        record = (self.baseline_info or {}).get('record') if isinstance(self.baseline_info, dict) else None
        stored = summary.get('vs_baseline') if isinstance(summary.get('vs_baseline'), dict) else None
        if record is not None or stored is not None:
            lines.append('')
            lines.append('BASELINE (predominant-wave profile of this subject):')
            if record is not None:
                lines.append('  ' + baseline.profile_line(record))
                current, source = baseline.best_estimate(summary)
                label = baseline.SOURCE_LABELS.get(source) if source else 'no usable estimate'
                lines.append(f'  vs this session ({label}): '
                             + baseline.one_line(baseline.compare(record, current,
                                                                  source=source or 'stable')))
                lines.append('  ' + baseline.CONDITION_NOTE)
            else:
                lines.append('  No baseline is stored for this subject now.')
            if stored is not None:
                lines.append('  Stored at save time: ' + baseline.one_line(stored))
        lines.append('')
        lines.append('Signal quality/limitations from the saved summary:')
        lines.append('  ' + (report.get('reason') or 'no reason recorded'))
        lines.append('')
        lines.append('Descriptive research numbers only: not a diagnosis, not a '
                     'mental-state or clinical measurement.')
        self.details.setPlainText('\n'.join(str(line) for line in lines))
        while self.chart_row.count():
            item = self.chart_row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None); widget.deleteLater()
        paths = loaded['charts']
        if not paths:
            self._clear_charts('This session has no saved charts.')
            return
        for path in paths:
            self.chart_row.addWidget(self.show_chart(path))
        self.chart_row.addStretch()

    # ------------------------------------------------------------- baseline
    def select_subject(self, key):
        """Re-select one subject row by folder key (after a reload)."""
        for index, item in enumerate(self.subjects):
            if item.get('key') == key:
                self.subject_list.setCurrentRow(index)
                return
        if self.subjects:
            self.subject_list.setCurrentRow(0)

    def set_baseline_from_selected(self):
        """Store the selected saved session as this subject's baseline."""
        index = self.subject_index()
        row = self.session_list.currentRow()
        if index is None:
            QMessageBox.warning(self, 'Baseline', 'Select a subject first.')
            return
        if not (0 <= row < len(self.sessions)):
            QMessageBox.warning(self, 'Baseline', 'Select the saved session that should become the '
                                                 'baseline first.')
            return
        subject = self.subjects[index]
        item = self.sessions[row]
        summary = storage.read_json_object(item['path'] / storage.SUMMARY_FILE) or {}
        report, source = baseline.best_estimate(summary)
        if report is None:
            QMessageBox.warning(
                self, 'Baseline',
                'This session has no passing estimate (neither a valid stable multi-epoch estimate '
                'nor a valid five-second analysis), so it cannot become a baseline. A baseline is a '
                'reference profile, not a placeholder.')
            return
        dialog = BaselineDialog(
            self, source_text=f'Session {item.get("session_id") or "?"} of '
                              f'{subject.get("name") or "?"}; the estimate used is the '
                              f'{baseline.SOURCE_LABELS.get(source, source)}.')
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            record = baseline.build_baseline(
                report, dialog.condition(),
                subject_name=subject.get('name') or '', subject_phone=subject.get('phone') or '',
                device={'device_key': summary.get('device_type'),
                        'device_label': summary.get('device_label'),
                        'channel_source': summary.get('channel_source')},
                channel_source=summary.get('channel_source'),
                channels=summary.get('channels') or None,
                fs_hz=summary.get('fs_hz'), source=source,
                session_id=item.get('session_id'), captured_from='saved_session')
            path = baseline.save_baseline(self.root, subject.get('name') or '',
                                          subject.get('phone') or '', record)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Baseline', f'Baseline not saved: {exc}')
            return
        message = (f'Baseline stored locally: {path} (condition: {dialog.condition()}, from the '
                   f'{baseline.SOURCE_LABELS.get(source, source)}). It replaces any earlier baseline of '
                   'this subject.')
        if source != 'stable':
            message += ' Only the five-second window was available: this is a weaker reference.'
        QMessageBox.information(self, 'Baseline stored', message)
        key = subject.get('key')
        self.reload()
        self.select_subject(key)


class NeuroSyncApp(QMainWindow):
    def __init__(self, autostart=True, smoke=False, simulated=None):
        super().__init__()
        self.simulated = simulated          # 'brainbit' | 'dragon' | None (real BLE)
        self.second_window = None
        self.setWindowTitle('NeuroSync | Live EEG • research viewer (BrainBit + DragonEEG)')
        self.resize(1600, 960)
        self.setMinimumSize(1180, 820)
        self.setStyleSheet('''QWidget { background:#101820; color:#e3edf4; font-size:12px; }
            QPushButton,QComboBox { background:#20323e; padding:9px; border:1px solid #466070; border-radius:4px; }
            QPushButton:disabled { color:#667782; } QPushButton:hover { background:#304856; }
            QTableWidget { gridline-color:#314550; background:#13212c; }
            QHeaderView::section { background:#20323e; color:#cddfe8; padding:6px; }
            QLineEdit, QTextBrowser, QListWidget { background:#13212c; border:1px solid #466070; padding:5px; }
            QStatusBar { color:#7ce0c5; }''')
        pg.setConfigOptions(antialias=True, background='#101820', foreground='#b8cbd6')
        self.channels = tuple(CHANNELS)
        self.fs = devices.CLASSIC.fs
        self.samples = deque(maxlen=BUFFER_SECONDS * self.fs)
        self.packet_numbers = deque(maxlen=BUFFER_SECONDS * self.fs)
        self.last_received = None
        self.contact = None
        self.last_report = None
        self.last_stable = None
        self.ai_note = None
        self.ai_worker = None
        self.ai_requested_at = None
        self.chat_history = []          # previous live-chat turns (bounded)
        self.chat_worker = None
        self.chat_enabled = False
        self.chat_updates = 0
        self.chat_last_at = None
        self.chat_interval_s = CHAT_DEFAULT_INTERVAL_S
        self._chat_cursor = None        # where streamed chunks are inserted
        self.device_metadata = {}
        self.transport_metrics = {}
        self.known_devices = []
        self.saved_address = SAVED_ADDRESS
        self.subject = None
        self.baseline = None            # validated baseline record of the current subject
        self.baseline_error = ''
        self.baseline_delta = None      # last computed deltas of the current estimate
        self.session_started_at = None
        self.total_received = 0
        self.plot_updates = 0
        self.session_count = 0
        self.client = None
        self.runtime_enabled = autostart and not smoke
        self._build_ui()
        self._install_client()
        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(200)
        self.analysis_timer = QTimer(self)
        self.analysis_timer.timeout.connect(self.analyze_now)
        self.analysis_timer.start(1000)
        self.stable_timer = QTimer(self)
        self.stable_timer.timeout.connect(self.analyze_stable_now)
        self.stable_timer.start(2000)
        self.statusBar().showMessage('Disconnected — no measurements yet')
        self.refresh_history_summary()
        self.load_baseline()
        if autostart and not smoke:
            self.start_connection()

    # ------------------------------------------------------------------ layout
    def _panel(self, title, note=None):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(title))
        if note:
            label = QLabel(note)
            label.setWordWrap(True)
            label.setStyleSheet('color:#9aadba; font-size:11px;')
            layout.addWidget(label)
        return widget, layout

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        heading = QLabel('NEUROSYNC  /  LIVE EEG RESEARCH VIEWER')
        heading.setStyleSheet('font-size:21px; font-weight:700; color:#8be6ce; padding:6px;')
        layout.addWidget(heading)

        toolbar = QHBoxLayout()
        self.scan_button = QPushButton('Scan devices')
        self.connect_button = QPushButton('Connect')
        self.disconnect_button = QPushButton('Disconnect')
        self.contact_button = QPushButton('Check electrodes (pauses EEG)')
        self.save_button = QPushButton('Save session (permanent)')
        self.history_button = QPushButton('History…')
        self.second_button = QPushButton('Second device window…')
        self.report_button = QPushButton('Patient report (HTML)…')
        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(360)
        self.device_combo.addItem('No scan yet — press "Scan devices"', None)
        for widget in (self.scan_button, self.device_combo, self.connect_button,
                       self.disconnect_button, self.contact_button,
                       self.save_button, self.history_button,
                       self.second_button, self.report_button):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        controls = QHBoxLayout()
        self.subject_button = QPushButton('Set subject…')
        self.subject_label = QLabel('Subject: not set (required before saving)')
        self.subject_label.setStyleSheet('color:#f0c47b;')
        controls.addWidget(self.subject_button); controls.addWidget(self.subject_label)
        controls.addStretch()
        controls.addWidget(QLabel('Band waveform channel'))
        self.channel_select = QComboBox(); self.channel_select.addItems(list(self.channels))
        controls.addWidget(self.channel_select)
        controls.addWidget(QLabel('Mains notch'))
        self.notch_select = QComboBox(); self.notch_select.addItems(['60 Hz', '50 Hz', 'Off'])
        controls.addWidget(self.notch_select)
        controls.addWidget(QLabel('Head map band'))
        self.map_band_select = QComboBox(); self.map_band_select.addItems(list(BANDS))
        controls.addWidget(self.map_band_select)
        self.export5_button = QPushButton('Save latest 5 s (quick CSV)')
        controls.addWidget(self.export5_button)
        layout.addLayout(controls)

        self.connection_label = QLabel('BLE disconnected | no device selected | 0 samples')
        self.connection_label.setStyleSheet('padding:7px; background:#182730; color:#b9cad7;')
        layout.addWidget(self.connection_label)
        self.device_notes = QLabel(
            'DragonEEG first pairing: put the headset in pairing mode before scanning; the SDK '
            'then performs the BLE pairing. Amplifier parameters (frequency, per-channel gain and '
            'mode) are written automatically after connect and verified before streaming — an '
            'unconfigured Dragon streams flat zeros. Resistances are read in ohms per channel with '
            'A1/A2/Bias reference values.')
        self.device_notes.setWordWrap(True)
        self.device_notes.setStyleSheet('padding:6px; background:#16232c; color:#9fb6c3; font-size:11px;')
        layout.addWidget(self.device_notes)

        split = QSplitter(Qt.Orientation.Horizontal); layout.addWidget(split, 1)

        # --- left: raw channel grid + frequency components -------------------
        left_side = QWidget(); left = QVBoxLayout(left_side)
        self.raw_heading = QLabel(f'MEASURED CHANNELS · raw microvolts · unfiltered · {len(self.channels)} channels')
        left.addWidget(self.raw_heading)
        self.channel_grid_holder = QWidget()
        self.channel_grid = QGridLayout(self.channel_grid_holder)
        self.channel_grid.setSpacing(2)
        left.addWidget(self.channel_grid_holder)
        self.raw_curves = {}
        self.raw_plots = {}
        self._build_channel_plots(self.channels)
        self.band_heading = QLabel('FREQUENCY COMPONENTS (display only, zero-phase filters) · µV')
        left.addWidget(self.band_heading)
        self.band_curves = {}; self.band_plots = {}
        for name, (low, high) in BANDS.items():
            plot = pg.PlotWidget(title=f'<span style="color:{COLORS[name]}">{name} · {low:g}–{high:g} Hz</span>')
            plot.setLabel('left', 'µV'); plot.setLabel('bottom', 'Time relative to now', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.setMinimumHeight(90)
            self.band_curves[name] = plot.plot(pen=pg.mkPen(COLORS[name], width=1.3))
            self.band_plots[name] = plot
            left.addWidget(plot, 1)
        left_scroll = QScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setWidget(left_side)
        split.addWidget(left_scroll)

        # --- middle: results graphics ---------------------------------------
        middle_side = QWidget(); middle = QVBoxLayout(middle_side)
        bars_panel, bars = self._panel(
            'BAND POWER · latest 5 s · µV² (absolute) and % of 1–45 Hz',
            'Bars are the measured Welch band powers, averaged over the analysed channels. '
            'Descriptive only.')
        self.bars_plot = pg.PlotWidget(axisItems={'bottom': BandAxis(orientation='bottom')})
        self.bars_plot.setLabel('left', 'power', units='µV²')
        self.bars_plot.setMinimumHeight(190)
        self.bars_absolute = pg.BarGraphItem(x=[], height=[], width=.6)
        self.bars_relative = pg.BarGraphItem(x=[], height=[], width=.6, brush='#3f6379')
        self.bars_plot.addItem(self.bars_relative)
        self.bars_plot.addItem(self.bars_absolute)
        bars.addWidget(self.bars_plot)
        middle.addWidget(bars_panel)

        psd_panel, psd = self._panel(
            'POWER SPECTRAL DENSITY · mean over channels · µV²/Hz (Welch, Hann 2 s, 50% overlap)',
            'Display curve only: band powers in the tables come from the per-channel integrated '
            'Welch path, never from this chart.')
        self.psd_plot = pg.PlotWidget()
        self.psd_plot.setLabel('left', 'PSD', units='µV²/Hz')
        self.psd_plot.setLabel('bottom', 'frequency', units='Hz')
        self.psd_plot.setLogMode(False, True)
        self.psd_plot.setMinimumHeight(190)
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen('#7fd1f7', width=1.4))
        psd.addWidget(self.psd_plot)
        middle.addWidget(psd_panel)

        avg_panel, avg = self._panel(
            'AVERAGE BRAIN WAVE · mean across every channel · 1–45 Hz components · display only',
            'One descriptive curve: the average of all connected channels, band-limited to '
            '1–45 Hz (notch applied), summed back from the five band components. It shows the '
            'shared rhythm of the montage, not an anatomical source. For a second headset at '
            'the same time, open the second device window.')
        self.avg_plot = pg.PlotWidget()
        self.avg_plot.setLabel('left', 'µV')
        self.avg_plot.setLabel('bottom', 'Time relative to now', units='s')
        self.avg_plot.setMinimumHeight(150)
        self.avg_plot.showGrid(x=True, y=True, alpha=.15)
        self.avg_curve = self.avg_plot.plot(pen=pg.mkPen('#9be8d0', width=1.4))
        avg.addWidget(self.avg_plot)
        middle.addWidget(avg_panel)

        map_panel, topo = self._panel(
            'BRAIN MAP · band-power heat map · schematic 10-20 layout (%), scipy-interpolated',
            'Schematic drawing positions and engineering interpolation: NOT a source localisation, '
            'NOT a clinical brain map. Maps need at least three usable electrodes.')
        self.map_plot = pg.PlotWidget()
        self.map_plot.setAspectLocked(True)
        self.map_plot.setMinimumHeight(330)
        self.map_plot.setRange(xRange=(-1.25, 1.25), yRange=(-1.25, 1.25), padding=0)
        self.map_image = pg.ImageItem()
        try:
            self.map_cmap = pg.colormap.get('turbo')     # heat-map colour ramp
        except Exception:
            self.map_cmap = pg.colormap.get('jet')
        self.map_image.setLookupTable(self.map_cmap.getLookupTable(nPts=256))
        self.map_plot.addItem(self.map_image)
        outline_x, outline_y = topomap.head_outline(181)
        self.map_plot.plot(outline_x, outline_y, pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_plot.plot([-.22, 0, .22], [.97, 1.14, .97], pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_plot.plot([-1, -1.12, -1.12, -1], [-.22, -.12, .12, .22], pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_plot.plot([1, 1.12, 1.12, 1], [-.22, -.12, .12, .22], pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_plot.plot([-.62, .62], [-1.06, -1.06], pen=pg.mkPen('#6f8ea0', width=1.4))
        self.map_electrodes = pg.ScatterPlotItem(size=7, brush=pg.mkBrush('#e3edf4'),
                                                 pen=pg.mkPen('#20323e'))
        self.map_plot.addItem(self.map_electrodes)
        try:
            self.map_bar = pg.ColorBarItem(colorMap=self.map_cmap, interactive=False,
                                           width=16, label='relative power (%)')
            self.map_bar.setImageItem(self.map_image, insert_in=self.map_plot.getPlotItem())
        except Exception:
            self.map_bar = None          # colour ramp still applies without the scale bar
        topo.addWidget(self.map_plot)
        map_controls = QHBoxLayout()
        self.map_status = QLabel('Waiting for a valid five-second window.')
        self.map_status.setWordWrap(True)
        self.map_status.setStyleSheet('color:#9aadba; font-size:11px;')
        self.map_export_button = QPushButton('Export head map PNG…')
        map_controls.addWidget(self.map_status, 1)
        map_controls.addWidget(self.map_export_button)
        topo.addLayout(map_controls)
        middle.addWidget(map_panel)

        trend_panel, trend = self._panel(
            'CROSS-SESSION TREND · per-band relative power (%) for the selected subject',
            'One point per saved session (stable multi-epoch estimate when available, otherwise the '
            '5 s window). Session-to-session differences also reflect electrode placement, contact '
            'and movement, not only physiology.')
        self.trend_plot = pg.PlotWidget()
        self.trend_plot.setLabel('left', 'relative power', units='% of 1–45 Hz')
        self.trend_plot.setLabel('bottom', 'saved session (oldest → newest)')
        self.trend_plot.setMinimumHeight(210)
        self.trend_plot.addLegend(offset=(6, 6))
        self.trend_plot.showGrid(x=True, y=True, alpha=.15)
        self.trend_curves = {}
        trend.addWidget(self.trend_plot)
        self.trend_status = QLabel('Set a subject and save sessions to see a trend.')
        self.trend_status.setWordWrap(True)
        self.trend_status.setStyleSheet('color:#9aadba; font-size:11px;')
        trend.addWidget(self.trend_status)
        middle.addWidget(trend_panel)
        middle_scroll = QScrollArea(); middle_scroll.setWidgetResizable(True); middle_scroll.setWidget(middle_side)
        split.addWidget(middle_scroll)

        # --- right: numbers, quality, AI doctor ------------------------------
        side = QWidget(); right = QVBoxLayout(side)
        right.addWidget(QLabel('LATEST 5 SECONDS  ·  refreshed every second'))
        self.conclusion = QLabel('Waiting for five seconds of continuous EEG.')
        self.conclusion.setWordWrap(True)
        self.conclusion.setStyleSheet('font-size:17px; padding:12px; background:#20323e;')
        right.addWidget(self.conclusion)
        self.predominant = QLabel('PREDOMINANT RHYTHM · — (waiting for a valid window)')
        self.predominant.setWordWrap(True)
        self.predominant.setStyleSheet('font-size:15px; padding:9px; background:#17262f; color:#e3edf4;')
        right.addWidget(self.predominant)
        self.progress_label = QLabel('Window: 0 / 1250 samples')
        right.addWidget(self.progress_label)
        self.power_table = QTableWidget(5, 3)
        self.power_table.setHorizontalHeaderLabels(['Band', 'Power µV²', 'Relative %'])
        self.power_table.verticalHeader().hide()
        self.power_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.power_table.setMaximumHeight(205)
        self.power_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(BANDS):
            item = QTableWidgetItem(name); item.setForeground(QColor(COLORS[name]))
            self.power_table.setItem(row, 0, item)
            self.power_table.setItem(row, 1, QTableWidgetItem('—'))
            self.power_table.setItem(row, 2, QTableWidgetItem('—'))
        right.addWidget(self.power_table)
        note = QLabel('Mean band power across every analysed channel; relative share of 1–45 Hz. '
                      'Five seconds is a short descriptive window — not a diagnosis, emotion or '
                      'sleep-stage conclusion.')
        note.setWordWrap(True); right.addWidget(note)
        right.addWidget(QLabel('PER-CHANNEL · 5 s window'))
        self.channel_table = QTableWidget(0, 4)
        self.channel_table.setHorizontalHeaderLabels(['Channel', 'Power µV²', 'Relative %', 'Dominant'])
        self.channel_table.verticalHeader().hide()
        self.channel_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.channel_table.setMinimumHeight(180)
        self.channel_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right.addWidget(self.channel_table)
        right.addWidget(QLabel('STABLE ESTIMATE  ·  2 s clean epochs  ·  ± standard error'))
        self.stable_status = QLabel('Collecting: 0 / 3000 samples (six 2 s epochs minimum; window '
                                    'builds to 60 s).')
        self.stable_status.setWordWrap(True); right.addWidget(self.stable_status)
        self.stable_table = QTableWidget(5, 3)
        self.stable_table.setHorizontalHeaderLabels(['Band', 'Power ±SEM µV²', 'Relative %'])
        self.stable_table.verticalHeader().hide()
        self.stable_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.stable_table.setMaximumHeight(190)
        self.stable_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(BANDS):
            item = QTableWidgetItem(name); item.setForeground(QColor(COLORS[name]))
            self.stable_table.setItem(row, 0, item)
            self.stable_table.setItem(row, 1, QTableWidgetItem('—'))
            self.stable_table.setItem(row, 2, QTableWidgetItem('—'))
        right.addWidget(self.stable_table)
        stable_note = QLabel('Averages many clean 2 s epochs and excludes artifact epochs (moves, '
                             'blinks, clenching, mains), so it is more stable than the single 5 s '
                             'view. Descriptive only; not a diagnosis.')
        stable_note.setWordWrap(True); stable_note.setStyleSheet('color:#9aadba; font-size:11px;')
        right.addWidget(stable_note)
        right.addWidget(QLabel('BASELINE · per-subject predominant-wave reference (local only)'))
        self.baseline_status = QLabel('No baseline set — set a subject, then capture one from a '
                                      'passing estimate.')
        self.baseline_status.setWordWrap(True)
        self.baseline_status.setStyleSheet('background:#182730; padding:8px; font-size:11px;')
        right.addWidget(self.baseline_status)
        baseline_row = QHBoxLayout()
        baseline_row.addWidget(QLabel('Condition'))
        self.baseline_condition = QComboBox()
        self.baseline_condition.addItems(list(baseline.CONDITIONS))
        baseline_row.addWidget(self.baseline_condition)
        self.baseline_button = QPushButton('Set as baseline')
        self.baseline_button.setToolTip('Store the current passing estimate as this subject\'s '
                                        'baseline ("predominant-wave profile"). The stable multi-epoch '
                                        'estimate is preferred; a valid five-second window is only '
                                        'used when no stable estimate is available yet, and the source '
                                        'is recorded with the baseline.')
        baseline_row.addWidget(self.baseline_button)
        baseline_row.addStretch()
        right.addLayout(baseline_row)
        self.baseline_delta_label = QLabel('vs baseline: —')
        self.baseline_delta_label.setWordWrap(True)
        self.baseline_delta_label.setStyleSheet('font-size:11px;')
        right.addWidget(self.baseline_delta_label)
        self.baseline_table = QTableWidget(5, 4)
        self.baseline_table.setHorizontalHeaderLabels(['Band', 'Baseline %', 'Now %', 'Δ pp'])
        self.baseline_table.verticalHeader().hide()
        self.baseline_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.baseline_table.setMaximumHeight(190)
        self.baseline_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        for row, name in enumerate(BANDS):
            item = QTableWidgetItem(name); item.setForeground(QColor(COLORS[name]))
            self.baseline_table.setItem(row, 0, item)
            for column in (1, 2, 3):
                self.baseline_table.setItem(row, column, QTableWidgetItem('—'))
        right.addWidget(self.baseline_table)
        self.baseline_note = QLabel(baseline.BASELINE_NOTE + ' ' + baseline.CONDITION_NOTE)
        self.baseline_note.setWordWrap(True)
        self.baseline_note.setStyleSheet('color:#9aadba; font-size:11px;')
        right.addWidget(self.baseline_note)
        right.addWidget(QLabel('LOCAL HISTORY  ·  recordings/subjects/ (never uploaded)'))
        self.history_label = QLabel('No saved sessions yet.')
        self.history_label.setWordWrap(True)
        self.history_label.setStyleSheet('background:#182730; padding:8px; font-size:11px;')
        right.addWidget(self.history_label)
        self.contact_title = QLabel('ELECTRODE CONTACT  ·  not checked')
        right.addWidget(self.contact_title)
        self.contact_holder = QWidget()
        self.contact_layout = QVBoxLayout(self.contact_holder)
        self.contact_layout.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.contact_holder)
        self.contact_labels = {}
        self._build_contact_labels(self.channels)
        self.quality_label = QLabel('Signal quality unverified. Keep still; relax jaw; avoid blinking '
                                    'during a short study.')
        self.quality_label.setWordWrap(True)
        self.quality_label.setStyleSheet('color:#f0c47b; padding:8px;')
        right.addWidget(self.quality_label)
        right.addWidget(QLabel('AI DOCTOR  ·  local model  ·  assistive only'))
        ai_row = QHBoxLayout()
        self.ai_button = QPushButton('Ask AI doctor')
        self.ai_auto = QCheckBox('auto every 60 s')
        self.ai_button.clicked.connect(self.run_ai_doctor)
        ai_row.addWidget(self.ai_button); ai_row.addWidget(self.ai_auto); ai_row.addStretch()
        right.addLayout(ai_row)
        self.ai_status = QLabel('Model: ' + ai_doctor.DEFAULT_MODEL
                                + ' · local endpoint 127.0.0.1:11434 · offline check on use')
        self.ai_status.setWordWrap(True); self.ai_status.setStyleSheet('color:#9aadba; font-size:11px;')
        right.addWidget(self.ai_status)
        self.ai_text = QLabel('Ask for an assistive description of the latest measured numbers. Only '
                              'derived values (band powers, quality flags, contact, transport '
                              'counters) reach the local model; raw EEG never leaves this PC. Not a '
                              'medical device and not a diagnosis.')
        self.ai_text.setWordWrap(True)
        self.ai_text.setStyleSheet('background:#182730; padding:10px; font-size:11px;')
        right.addWidget(self.ai_text)
        right.addWidget(QLabel('LIVE CHAT  ·  local model monitor  ·  updates while recording'))
        chat_row = QHBoxLayout()
        self.chat_toggle = QPushButton('▶ Start live monitor')
        self.chat_toggle.setToolTip('Post short local-model observations while a recording '
                                    'runs. Only derived numbers reach the model; nothing '
                                    'leaves this PC.')
        self.chat_every = QComboBox()
        for _seconds in CHAT_INTERVAL_CHOICES:
            self.chat_every.addItem(f'every {_seconds} s', _seconds)
        self.chat_every.setCurrentIndex(
            CHAT_INTERVAL_CHOICES.index(int(CHAT_DEFAULT_INTERVAL_S)))
        self.chat_clear = QPushButton('Clear')
        chat_row.addWidget(self.chat_toggle)
        chat_row.addWidget(self.chat_every)
        chat_row.addWidget(self.chat_clear)
        chat_row.addStretch()
        right.addLayout(chat_row)
        self.chat_status = QLabel('Off — start it to get short updates while a recording runs. '
                                  'The local model describes the measured numbers only; '
                                  'assistive, not a diagnosis.')
        self.chat_status.setWordWrap(True)
        self.chat_status.setStyleSheet('color:#9aadba; font-size:11px;')
        right.addWidget(self.chat_status)
        self.chat_log = QTextBrowser()
        self.chat_log.setMinimumHeight(230)
        self.chat_log.setStyleSheet('QTextBrowser { background:#12181f; color:#dbe6ee; '
                                    'border:1px solid #26333f; }')
        right.addWidget(self.chat_log)
        chat_send_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText('Ask about the latest reading…')
        self.chat_send = QPushButton('Send')
        chat_send_row.addWidget(self.chat_input)
        chat_send_row.addWidget(self.chat_send)
        right.addLayout(chat_send_row)
        chat_note = QLabel('Streamed in live from the same local endpoint as the AI doctor. '
                           'Nothing is uploaded; simulated sessions stay labelled. Not a '
                           'medical interpretation.')
        chat_note.setWordWrap(True); chat_note.setStyleSheet('color:#9aadba; font-size:11px;')
        right.addWidget(chat_note)
        self.chat_toggle.clicked.connect(self.toggle_chat_monitor)
        self.chat_clear.clicked.connect(self.clear_chat)
        self.chat_send.clicked.connect(self.send_chat_question)
        self.chat_input.returnPressed.connect(self.send_chat_question)
        self.chat_every.currentIndexChanged.connect(self._chat_interval_changed)
        right.addStretch()
        scroller = QScrollArea(); scroller.setWidgetResizable(True); scroller.setWidget(side)
        self.right_scroller = scroller
        scroller.setMinimumWidth(360); scroller.setMaximumWidth(520)
        scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        split.addWidget(scroller); split.setSizes([560, 470, 400])

        footer = QLabel('Raw EEG is unchanged. Colored components are window-filtered views with edge '
                        'effects. Power uses Welch PSD, not FFT amplitude. Head maps are schematic '
                        'interpolations. Contact checks and interruptions restart the five-second '
                        'window. All data stays on this PC.')
        footer.setWordWrap(True); footer.setStyleSheet('color:#9aadba; padding:6px;')
        layout.addWidget(footer)

        # signals
        self.scan_button.clicked.connect(self.scan_devices)
        self.connect_button.clicked.connect(self.start_connection)
        self.disconnect_button.clicked.connect(self.disconnect_device)
        self.contact_button.clicked.connect(self.request_contact)
        self.save_button.clicked.connect(self.save_session_clicked)
        self.history_button.clicked.connect(self.show_history)
        self.subject_button.clicked.connect(self.set_subject)
        self.export5_button.clicked.connect(self.save_clicked)
        self.map_export_button.clicked.connect(self.export_head_map)
        self.baseline_button.clicked.connect(self.set_baseline_from_current)
        self.second_button.clicked.connect(self.open_second_device)
        self.report_button.clicked.connect(self.export_patient_report)
        self.channel_select.currentIndexChanged.connect(lambda _: self.update_plots())
        self.notch_select.currentIndexChanged.connect(lambda _: self.analyze_now())
        self.map_band_select.currentIndexChanged.connect(lambda _: self.update_topomap())

    def _build_channel_plots(self, channels):
        while self.channel_grid.count():
            item = self.channel_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None); widget.deleteLater()
        self.raw_curves = {}; self.raw_plots = {}
        columns = 3 if len(channels) <= 6 else 4
        for index, name in enumerate(channels):
            color = CLASSIC_COLORS.get(name) or CHANNEL_PALETTE[index % len(CHANNEL_PALETTE)]
            plot = pg.PlotWidget(title=name)
            plot.setLabel('left', 'µV')
            plot.setLabel('bottom', 'Time relative to now', units='s')
            plot.showGrid(x=True, y=True, alpha=.15)
            plot.setMinimumHeight(104)
            self.raw_curves[name] = plot.plot(pen=pg.mkPen(color, width=1))
            self.raw_plots[name] = plot
            self.channel_grid.addWidget(plot, index // columns, index % columns)

    def _build_contact_labels(self, channels):
        while self.contact_layout.count():
            item = self.contact_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None); widget.deleteLater()
        self.contact_labels = {}
        for name in channels:
            label = QLabel(f'{name}: unverified')
            self.contact_labels[name] = label
            self.contact_layout.addWidget(label)
        self.contact_aux_label = QLabel('A1/A2/Bias: not checked')
        self.contact_aux_label.setStyleSheet('color:#9aadba; font-size:11px;')
        self.contact_layout.addWidget(self.contact_aux_label)

    def _reset_buffers(self):
        self.samples = deque(maxlen=BUFFER_SECONDS * self.fs)
        self.packet_numbers = deque(maxlen=BUFFER_SECONDS * self.fs)

    def apply_device_info(self, info):
        """Adopt the connected device's channel table and rate (from the SDK)."""
        channels = tuple(info.get('channels') or self.channels)
        fs = int(info.get('fs') or self.fs)
        changed = channels != self.channels or fs != self.fs
        self.channels = channels
        self.fs = fs
        if changed:
            self._reset_buffers()
            self._build_channel_plots(channels)
            self._build_contact_labels(channels)
            self.channel_select.clear()
            self.channel_select.addItems(list(channels))
            self.raw_heading.setText(f'MEASURED CHANNELS · raw microvolts · unfiltered · '
                                     f'{len(channels)} channels')
            self.map_electrodes.setData(pos=np.asarray(
                [topomap.ELECTRODE_POSITIONS[name] for name in devices.scalp_channels(channels)]
            ).reshape(-1, 2) if devices.scalp_channels(channels) else [])
        self.device_metadata = info
        label = (f"{info.get('device_label', 'device')} | serial {info.get('serial', '—')} | "
                 f"battery {info.get('battery', '—')}% | {fs} Hz | {len(channels)} channels "
                 f"({len(info.get('scalp_channels') or [])} scalp + "
                 f"{len(info.get('poly_channels') or [])} aux) | "
                 f"channel table: {info.get('channel_source', 'unknown')}")
        self.connection_label.setText(label)
        self.statusBar().showMessage(
            f"Connected: {info.get('device_label')} · configuration: "
            f"{info.get('configuration', {}).get('amplifier', 'n/a')}")

    # ------------------------------------------------------------------ client
    def _install_client(self, **kwargs):
        options = dict(auto_contact=True, target_address=self.saved_address)
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

    def _chosen_address(self):
        data = self.device_combo.currentData()
        return data.get('address') if isinstance(data, dict) else None

    def scan_devices(self):
        """One scan covering BrainBit Classic/Black/2 and DragonEEG."""
        if self.simulated:
            self.statusBar().showMessage('Simulated mode: no scan needed — starting the '
                                         'simulated stream instead.')
            self.start_connection()
            return
        if self.client is not None and self.client.isRunning():
            self.statusBar().showMessage('A scan or stream is already running.')
            return
        self.known_devices = []
        self.device_combo.clear()
        self.device_combo.addItem('Scanning…', None)
        self._install_client(selection_required=True, selection_timeout=600.,
                             target_address=self.saved_address)
        self.reset_window('Scanning for devices; no EEG conclusion yet')
        self.client.start()
        self.update_buttons()

    def start_connection(self):
        if self.client is not None and self.client.isRunning():
            if getattr(self.client, 'mode', None) == 'awaiting_selection':
                address = self._chosen_address()
                if address:
                    self.client.select_device(address)
                    self.statusBar().showMessage(f'Connecting to the selected device {address}…')
                else:
                    self.statusBar().showMessage('Select a device in the list first.')
            return
        if self.simulated:
            self._install_client(auto_contact=True)
        else:
            address = self._chosen_address() or self.saved_address
            self._install_client(target_address=address, selection_required=False)
        self.contact = None; self.device_metadata = {}; self.transport_metrics = {}
        self.reset_window('Connecting; no valid EEG conclusion yet')
        self.client.start()
        self.update_buttons()

    def disconnect_device(self):
        if self.client:
            self.client.stop()
        self.contact = None
        self.reset_window('Disconnected — no live conclusion')
        self.update_buttons()

    def request_contact(self):
        if self.client and self.client.isRunning():
            self.contact = None
            self.reset_window('Electrode check requested; EEG will pause')
            self.client.check_contact()

    def handle_devices(self, described):
        self.known_devices = described
        self.device_combo.clear()
        if not described:
            self.device_combo.addItem('No devices found', None)
            self.statusBar().showMessage(
                'No devices found. Power the headset on (DragonEEG: put it in pairing mode the '
                'first time), keep it close, close other BrainBit apps, then scan again.')
            self.connection_label.setText('No devices found — the app keeps running; scan again when '
                                          'the headset is powered on.')
            return
        for item in described:
            channels = len(item.get('channels') or [])
            label = (f"{item['device_label']} · {item['address']}"
                     f"{'' if item['supported'] else ' (unsupported family)'}"
                     f" · {channels} channels · RSSI {item.get('rssi')}"
                     f"{' · pairing required' if item.get('pairing_required') else ''}")
            self.device_combo.addItem(label, item)
        self.statusBar().showMessage(f"{len(described)} device(s) found — select one, then Connect.")
        self.device_combo.setCurrentIndex(0)

    # ------------------------------------------------------------- device data
    def handle_batch(self, batch):
        channels = tuple(batch.get('channels') or self.channels)
        if channels != self.channels:
            self.channels = channels
            self._reset_buffers()
            self._build_channel_plots(channels)
            self._build_contact_labels(channels)
        self.samples.extend(batch['samples_uv'])
        self.packet_numbers.extend(batch['packnums'])
        self.last_received = batch['arrived_monotonic']
        self.total_received = batch['sample_count']

    def handle_status(self, text):
        self.statusBar().showMessage(text)
        self.update_buttons()

    def handle_device(self, info):
        self.apply_device_info(info)
        if not self.session_started_at:
            self.session_started_at = time.monotonic()

    def handle_metrics(self, metrics):
        self.transport_metrics = metrics
        rate = metrics.get('rate_hz')
        rate_text = f'{rate:.1f} Hz received' if rate is not None else 'measuring received rate'
        self.connection_label.setText(
            f"{self.device_metadata.get('device_label', 'device')} "
            f"{self.device_metadata.get('serial', '—')} | battery {metrics.get('battery', '—')}% | "
            f"{rate_text} | {metrics['samples']:,} samples | {metrics['gap_events']} discontinuities"
            f" | {metrics.get('counter_anomalies', 0)} packet-counter anomalies")
        self.update_buttons()

    def update_buttons(self):
        running = self.client is not None and self.client.isRunning()
        self.connect_button.setEnabled(not running)
        self.disconnect_button.setEnabled(running)
        self.contact_button.setEnabled(running and self.client.mode == 'signal')
        self.scan_button.setEnabled(not running)
        self.save_button.setEnabled(len(self.samples) >= self.fs * 5)
        self.export5_button.setEnabled(len(self.samples) >= self.fs * 5)
        busy = self.ai_worker is not None and self.ai_worker.isRunning()
        self.ai_button.setEnabled(not busy and (self.last_report is not None or self.last_stable is not None))
        if widget_alive(getattr(self, 'chat_send', None)):
            chat_busy = self.chat_worker is not None and self.chat_worker.isRunning()
            self.chat_send.setEnabled(not chat_busy)

    # ---------------------------------------------------------------- subjects
    def set_subject(self):
        dialog = SubjectDialog(self, **(self.subject or {}))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        subject = dialog.subject()
        if not subject['name']:
            QMessageBox.warning(self, 'Subject required',
                                'A subject name is required so the recording can be stored under '
                                'recordings/subjects/.')
            return
        self.subject = subject
        self.subject_label.setText(f"Subject: {subject['name']} · {subject['phone'] or 'no phone'}")
        self.subject_label.setStyleSheet('color:#7cddaf;')
        self.refresh_trend()
        self.load_baseline()

    # --------------------------------------------------------------- baseline
    def load_baseline(self):
        """(Re)read this subject's baseline.json; never raises, never guesses."""
        self.baseline = None
        self.baseline_error = ''
        if not self.subject or not self.subject.get('name'):
            self.baseline_error = 'no subject set'
        else:
            try:
                info = baseline.load_baseline(SUBJECT_ROOT, self.subject['name'],
                                              self.subject.get('phone', ''))
                self.baseline = info.get('record')
                self.baseline_error = info.get('error') or ''
            except (OSError, ValueError) as exc:
                self.baseline_error = str(exc)
        self.update_baseline_panel()

    def _current_estimate(self):
        """Best current estimate: a valid stable multi-epoch value, else a valid 5 s window."""
        if self.last_stable and self.last_stable.get('valid'):
            return self.last_stable, 'stable'
        if self.last_report and self.last_report.get('valid'):
            return self.last_report, 'five_second'
        return None, None

    def update_baseline_panel(self):
        """Profile line + deltas of the current estimate; refreshed on analysis ticks only."""
        if not self.subject or not self.subject.get('name'):
            self.baseline_status.setText('No baseline set — a baseline is stored per subject; set a '
                                         'subject first, then capture one from a passing estimate.')
        elif self.baseline_error:
            self.baseline_status.setText(f'Baseline file present but not usable: {self.baseline_error} '
                                         '— capture again to replace it.')
        elif self.baseline is None:
            self.baseline_status.setText('No baseline set for this subject. Record a passing estimate '
                                         '(stable preferred), pick the condition, then press '
                                         '"Set as baseline". Capturing again replaces the baseline.')
        else:
            self.baseline_status.setText(baseline.profile_line(self.baseline))
        self.update_baseline_deltas()

    def update_baseline_deltas(self):
        report, source = self._current_estimate()
        delta = baseline.compare(self.baseline, report, source=source or 'stable',
                                 device=self.device_metadata)
        self.baseline_delta = delta
        if not delta.get('available'):
            self.baseline_delta_label.setText('vs baseline: ' + baseline.reason_text(delta.get('reason')))
            self.baseline_delta_label.setStyleSheet('font-size:11px; color:#9aadba;')
        else:
            text = baseline.one_line(delta)
            caution = []
            if delta.get('channels_changed'):
                caution.append('the baseline used a different channel set, so the numbers are not '
                               'directly comparable')
            if delta.get('device_changed'):
                caution.append('the baseline was recorded with a different device')
            if caution:
                text += ' · caution: ' + '; '.join(caution)
            self.baseline_delta_label.setText(text)
            self.baseline_delta_label.setStyleSheet(
                'font-size:11px; color:' + ('#f0c47b' if caution else '#7cddaf') + ';')
        for row, (name, base, current, change) in enumerate(baseline.band_table(delta)):
            self.baseline_table.item(row, 1).setText('—' if base is None else f'{base:.1f}%')
            self.baseline_table.item(row, 2).setText('—' if current is None else f'{current:.1f}%')
            item = self.baseline_table.item(row, 3)
            item.setText('—' if change is None else f'{change:+.1f}')
            item.setForeground(QColor('#9aadba' if change is None
                                      else ('#7cddaf' if change > 0 else
                                            ('#ffab83' if change < 0 else '#9aadba'))))
        if self.baseline is not None:
            note = baseline.interpretive_note(self.baseline)
            self.baseline_note.setText((note + ' ' if note else '') + baseline.CONDITION_NOTE)
        else:
            self.baseline_note.setText(baseline.BASELINE_NOTE + ' ' + baseline.CONDITION_NOTE)

    def set_baseline_from_current(self):
        """One action: store the current passing estimate as this subject's baseline."""
        if not self.subject or not self.subject.get('name'):
            self.set_subject()
            if not self.subject or not self.subject.get('name'):
                return
        condition = self.baseline_condition.currentText()
        report, source = self._current_estimate()
        if report is None:
            QMessageBox.warning(
                self, 'No passing estimate',
                'A baseline is a reference profile, so it must come from a passing estimate: wait '
                'until the stable multi-epoch estimate (or at least a valid five-second window) is '
                'available, then press "Set as baseline" again.')
            return
        try:
            record = baseline.build_baseline(
                report, condition,
                subject_name=self.subject['name'], subject_phone=self.subject.get('phone', ''),
                device=self.device_metadata,
                channel_source=self.device_metadata.get('channel_source'),
                channels=list(self.channels), fs_hz=self.fs, source=source)
            path = baseline.save_baseline(SUBJECT_ROOT, self.subject['name'],
                                          self.subject.get('phone', ''), record)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Baseline not saved', str(exc))
            return
        self.baseline = record
        self.baseline_error = ''
        self.update_baseline_panel()
        label = baseline.SOURCE_LABELS.get(source, source)
        message = (f'Baseline stored locally: {path}\nCondition: {condition}\nEstimate used: {label}\n\n'
                   'It replaces any earlier baseline of this subject and is compared from now on '
                   'whenever a later estimate is shown or saved. Descriptive EEG pattern reference '
                   'only — not a diagnosis and not an assessment of the person.')
        if source != 'stable':
            message += ('\n\nNo stable multi-epoch estimate was available, so the weaker five-second '
                        'window was used; this is recorded in baseline.json.')
        QMessageBox.information(self, 'Baseline captured', message)
        self.statusBar().showMessage(f'Baseline captured for {self.subject["name"]} '
                                     f'({condition}, {label}): {path}')

    def open_second_device(self):
        """A second, independent live window for the other headset at once."""
        import second_device
        dialog = self.second_window
        if dialog is not None:
            try:
                if not sip.isdeleted(dialog):
                    dialog.show(); dialog.raise_(); dialog.activateWindow()
                    return
            except RuntimeError:
                pass
        self.second_window = second_device.SecondDeviceWindow(parent=self,
                                                              simulated=self.simulated)
        self.second_window.show()

    def export_patient_report(self):
        """Bundle subject sessions + charts + AI text into one HTML (print to PDF)."""
        if not self.subject or not self.subject.get('name'):
            self.set_subject()
        if not self.subject or not self.subject.get('name'):
            return
        name = self.subject['name']; phone = self.subject.get('phone', '')
        default_dir = storage.subject_dir(SUBJECT_ROOT, name, phone)
        default = str(default_dir / f'patient_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        path, _ = QFileDialog.getSaveFileName(self, 'Save patient report (HTML) — download/copy',
                                              default, 'HTML report (*.html)')
        if not path:
            return
        try:
            written = patient_report.build(SUBJECT_ROOT, name, phone, out_path=Path(path))
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Patient report', str(exc))
            return
        self.statusBar().showMessage(f'Patient report written locally: {written}')
        QMessageBox.information(
            self, 'Patient report',
            f'Downloadable report written locally:\n{written}\n\n'
            'One self-contained HTML file: every saved session, the charts, the history table, '
            'the stored AI doctor text and the interpretation helper. Open it in any browser '
            'and print to PDF to share.')

    def save_session_clicked(self):
        if len(self.samples) < self.fs * 5:
            QMessageBox.warning(self, 'Nothing to save',
                                f'A session needs at least five seconds of continuous EEG '
                                f'({self.fs * 5} samples).')
            return
        if not self.subject or not self.subject.get('name'):
            self.set_subject()
            if not self.subject or not self.subject.get('name'):
                return
        self.analyze_now(); self.analyze_stable_now()
        rows = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        charts = self._session_charts()
        summary = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'device_type': self.device_metadata.get('device_key', 'unknown'),
            'device_label': self.device_metadata.get('device_label', 'unknown'),
            'device': self.device_metadata,
            'simulated': bool(self.simulated),
            'family': self.device_metadata.get('family'),
            'channels': list(self.channels),
            'scalp_channels': list(self.device_metadata.get('scalp_channels') or
                                   devices.scalp_channels(self.channels)),
            'poly_channels': list(self.device_metadata.get('poly_channels') or
                                  devices.poly_channels(self.channels)),
            'channel_source': self.device_metadata.get('channel_source', 'unknown'),
            'fs_hz': self.fs,
            'raw_rows': int(rows.shape[0]),
            'duration_s': round(rows.shape[0] / self.fs, 3),
            'raw_span_note': f'Raw CSV holds the rolling {BUFFER_SECONDS} s analysis buffer, not the '
                             'whole session length.',
            'notch_hz': self._notch(),
            'analysis': self.last_report,
            'stable_estimate': self.last_stable,
            'contact': self.contact,
            'transport': self.transport_metrics,
            'ai_doctor': self.ai_note,
            'methodology': 'METHODS.md',
            'limits': 'Engineering signal-quality estimates only: not a diagnosis, not a clinical '
                      'measurement, no mental-state, emotion or sleep-stage inference.',
        }
        estimate, estimate_source = self._current_estimate()
        vs_baseline = baseline.vs_baseline_for_summary(
            self.baseline, estimate, source=estimate_source or 'stable', device=self.device_metadata)
        if vs_baseline is not None:
            # Only written when this subject had a baseline at save time; the
            # baseline's condition is inside the record because it changes how
            # the deltas may be read.
            summary['vs_baseline'] = vs_baseline
        header = (['sample_index', 'relative_time_s', 'packet_number']
                  + [f'raw_{name}_uV' for name in self.channels])
        data_rows = [[index, round(index / self.fs, 6), number, *values]
                     for index, (number, values) in enumerate(
                         zip(list(self.packet_numbers), rows.tolist()))]
        try:
            folder = storage.save_session(SUBJECT_ROOT, self.subject['name'],
                                          self.subject.get('phone', ''), summary, header,
                                          data_rows, charts=charts)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Save failed', str(exc))
            return
        self.session_count += 1
        self.statusBar().showMessage(f'Session saved locally: {folder}')
        self.refresh_history_summary()
        self.refresh_trend()

    def _session_charts(self):
        """PNG bytes per chart; failures are reported instead of hidden."""
        charts, skipped = {}, []
        wanted = {
            'traces_raw_channels.png': lambda: grab_png(self.channel_grid_holder),
            'band_power_bars.png': lambda: grab_png(self.bars_plot),
            'psd_spectrum.png': lambda: grab_png(self.psd_plot),
            'head_map_current.png': lambda: grab_png(self.map_plot),
        }
        for name, render in wanted.items():
            try:
                charts[name] = render()
            except Exception as exc:
                skipped.append(f'{name}: {exc}')
        for band in BANDS:
            try:
                charts[f'head_map_{band.lower()}.png'] = self._render_topomap_png(band)
            except Exception as exc:
                skipped.append(f'head_map_{band.lower()}.png: {exc}')
        if skipped:
            self.statusBar().showMessage('Chart rendering issues: ' + '; '.join(skipped))
        return charts

    def _render_topomap_png(self, band, grid_n=288, width=1400, height=1280):
        """Render one band's head map off-screen at presentation quality; PNG bytes."""
        field = self._topomap_field(band, grid_n=grid_n)
        if field is None:
            raise RuntimeError(f'no usable electrodes for the {band} map')
        plot = pg.PlotWidget()
        plot.setBackground('#0e1c26')
        plot.setAspectLocked(True)
        plot.setRange(xRange=(-1.32, 1.32), yRange=(-1.32, 1.32), padding=0)
        plot.getPlotItem().setTitle(
            f'NeuroSync · {band} relative power (%) · schematic head map',
            size='15pt', color='#e3edf4')
        outline_pen = pg.mkPen('#7fa3b8', width=1.6)
        image = pg.ImageItem()
        image.setLookupTable(self.map_cmap.getLookupTable(nPts=256))
        plot.addItem(image)
        x, y = topomap.head_outline(361)
        plot.plot(x, y, pen=outline_pen)
        plot.plot([-.22, 0, .22], [.97, 1.14, .97], pen=outline_pen)
        plot.plot([-1, -1.12, -1.12, -1], [-.22, -.12, .12, .22], pen=outline_pen)
        plot.plot([1, 1.12, 1.12, 1], [-.22, -.12, .12, .22], pen=outline_pen)
        scalp = devices.scalp_channels(self.channels)
        positions = [topomap.ELECTRODE_POSITIONS[name] for name in scalp]
        plot.addItem(pg.ScatterPlotItem(
            pos=np.asarray(positions), size=9, brush=pg.mkBrush('#f2f7fb'),
            pen=pg.mkPen('#12222c', width=1.2)))
        for (ex, ey), name in zip(positions, scalp):   # electrode labels, pushed outward
            lx = max(-1.28, min(1.28, ex * 1.16))
            ly = max(-1.28, min(1.28, ey * 1.16))
            label = pg.TextItem(
                html=f'<span style="color:#cfe3ee; font-size:10pt">{name}</span>',
                anchor=(0.5, 0.5))
            label.setPos(lx, ly)
            plot.addItem(label)
        # rows flipped for nose-up display, same convention as the live map
        image.setImage(field[::-1, :], autoLevels=True, axisOrder='row-major')
        image.setRect(QRectF(-1, -1, 2, 2))
        try:
            bar = pg.ColorBarItem(colorMap=self.map_cmap, interactive=False, width=18,
                                  label='relative power (%)')
            bar.setImageItem(image, insert_in=plot.getPlotItem())
        except Exception:
            pass
        note = pg.TextItem(
            html='<span style="color:#7f97a8; font-size:9pt">'
                 'Schematic positions · engineering interpolation only — '
                 'not source localisation</span>',
            anchor=(0, 1))
        note.setPos(-1.3, -1.26)
        plot.addItem(note)
        plot.resize(width, height)
        png = grab_png(plot)
        plot.setParent(None)
        plot.deleteLater()
        return png

    def refresh_history_summary(self):
        try:
            subjects = storage.list_subjects(SUBJECT_ROOT)
        except OSError as exc:
            self.history_label.setText(f'Could not read the local store: {exc}')
            return
        sessions = sum(item.get('sessions', 0) for item in subjects)
        if not subjects:
            self.history_label.setText(
                'No saved sessions yet. Set a subject, record, then press "Save session". '
                f'Sessions are written to {SUBJECT_ROOT}')
            return
        latest = subjects[0]
        self.history_label.setText(
            f'{len(subjects)} subject(s) · {sessions} saved session(s) under {SUBJECT_ROOT}. '
            f"Most recent: {latest.get('name') or latest['key']} "
            f"({(latest.get('last_session_at') or '')[:19].replace('T', ' ')}). "
            'Press "History…" to reopen a saved summary and its charts.')

    def show_history(self):
        dialog = HistoryDialog(SUBJECT_ROOT, self)
        dialog.exec()
        self.refresh_history_summary()
        self.refresh_trend()
        self.load_baseline()

    def _trend_plot_ready(self) -> bool:
        """True while the trend plot and its ViewBox can still take items.

        Qt deletes the C++ object behind a widget wrapper when the widget (or
        its window) goes away; plotting into it afterwards raises RuntimeError.
        """
        if not widget_alive(self.trend_plot):
            return False
        try:
            viewbox = self.trend_plot.plotItem.vb
        except (RuntimeError, AttributeError):
            return False
        return widget_alive(viewbox)

    def _set_trend_status(self, text):
        """Trend status line, or nothing when its label is already gone."""
        if widget_alive(self.trend_status):
            self.trend_status.setText(text)

    def refresh_trend(self):
        """Redraw the cross-session trend; skips drawing when its widgets are gone.

        The data gathering is pure storage work; every touch of the trend plot
        or its labels is guarded against Qt having already deleted the widget,
        so a save/session change after a plot was torn down is skipped instead
        of raising RuntimeError.
        """
        redraw = self._trend_plot_ready()
        if redraw:
            for curve in self.trend_curves.values():
                try:
                    self.trend_plot.removeItem(curve)
                except RuntimeError:
                    pass
        self.trend_curves = {}
        if not self.subject or not self.subject.get('name'):
            self._set_trend_status('Set a subject and save sessions to see a trend.')
            return
        sessions = storage.list_sessions(SUBJECT_ROOT, self.subject['name'],
                                         self.subject.get('phone', ''))
        series = {name: [] for name in BANDS}
        used, source = 0, None
        for item in reversed(sessions):
            try:
                summary = storage.load_session(item['path'])['summary']
            except (OSError, ValueError):
                continue
            stable = summary.get('stable_estimate') or {}
            report = summary.get('analysis') or {}
            bands = (stable.get('bands') if stable.get('valid') else None) or report.get('bands') or {}
            row = {}
            for name in BANDS:
                value = (bands.get(name) or {}).get('relative_pct')
                row[name] = value
            if not all(isinstance(row[name], (int, float)) for name in BANDS):
                continue
            for name in BANDS:
                series[name].append(float(row[name]))
            used += 1
            source = 'stable multi-epoch estimate' if stable.get('valid') else '5 s window'
        if not used:
            self._set_trend_status(f'No saved session with usable band numbers yet for '
                                   f'{self.subject["name"]}. Save a session first.')
            return
        if not redraw:
            return
        x = list(range(1, used + 1))
        try:
            for name in BANDS:
                curve = self.trend_plot.plot(x, series[name], pen=pg.mkPen(COLORS[name], width=2),
                                             symbol='o', symbolSize=6, name=name)
                self.trend_curves[name] = curve
        except RuntimeError:
            # The plot was deleted between the guard above and drawing.
            self.trend_curves = {}
            return
        self._set_trend_status(
            f'{used} saved session(s) for {self.subject["name"]} · values are % of 1–45 Hz, source: '
            f'{source} where available. Session order is by stored timestamp.')

    # ------------------------------------------------------------- analysis UI
    def analyze_now(self):
        n = len(self.samples)
        needed = self.fs * 5
        self.progress_label.setText(f'Window: {min(n, needed)} / {needed} samples · rolling 5.0 s')
        age = None if self.last_received is None else time.monotonic() - self.last_received
        contact_age = None if self.contact is None else time.monotonic() - self.contact['measured_monotonic']
        self.contact_title.setText('ELECTRODE CONTACT · ' + (
            'not checked' if contact_age is None
            else f'{contact_age:.0f}s ago' + (' — recheck' if contact_age > 120 else '')))
        continuous = age is not None and age <= .5
        try:
            samples = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        except ValueError:
            return
        report = analyze_window(samples, fs=self.fs, notch_hz=self._notch(),
                                contact_ohms=self.contact['ohms'] if self.contact else None,
                                contact_age_s=contact_age, continuous=continuous,
                                channels=self.channels)
        self.last_report = report
        self.conclusion.setText(report['conclusion'])
        self.conclusion.setStyleSheet('font-size:17px; padding:12px; background:#20323e; color:'
                                      + ('#8be6ce' if report['valid'] else '#f0c47b'))
        self.quality_label.setText(report['reason'])
        for row, name in enumerate(BANDS):
            band = report['bands'].get(name, {})
            power = band.get('power_uv2'); relative = band.get('relative_pct')
            self.power_table.item(row, 1).setText(f'{power:.2f}' if power is not None else '—')
            self.power_table.item(row, 2).setText(f'{relative:.1f}%' if relative is not None else '—')
        self._fill_channel_table(report)
        self._update_bars(report)
        self._update_psd()
        if isinstance(report, dict) and report.get('dominant'):
            self._set_predominant(report, '5 s window')
        self.update_topomap()
        self.update_buttons()
        self.update_baseline_deltas()
        if self.runtime_enabled:
            state = {'updated_at': datetime.now(timezone.utc).isoformat(),
                     'transport': self.transport_metrics, 'device': self.device_metadata,
                     'buffer_samples': n, 'plot_updates': self.plot_updates,
                     'last_sample_age_s': age, 'contact': self.contact, 'report': report,
                     'stable_estimate': self.last_stable, 'ai_doctor': self.ai_note,
                     'simulated': bool(self.simulated)}
            try:
                pending = ROOT / 'runtime_status.pending'
                pending.write_text(json.dumps(state, indent=2, allow_nan=False), encoding='utf-8')
                pending.replace(ROOT / 'runtime_status.json')
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(f'Local status write failed: {exc}')

    def _set_predominant(self, estimate, source):
        """One-line 'most predominant brain wave' read-out (descriptive only)."""
        if not isinstance(estimate, dict) or not estimate.get('dominant'):
            self.predominant.setText('PREDOMINANT RHYTHM · — (waiting for a valid window)')
            self.predominant.setStyleSheet('font-size:15px; padding:9px; background:#17262f; '
                                           'color:#e3edf4;')
            return
        text = f"PREDOMINANT RHYTHM · {str(estimate.get('dominant')).upper()}"
        share = estimate.get('dominant_pct')
        if share is not None:
            text += f" · {float(share):.1f}% of 1–45 Hz"
        peak = estimate.get('peak_hz')
        if peak is not None:
            text += f" · peak {float(peak):.1f} Hz"
        self.predominant.setText(text + f"  ({source})")
        self.predominant.setStyleSheet('font-size:15px; padding:9px; background:#17262f; '
                                       'color:#8be6ce;')

    def _fill_channel_table(self, report):
        channels = list(report.get('channels', {}))
        self.channel_table.setRowCount(len(channels))
        for row, name in enumerate(channels):
            entry = report['channels'][name]
            relative = entry.get('relative_pct') or {}
            powers = entry.get('powers_uv2') or {}
            top_band = max(relative, key=relative.get) if relative else None
            power_text = (f"{powers[top_band]:.2f} ({top_band})" if top_band else '—')
            dominant = entry.get('dominant') or ('mixed' if relative else 'unreliable')
            cells = [name, power_text,
                     f"{relative[top_band]:.1f}%" if top_band else '—',
                     dominant if entry.get('valid') else f"{dominant} · flagged"]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(str(text))
                if column == 0:
                    item.setForeground(QColor('#cde6f2'))
                if not entry.get('valid') and column == 3:
                    item.setForeground(QColor('#ffab83'))
                self.channel_table.setItem(row, column, item)

    def _update_bars(self, report):
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
        self.bars_plot.setTitle(
            'absolute µV² (bars) · relative % scaled to the same height '
            f'(max {scale:.1f}%)' if report.get('valid') else
            'band power unavailable until the quality gates pass')

    def _update_psd(self):
        if len(self.samples) < 500:
            return
        try:
            samples = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        except ValueError:
            return
        frequencies, density = mean_psd_curve(samples, fs=self.fs, notch_hz=self._notch(),
                                              channels=self.channels, fmax=min(100., self.fs / 2))
        if frequencies:
            self.psd_curve.setData(frequencies, density)
        else:
            self.psd_curve.clear()

    def _topomap_field(self, band, grid_n=64):
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
        field = self._topomap_field(band, grid_n=160)     # denser raster: smoother display
        if field is None:
            self.map_image.clear()
            self.map_status.setText(
                f'No {band} head map yet: needs a valid five-second window and at least three '
                f'usable electrodes (currently {len(scalp)} scalp channel(s) known).')
            return
        # topomap contract: row 0 is y=+radius (nose); pyqtgraph row-major draws
        # row 0 at the view bottom, so flip rows for a nose-up display.
        self.map_image.setImage(field[::-1, :], autoLevels=True, axisOrder='row-major')
        self.map_image.setRect(QRectF(-1, -1, 2, 2))
        if self.map_bar is not None:                      # keep the scale bar in range
            levels = self.map_image.getLevels()
            if levels is not None:
                self.map_bar.setLevels(tuple(levels), update_items=False)
        finite = field[np.isfinite(field)]
        self.map_status.setText(
            f'{band}: relative power on {len(scalp)} schematic electrode positions, '
            f'interpolated (scipy) to a {field.shape[0]}x{field.shape[1]} grid; '
            f'colour range {finite.min():.1f}–{finite.max():.1f}%. '
            'Approximate layout, not source localisation.')

    def export_head_map(self):
        band = self.map_band_select.currentText()
        try:
            png = self._render_topomap_png(band)
        except Exception as exc:
            QMessageBox.warning(self, 'Head map export', f'No map to export: {exc}')
            return
        default = str(ROOT / f'head_map_{band.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        path, _ = QFileDialog.getSaveFileName(self, 'Save head map PNG', default, 'PNG image (*.png)')
        if not path:
            return
        try:
            Path(path).write_bytes(png)
        except OSError as exc:
            QMessageBox.warning(self, 'Head map export', str(exc))
            return
        self.statusBar().showMessage(f'Head map PNG written locally: {path}')

    def analyze_stable_now(self):
        n = len(self.samples)
        needed = STABLE_MIN_SAMPLES if self.fs == 250 else int(self.fs * 12)
        if n < needed:
            self.stable_status.setText(
                f'Collecting: {n} / {needed} samples (six 2 s epochs minimum; window builds to 60 s).')
            return
        age = None if self.last_received is None else time.monotonic() - self.last_received
        contact_age = None if self.contact is None else time.monotonic() - self.contact['measured_monotonic']
        try:
            samples = np.asarray(self.samples, dtype=float).reshape((-1, len(self.channels)))
        except ValueError:
            return
        stable = analyze_stable(samples, fs=self.fs, notch_hz=self._notch(),
                               contact_ohms=self.contact['ohms'] if self.contact else None,
                               contact_age_s=contact_age,
                               continuous=age is not None and age <= .5,
                               channels=self.channels)
        self.last_stable = stable
        for row, name in enumerate(BANDS):
            band = stable['bands'].get(name, {})
            power = band.get('power_uv2'); sem = band.get('sem_uv2'); relative = band.get('relative_pct')
            self.stable_table.item(row, 1).setText(
                f'{power:.2f} ±{sem:.2f}' if power is not None and sem is not None else '—')
            self.stable_table.item(row, 2).setText(f'{relative:.1f}%' if relative is not None else '—')
        if stable['valid']:
            self.stable_status.setText(
                f"Clean epochs: {stable['epochs_used']} / {stable['epochs_total']} · "
                f"{stable['seconds_used']:.0f} s of {stable['seconds_total']:.0f} s used · "
                f"largest band {stable['dominant']} ({stable['dominant_pct']:.1f}%)")
            self._set_predominant(stable, 'stable estimate')
        else:
            self.stable_status.setText('No stable estimate: ' + (stable['reason'] or 'insufficient clean data')[:300])
        self.maybe_auto_ai()
        self.maybe_chat_update()
        self.update_baseline_deltas()

    def maybe_auto_ai(self):
        if not self.ai_auto.isChecked(): return
        if self.ai_worker is not None and self.ai_worker.isRunning(): return
        if self.last_report is None and self.last_stable is None: return
        if self.ai_requested_at is not None and time.monotonic() - self.ai_requested_at < AI_AUTO_INTERVAL_S:
            return
        self.run_ai_doctor()

    def run_ai_doctor(self):
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return
        if self.last_report is None and self.last_stable is None:
            self.ai_status.setText('No analysis yet: wait for at least five seconds of valid EEG.')
            return
        payload = ai_doctor.build_payload(report=self.last_report, stable=self.last_stable,
                                          transport=self.transport_metrics,
                                          device=self.device_metadata,
                                          contact=self.contact['ohms'] if self.contact else None)
        self.ai_requested_at = time.monotonic()
        self.ai_button.setEnabled(False)
        self.ai_status.setText(f'Asking {ai_doctor.DEFAULT_MODEL} on the local endpoint… '
                               '(this can take up to a minute)')
        self.ai_worker = AIDoctorWorker(payload, ai_doctor.DEFAULT_MODEL, self)
        self.ai_worker.note_ready.connect(self.handle_ai_note)
        self.ai_worker.start()

    def handle_ai_note(self, result):
        self.ai_button.setEnabled(True)
        note = {'ok': bool(result.get('ok')), 'model': result.get('model'),
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'text': result.get('text') or '', 'error': result.get('error') or ''}
        self.ai_note = note
        if note['ok']:
            self.ai_text.setText(note['text'] + '\n\n' + ai_doctor.DISCLAIMER)
            self.ai_status.setText(f"Answered by {note['model']} (local) at "
                                   f"{note['generated_at'][11:19]} UTC · assistive description, not a diagnosis")
        else:
            self.ai_text.setText('AI doctor unavailable: ' + note['error']
                                 + '\n\nNo substitute analysis is shown.')
            self.ai_status.setText('Local model unavailable — start Ollama and ask again.')
        self.update_buttons()

    # -------------------------------------------------------------- live chat
    def _chat_interval_changed(self, _index):
        self.chat_interval_s = float(self.chat_every.currentData() or CHAT_DEFAULT_INTERVAL_S)

    def _set_chat_status(self, text):
        if widget_alive(self.chat_status):
            self.chat_status.setText(text)

    def _show_chat(self):
        """Scroll the analysis column so the live chat is in view."""
        scroller = getattr(self, 'right_scroller', None)
        if scroller is not None and widget_alive(scroller):
            QTimer.singleShot(60, lambda: scroller.ensureWidgetVisible(self.chat_log))

    def toggle_chat_monitor(self):
        self.chat_enabled = not self.chat_enabled
        if self.chat_enabled:
            self.chat_toggle.setText('⏹ Stop live monitor')
            self._set_chat_status('On — preloading the local model (the first reply can take '
                                  'a minute on a cold model)…')
            threading.Thread(target=lambda: ai_doctor.warm(ai_doctor.DEFAULT_MODEL),
                             daemon=True).start()
            self._show_chat()
            self.maybe_chat_update()
        else:
            self.chat_toggle.setText('▶ Start live monitor')
            self._set_chat_status('Stopped — the current reply finishes; no new updates.')

    def clear_chat(self):
        self.chat_history = []
        self.chat_updates = 0
        if widget_alive(self.chat_log):
            self.chat_log.clear()
        self._set_chat_status('Cleared.' + (' Monitor stays on.' if self.chat_enabled else ''))

    def _chat_append(self, role, text, meta=''):
        """Add one chat block; returns the cursor used for streamed text appends."""
        if not widget_alive(self.chat_log):
            return None
        palette = {'user': ('#9fd3ff', 'YOU'), 'assistant': ('#8be6ce', 'LOCAL MODEL'),
                   'sys': ('#f0c47b', '—')}
        colour, who = palette.get(role, palette['sys'])
        label = who + (f' · {meta}' if meta else '')
        cursor = self.chat_log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertHtml(
            f'<p style="margin:2px 0 0; color:{colour}; font-size:8pt;">{esc_html(label)}</p>')
        body = esc_html(text or '').replace('\n', '<br>')
        cursor.insertHtml(f'<p style="margin:0 0 8px; color:#dbe6ee;">{body}</p>')
        self.chat_log.ensureCursorVisible()
        return cursor

    def _chat_chunk(self, text):
        if not widget_alive(self.chat_log) or self._chat_cursor is None:
            return
        self._chat_cursor.insertText(text)
        self.chat_log.ensureCursorVisible()

    def _chat_done(self, result):
        self._chat_cursor = None
        if bool(result.get('ok')):
            text = str(result.get('text') or '')
            self.chat_history.append({'role': 'assistant', 'content': text[:900]})
            if len(self.chat_history) > CHAT_HISTORY_MAX * 2:
                self.chat_history = self.chat_history[-CHAT_HISTORY_MAX * 2:]
            started = getattr(self, '_chat_started', None)
            took = f'{(time.monotonic() - started):.1f} s · ' if started else ''
            self._set_chat_status(
                f'Last reply {time.strftime("%H:%M:%S")} ({took}{result.get("model")}, local) '
                '· assistive description, not a diagnosis.')
        else:
            error = str(result.get('error') or 'unknown error')
            self._chat_append('sys', 'Local model error: ' + error
                              + '\nNo substitute analysis is shown.')
            self._set_chat_status('Local model error — is Ollama running at 127.0.0.1:11434?')
        self.update_buttons()

    def _chat_request(self, question=None):
        if self.chat_worker is not None and self.chat_worker.isRunning():
            self._set_chat_status('The local model is busy with the previous reply.')
            return
        if self.last_report is None and self.last_stable is None:
            self._set_chat_status('No analysis yet — wait for at least five seconds of valid EEG.')
            return
        if self.ai_worker is not None and self.ai_worker.isRunning():
            self._set_chat_status('The AI doctor is using the model — the update follows after.')
            return
        payload = ai_doctor.build_payload(report=self.last_report, stable=self.last_stable,
                                          transport=self.transport_metrics,
                                          device=self.device_metadata,
                                          contact=self.contact['ohms'] if self.contact else None)
        self.chat_updates += 1
        messages = ai_doctor.build_live_messages(payload, history=self.chat_history,
                                                 question=question,
                                                 update_number=self.chat_updates)
        if question:
            self._show_chat()
            self._chat_append('user', question)
            self.chat_history.append({'role': 'user', 'content': question[:900]})
            meta = 'answer'
        else:
            meta = f'live update #{self.chat_updates}'
        self._chat_cursor = self._chat_append('assistant', '', meta)
        self.chat_last_at = time.monotonic()
        self._chat_started = self.chat_last_at
        self._set_chat_status('Streaming ' + meta + '…')
        self.chat_worker = LiveChatWorker(messages, ai_doctor.DEFAULT_MODEL, self)
        self.chat_worker.chunk_ready.connect(self._chat_chunk)
        self.chat_worker.finished_run.connect(self._chat_done)
        self.chat_worker.start()
        self.update_buttons()

    def maybe_chat_update(self):
        if not self.chat_enabled:
            return
        if self.chat_worker is not None and self.chat_worker.isRunning():
            return
        fresh = (self.last_received is not None
                 and time.monotonic() - self.last_received <= 2.0)
        if not fresh:
            self._set_chat_status('On — waiting for live samples…')
            return
        if (self.chat_last_at is not None
                and time.monotonic() - self.chat_last_at < self.chat_interval_s):
            return
        self._chat_request(question=None)

    def send_chat_question(self):
        question = self.chat_input.text().strip()
        if not question:
            return
        self.chat_input.clear()
        self._chat_request(question=question)

    # -------------------------------------------------------------- display IO
    def handle_contact(self, contact):
        self.contact = contact
        for name, label in self.contact_labels.items():
            value = (contact.get('ohms') or {}).get(name)
            good = value is not None and 0 < value <= 1e6
            display = f'{value / 1000:,.0f} kΩ' if value is not None else 'open / unavailable'
            label.setText(f'{name}: {display} · ' + ('within ≤1 MΩ' if good else 'CHECK CONTACT'))
            label.setStyleSheet('color:' + ('#7cddaf' if good else '#ffab83'))
        aux = contact.get('aux') or {}
        if aux:
            text = ' · '.join(
                f'{key}: ' + (f'{value / 1000:,.0f} kΩ' if value is not None else 'unavailable')
                for key, value in aux.items())
            self.contact_aux_label.setText('Reference contacts — ' + text
                                           + ' (A1/A2/Bias, from the device resistance packet)')
        else:
            self.contact_aux_label.setText('A1/A2/Bias: not reported by this device.')

    def reset_window(self, reason):
        self.samples.clear(); self.packet_numbers.clear()
        self.last_received = None; self.last_report = None; self.last_stable = None
        for curve in list(self.raw_curves.values()) + list(self.band_curves.values()):
            curve.clear()
        self.conclusion.setText(reason)
        self.conclusion.setStyleSheet('font-size:17px; padding:12px; background:#20323e; color:#e3edf4;')
        self.quality_label.setText('No conclusion until a new clean, continuous five-second window is available.')
        for row in range(5):
            self.power_table.item(row, 1).setText('—'); self.power_table.item(row, 2).setText('—')
            self.stable_table.item(row, 1).setText('—'); self.stable_table.item(row, 2).setText('—')
        self.channel_table.setRowCount(0)
        self.progress_label.setText(f'Window: 0 / {self.fs * 5} samples')
        self.stable_status.setText(f'Collecting: 0 / {STABLE_MIN_SAMPLES if self.fs == 250 else int(self.fs * 12)} '
                                   'samples (six 2 s epochs minimum; window builds to 60 s).')
        self.bars_absolute.setOpts(x=[], height=[])
        self.bars_relative.setOpts(x=[], height=[])
        self.psd_curve.clear()
        self.avg_curve.clear()
        self.predominant.setText('PREDOMINANT RHYTHM · — (no current estimate)')
        self.predominant.setStyleSheet('font-size:15px; padding:9px; background:#17262f; '
                                       'color:#e3edf4;')
        self.map_image.clear()
        self.map_status.setText('Waiting for a valid five-second window.')
        self.update_buttons()
        self.update_baseline_deltas()

    def _notch(self):
        return {'60 Hz': 60, '50 Hz': 50, 'Off': None}[self.notch_select.currentText()]

    def update_plots(self):
        self.update_buttons()
        if not self.samples:
            return
        data = np.asarray(self.samples, dtype=float)
        if data.ndim != 2 or data.shape[1] != len(self.channels):
            return
        shown = data[-self.fs * 5:]
        x = (np.arange(len(shown)) - len(shown) + 1) / self.fs
        for index, channel in enumerate(self.channels):
            curve = self.raw_curves.get(channel)
            if curve is not None:
                curve.setData(x, shown[:, index])
        index = min(self.channel_select.currentIndex(), len(self.channels) - 1)
        channel_name = self.channels[max(index, 0)]
        self.band_heading.setText(f'FREQUENCY COMPONENTS · {channel_name} · display only · µV')
        if len(data) >= self.fs * 5:
            filtered = band_traces(data, fs=self.fs, channel=max(index, 0),
                                   notch_hz=self._notch(), channels=self.channels)
            for name in BANDS:
                y = filtered.get(name, [])[-len(x):]
                if len(y):
                    self.band_curves[name].setData(x[-len(y):], y)
                else:
                    self.band_curves[name].clear()
            # average brain wave: channel mean, rebuilt from the 1–45 Hz components
            try:
                mean_signal = data[-self.fs * 5:].mean(axis=1).reshape(-1, 1)
                components = band_traces(mean_signal, fs=self.fs, channel=0,
                                         notch_hz=self._notch(), channels=('mean',))
                sized = np.zeros(len(x))
                for name in BANDS:
                    y = np.asarray(components.get(name, []), dtype=float)
                    if y.size:
                        count = min(len(sized), y.size)
                        sized[-count:] += y[-count:]
                self.avg_curve.setData(x, sized)
            except Exception:
                pass
        self.plot_updates += 1
        age = None if self.last_received is None else time.monotonic() - self.last_received
        if age is None or age > .5:
            self.conclusion.setText('Data is stale / paused — no current conclusion.')

    def export_snapshot(self, destination=None):
        """Quick five-second raw CSV + report (kept from the verified workflow)."""
        needed = self.fs * 5
        if len(self.samples) < needed:
            raise ValueError('A full five-second window is required')
        self.analyze_now()
        folder = (Path(destination) if destination
                  else ROOT / 'recordings' / datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=False)
        data = list(self.samples)[-needed:]; numbers = list(self.packet_numbers)[-needed:]
        with (folder / 'raw.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['sample_index', 'relative_time_s', 'packet_number']
                            + [f'raw_{name}_uV' for name in self.channels])
            for index, (values, number) in enumerate(zip(data, numbers)):
                writer.writerow([index, index / self.fs, number, *values])
        meta = {'captured_at': datetime.now(timezone.utc).isoformat(),
                'device': self.device_metadata, 'fs_hz': self.fs, 'notch_hz': self._notch(),
                'channels': list(self.channels), 'report': self.last_report,
                'stable_estimate': self.last_stable, 'ai_doctor': self.ai_note,
                'contact': self.contact, 'transport': self.transport_metrics,
                'source': 'Real SDK samples; no synthetic data in application',
                'timing': 'Relative sample time reconstructed at the confirmed nominal rate, '
                          'not a hardware absolute timestamp',
                'methodology': 'METHODS.md',
                'warning': 'Descriptive research data, not a clinical or mental-state diagnosis'}
        (folder / 'report.json').write_text(json.dumps(meta, indent=2, allow_nan=False),
                                            encoding='utf-8')
        self.statusBar().showMessage(f'Saved five-second raw EEG and quality report: {folder}')
        return folder

    def save_clicked(self):
        try:
            self.export_snapshot()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, 'Export failed', str(exc))

    def closeEvent(self, event):
        self.plot_timer.stop(); self.analysis_timer.stop(); self.stable_timer.stop()
        if self.ai_worker is not None and self.ai_worker.isRunning() and not self.ai_worker.wait(5000):
            # Never abort a mid-request thread on window close; detach it instead.
            self.ai_worker.setParent(None); _ORPHAN_WORKERS.append(self.ai_worker)
        if self.chat_worker is not None and self.chat_worker.isRunning():
            self.chat_worker.stop_requested = True
            if not self.chat_worker.wait(2500):
                # Same policy as the AI doctor: never abort a live request; detach it.
                try:
                    self.chat_worker.chunk_ready.disconnect()
                    self.chat_worker.finished_run.disconnect()
                except TypeError:
                    pass
                self.chat_worker.setParent(None); _ORPHAN_WORKERS.append(self.chat_worker)
        if self.client is None or self.client.stop():
            event.accept()
        else:
            event.ignore()
            self.statusBar().showMessage('Waiting for SDK shutdown...')
            QTimer.singleShot(1000, self.close)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='NeuroSync live EEG research viewer')
    parser.add_argument('--smoke', action='store_true',
                        help='headless launch check: build the window, then quit after a few seconds')
    parser.add_argument('--smoke-seconds', type=float, default=4.,
                        help='how long the --smoke check keeps the window open')
    parser.add_argument('--no-autostart', action='store_true',
                        help='do not start a BLE scan/connection on launch')
    parser.add_argument('--simulate', choices=['brainbit', 'dragon'], default=None,
                        help='run a clearly-labelled SIMULATED stream instead of BLE '
                             '(pipeline rehearsal; no headset needed)')
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    app = QApplication([sys.argv[0]])
    window = NeuroSyncApp(autostart=not args.no_autostart, smoke=args.smoke,
                          simulated=args.simulate)
    window.show()
    if args.smoke:
        print(f'SMOKE: window constructed and shown; closing in {args.smoke_seconds:g}s', flush=True)
        QTimer.singleShot(int(args.smoke_seconds * 1000), app.quit)
    code = app.exec()
    if args.smoke:
        print(f'SMOKE OK: no exception; exit code {code}', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
