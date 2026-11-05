"""qEEG grid UI tests: offscreen Qt, synthetic EEG, no hardware.

The synthetic fixture is TEST ONLY. It is never produced by the live app, and
nothing here opens a device or a BLE socket.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
from PyQt6.QtWidgets import QApplication

import devices
import patient_report
import qeeg
import qeeg_ui
import storage

FS = 250.0
SECONDS = 8.0


def synthetic_session(channels=None):
    """Deterministic 21-channel EEG: alpha + theta + a little noise."""
    channels = list(channels or devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
    rng = np.random.default_rng(7)
    t = np.arange(int(FS * SECONDS)) / FS
    samples = np.empty((t.size, len(channels)), dtype=float)
    for index in range(len(channels)):
        samples[:, index] = (12.0 * np.sin(2 * np.pi * 10.0 * t + 0.2 * index)
                             + 5.0 * np.sin(2 * np.pi * 5.0 * t + 0.1 * index)
                             + rng.normal(0.0, 0.8, t.size))
    return samples, channels


def minimal_report():
    """A passing analysis dict just rich enough for the report template."""
    return {'valid': True, 'ready': True, 'reason': 'test fixture',
            'dominant': 'Alpha', 'dominant_pct': 45.0, 'peak_hz': 10.0,
            'samples': int(FS * SECONDS),
            'bands': {name: {'power_uv2': 5.0, 'relative_pct': 20.0}
                      for name in ('Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')},
            'channels': {}}


class QEEGPanelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.samples, self.channels = synthetic_session()
        self.panel = qeeg_ui.QEEGPanel(data_provider=self.provider)
        self.panel.resize(1250, 1200)
        self.panel.show()
        self.addCleanup(self.panel.close)

    def provider(self):
        return {'samples': self.samples, 'fs': FS, 'channels': self.channels,
                'session_key': 'synthetic-test'}

    # ------------------------------------------------------------- structure
    def test_panel_constructs_offscreen_with_25_cells(self):
        self.panel.refresh(force=True)
        self.assertEqual(len(self.panel.sheet.cells), 25)
        self.assertEqual(set(self.panel.sheet.cells),
                         {(row, band) for row in qeeg_ui.ROW_KEYS
                          for band in qeeg_ui.BAND_KEYS})
        self.assertIs(self.panel.stack.currentWidget(), self.panel.sheet_page)
        self.assertEqual(len(self.panel.band_checks), 5)
        self.assertEqual(len(self.panel.row_checks), 5)
        self.assertEqual([self.panel.reference_combo.itemText(index)
                          for index in range(self.panel.reference_combo.count())],
                         ['This session', 'Saved baseline', 'Load norm file…', 'Raw'])
        self.assertIn(self.panel.condition_combo.currentText(),
                      ('eyes closed', 'eyes open', 'other'))

    def test_cells_carry_raw_value_and_z_in_tooltip(self):
        self.panel.refresh(force=True)
        tooltip = self.panel.sheet.cells[('absolute', 'Alpha')].toolTip()
        self.assertIn('Absolute power', tooltip)
        self.assertIn('raw', tooltip)
        self.assertIn('z', tooltip)
        self.assertIn('Fp1', tooltip)          # 21-channel scalp set
        self.assertNotIn('D1', tooltip)

    # ------------------------------------------------------------- reference
    def test_reference_line_names_kind_label_n_and_disclaimer(self):
        self.panel.refresh(force=True)
        text = self.panel.reference_label.text()
        self.assertIn('kind: session', text)
        self.assertIn("self-referenced vs this session's channel mean", text)
        self.assertIn('n=21', text)
        self.assertIn('not a normative clinical database', text)
        self.assertNotIn('NeuroGuide', text)
        # The exported sheet carries the same line, not a shorter one.
        self.assertEqual(self.panel.sheet.sheet_reference_label.text(), text)
        metadata = self.panel.report_metadata()
        self.assertEqual(metadata['reference_kind'], 'session')
        self.assertEqual(metadata['reference_n'], 21)

    def test_raw_reference_says_raw_and_marks_no_z(self):
        self.panel.reference_combo.setCurrentIndex(3)      # Raw
        self.panel.refresh(force=True)
        text = self.panel.reference_label.text()
        self.assertIn('kind: none', text)
        self.assertIn('raw values', text)
        self.assertIn('not a normative clinical database', text)

    def test_reference_wording_never_upgrades_non_norm_kinds(self):
        for kind in ('session', 'baseline', 'none'):
            line = qeeg_ui.describe_reference({'kind': kind, 'label': 'x', 'n': 5})
            self.assertNotIn('NeuroGuide', line)
            without_disclaimer = line.replace('not a normative clinical database', '')
            self.assertNotIn('normative', without_disclaimer)
        norm = qeeg_ui.describe_reference({'kind': 'norm', 'label': '40-50',
                                           'n': 22, 'age_band': '40-50'})
        self.assertIn('vs local norm', norm)
        self.assertIn('n=22', norm)
        self.assertIn('not a normative clinical database', norm)

    # ----------------------------------------------------------------- export
    def test_export_writes_png_over_10kb(self):
        self.panel.refresh(force=True)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'qeeg_sheet.png'
            written = self.panel.export_sheet_png(target)
            self.assertEqual(written, target)
            self.assertTrue(target.is_file())
            data = target.read_bytes()
            self.assertGreater(len(data), 10_000)
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')

    def test_saved_baseline_unavailable_falls_back_to_raw(self):
        panel = qeeg_ui.QEEGPanel(data_provider=self.provider,
                                  baseline_provider=lambda: None)
        panel.resize(900, 800)
        self.addCleanup(panel.close)
        panel.reference_combo.setCurrentIndex(1)      # Saved baseline
        panel.refresh(force=True)
        self.assertIn('kind: none', panel.reference_label.text())
        self.assertIn('none is available', panel.status_label.text())

    def test_norm_selection_without_file_shows_raw_not_a_fake_reference(self):
        self.panel.reference_combo.setCurrentIndex(2)      # Load norm file…
        self.panel.refresh(force=True)
        self.assertIn('kind: none', self.panel.reference_label.text())
        self.assertIn('no local norm file', self.panel.status_label.text().lower())

    def test_loaded_norm_file_is_labelled_local_norm_with_n_and_age(self):
        metrics = qeeg.compute_metrics(self.samples, FS, self.channels)
        reference = qeeg.reference_from_baseline(metrics, '40-50', 22)
        reference.update(kind='norm', label='40-50', age_band='40-50')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'norm.json'
            qeeg.save_reference(reference, path)
            self.panel.norm_reference = qeeg.load_reference(path)
            self.panel.norm_path = str(path)
            self.panel.reference_combo.setCurrentIndex(2)  # Load norm file…
            self.panel.refresh(force=True)
        text = self.panel.reference_label.text()
        self.assertIn('kind: norm', text)
        self.assertIn('vs local norm n=22, age 40-50', text)
        self.assertIn('not a normative clinical database', text)
        self.assertNotIn('NeuroGuide', text)
        self.assertNotIn('self-referenced', text)

    # ------------------------------------------------------------------ cache
    def test_refresh_reuses_cache_and_only_force_recomputes(self):
        with mock.patch.object(qeeg, 'compute_metrics',
                               wraps=qeeg.compute_metrics) as counted:
            self.panel.refresh(force=True)
            first = counted.call_count
            self.panel.refresh()
            self.panel.refresh()
            self.assertEqual(counted.call_count, first)   # cache hit, no recompute
            self.panel.refresh(force=True)
            self.assertEqual(counted.call_count, first + 1)

    # ------------------------------------------------------------- empty state
    def test_fewer_than_three_channels_shows_empty_state(self):
        small_samples, small_channels = synthetic_session(['O1', 'O2'])
        panel = qeeg_ui.QEEGPanel(data_provider=lambda: {
            'samples': small_samples, 'fs': FS, 'channels': small_channels,
            'session_key': 'two'})
        panel.resize(900, 800)
        self.addCleanup(panel.close)
        panel.refresh(force=True)
        self.assertIs(panel.stack.currentWidget(), panel.empty_page)
        self.assertIn('at least 3 scalp channels', panel.empty_label.text())

    def test_no_recording_shows_empty_state(self):
        panel = qeeg_ui.QEEGPanel(data_provider=lambda: None)
        self.addCleanup(panel.close)
        panel.refresh(force=True)
        self.assertIs(panel.stack.currentWidget(), panel.empty_page)
        self.assertIn('No recording yet', panel.empty_label.text())

    # ----------------------------------------------------------- guide panel
    def test_guide_panel_is_collapsible_and_verbatim(self):
        self.assertFalse(self.panel.guide_panel.isVisible())
        self.panel.guide_toggle.setChecked(True)
        self.assertTrue(self.panel.guide_panel.isVisible())
        text = self.panel.guide_text.text()
        self.assertIn('Five bands = five "speeds". Delta (1–4 Hz) restorative sleep; '
                      'Theta (4–8) drowsy / shifting', text)
        self.assertIn('Read clusters, not single heads', text)
        self.assertIn('It is NOT an age-matched clinical normative database', text)


class QEEGPatientReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def test_report_contains_reference_labelled_section_and_disclaimer(self):
        samples, channels = synthetic_session()
        panel = qeeg_ui.QEEGPanel(data_provider=lambda: {
            'samples': samples, 'fs': FS, 'channels': channels,
            'session_key': 'report-test'})
        panel.resize(1250, 1200)
        self.addCleanup(panel.close)
        panel.refresh(force=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            root.mkdir()
            png = panel.sheet_png_bytes()
            summary = {
                'started_at': datetime.now(timezone.utc).isoformat(),
                'device_type': 'dragon_eeg',
                'device_label': 'DragonEEG / NeuroEEG',
                'device': {},
                'simulated': False,
                'channels': list(channels),
                'fs_hz': FS,
                'raw_rows': 30,
                'duration_s': SECONDS,
                'notch_hz': 60,
                'analysis': minimal_report(),
                'stable_estimate': None,
                'contact': None,
                'transport': {},
                'ai_doctor': None,
                'qeeg_grid': panel.report_metadata(),
            }
            header = (['sample_index', 'relative_time_s', 'packet_number']
                      + [f'raw_{name}_uV' for name in channels])
            rows = [[index, index / FS, index] + [0.0] * len(channels)
                    for index in range(30)]
            storage.save_session(root, 'Grid Test', '555', summary, header, rows,
                                 charts={'qeeg_grid.png': png})
            out = patient_report.build(root, 'Grid Test', '555',
                                       out_path=Path(tmp) / 'report.html')
            text = out.read_text(encoding='utf-8')

        self.assertIn('qEEG grid (reference:', text)
        self.assertIn('kind: session', text)
        self.assertIn('self-referenced vs this session', text)
        self.assertIn('Not a normative clinical database; educational display only.', text)
        self.assertIn('data:image/png;base64,', text)
        self.assertNotIn('NeuroGuide', text)


if __name__ == '__main__':
    unittest.main()
