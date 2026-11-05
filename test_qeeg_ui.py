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


def band_session(theta_amp=5.0, beta_amp=6.0, channels=None):
    """Deterministic session whose theta and beta power can be set independently."""
    channels = list(channels or devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
    rng = np.random.default_rng(11)
    t = np.arange(int(FS * SECONDS)) / FS
    samples = np.empty((t.size, len(channels)), dtype=float)
    for index, _name in enumerate(channels):
        samples[:, index] = (theta_amp * np.sin(2 * np.pi * 5.0 * t + 0.15 * index)
                             + beta_amp * np.sin(2 * np.pi * 20.0 * t + 0.3 * index)
                             + 6.0 * np.sin(2 * np.pi * 10.0 * t)
                             + rng.normal(0.0, 0.5, t.size))
    return samples, channels


def flat_norm_reference(channels, mean=10.0, sd=5.0):
    """A fixed mean/SD reference so EC/EO deltas are (EO − EC) / sd, by construction."""
    return {
        'schema': qeeg.REFERENCE_SCHEMA, 'kind': 'norm', 'label': 'teaching norm',
        'n': 22, 'age_band': '30-39', 'created_at': '2026-01-01T00:00:00+00:00',
        'bands': list(qeeg.BANDS), 'rows': list(qeeg.ROWS),
        'mean': {row: {band: {name: mean for name in channels} for band in qeeg.BANDS}
                 for row in qeeg.ROWS},
        'sd': {row: {band: {name: sd for name in channels} for band in qeeg.BANDS}
               for row in qeeg.ROWS},
    }


def write_norm_file(path, source='neuroguide', scope='21ch', n=22):
    """A validated NormFile on disk (source string is what the panel must print)."""
    import norms
    channels = list(devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
    rows = {row: {band: {name: {'mean': 20.0, 'sd': 4.0} for name in channels}
                  for band in qeeg.BANDS} for row in qeeg.ROWS}
    norm = norms.NormFile(source=source, label='teaching norm', n=n, age_band='30-39',
                          condition='eyes_closed', device_scope=scope, rows=rows,
                          provenance={'file': Path(path).name, 'sha256': 'a' * 64})
    return norms.save(norm, path)


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

    # ------------------------------------------------------- reading aids
    def test_theta_beta_card_shows_per_channel_ratios_mean_and_note(self):
        self.panel.refresh(force=True)
        card = self.panel.sheet.tbr_card
        self.assertTrue(card.isVisible())
        self.assertEqual(len(card.channel_labels), 21)
        self.assertIn('Fp1', card.channel_labels['Fp1'].text())
        self.assertRegex(card.channel_labels['Fp1'].text(), r'Fp1 \d+\.\d\d')
        self.assertIn('Channel mean θ/β', card.mean_label.text())
        self.assertIn('near 3:1', card.note.text())
        self.assertIn('not a diagnosis', card.note.text())
        self.assertIn('read it with the rest of the sheet', card.note.text())
        self.assertGreater(card.mean_value, 0.0)
        meta = self.panel.report_metadata()['theta_beta']
        self.assertEqual(len(meta['channels']), 21)
        self.assertEqual(meta['flag'], card.flagged)
        self.assertEqual(meta['note'], qeeg_ui.TBR_NOTE)

    def test_theta_beta_flag_needs_mean_above_three_and_both_bands(self):
        # Synthetic fixture has almost no beta power -> huge ratio, flagged.
        self.panel.refresh(force=True)
        self.assertTrue(self.panel.sheet.tbr_card.usable)
        self.assertGreaterEqual(self.panel.sheet.tbr_card.mean_value, 3.0)
        self.assertTrue(self.panel.sheet.tbr_card.flagged)
        self.assertTrue(self.panel.sheet.tbr_card.flag_label.isVisible())
        self.assertIn('not a diagnosis', self.panel.sheet.tbr_card.flag_label.text())
        # A metrics dict without a Beta band can never flag, however large theta is.
        ratios, mean, usable = qeeg_ui.theta_beta_values({
            'usable_channels': ['Fp1', 'Fp2'], 'bands': {'Theta': {'absolute': {'Fp1': 99.0,
                                                                              'Fp2': 99.0}}}})
        self.assertEqual(ratios, {})
        self.assertFalse(qeeg_ui.tbr_is_flagged(mean, usable))

    def test_raw_trace_reminder_is_visible_above_the_grid(self):
        self.panel.refresh(force=True)
        self.assertTrue(self.panel.raw_reminder.isVisible())
        self.assertEqual(self.panel.raw_reminder.text(), qeeg_ui.RAW_TRACE_REMINDER)
        self.assertIn('artifact looks like a region', self.panel.raw_reminder.text())
        layout = self.panel.layout()
        self.assertLess(layout.indexOf(self.panel.raw_reminder),
                        layout.indexOf(self.panel.stack))
        received = []
        self.panel.raw_trace_requested.connect(lambda: received.append(True))
        self.panel.raw_link.click()
        self.assertEqual(received, [True])

    @unittest.skipUnless(qeeg_ui.regions is not None, 'regions.py is not installed')
    def test_region_table_present_for_the_21_channel_grid(self):
        self.panel.refresh(force=True)
        table = self.panel.sheet.region_table
        self.assertTrue(table.isVisible())
        self.assertEqual(table.band_order, list(qeeg_ui.BAND_KEYS))
        self.assertEqual(len(table.region_order), len(qeeg_ui.regions.REGION_ORDER))
        self.assertEqual(len(table.cell_labels),
                         len(qeeg_ui.regions.REGION_ORDER) * len(qeeg_ui.BAND_KEYS))
        self.assertEqual(len(self.panel.report_metadata()['channels']), 21)
        meta = self.panel.region_metadata()
        self.assertTrue(meta['available'])
        for band in qeeg_ui.BAND_KEYS:
            self.assertEqual(set(meta['values'][band]),
                             set(qeeg_ui.regions.REGION_ORDER))
        # Every region was scored from the measured 21-channel set, not padded.
        sample = table.cell_labels[('Alpha', 'left_medial')]
        self.assertIn('mean z', sample.toolTip())
        self.assertIn('sites', sample.toolTip())

    def test_region_table_hides_cleanly_without_the_module(self):
        self.panel.refresh(force=True)
        with mock.patch.object(qeeg_ui, 'regions', None):
            self.panel.refresh()
        self.assertFalse(self.panel.sheet.region_table.isVisible())
        self.assertFalse(self.panel.report_metadata()['regions']['available'])
        self.assertIn('regions module not available',
                      self.panel.report_metadata()['regions']['reason'])

    # ------------------------------------------------------------- EC / EO
    def session_payload(self, condition, device='dragon_eeg', reference_kind='session',
                        samples=None, key='x'):
        return {'samples': samples if samples is not None else self.samples,
                'fs': FS, 'channels': self.channels, 'label': key,
                'condition': condition, 'device': device,
                'reference_kind': reference_kind, 'session_id': key}

    def test_compare_refuses_mismatched_devices_with_a_visible_message(self):
        left = self.session_payload('eyes closed', device='dragon_eeg', key='ec')
        right = self.session_payload('eyes open', device='brainbit', key='eo')
        self.assertIsNone(self.panel.compare_sessions(left, right))
        self.assertTrue(self.panel.compare_label.isVisible())
        self.assertIn('EC/EO comparison refused', self.panel.compare_label.text())
        self.assertIn('different devices', self.panel.compare_label.text())
        self.assertIsNone(self.panel.comparison_view)

    def test_compare_refuses_different_reference_kinds(self):
        left = self.session_payload('eyes closed', reference_kind='session', key='ec')
        right = self.session_payload('eyes open', reference_kind='norm', key='eo')
        self.assertIsNone(self.panel.compare_sessions(left, right))
        self.assertIn('different reference kinds', self.panel.compare_label.text())

    def test_compare_dialog_lists_the_subject_sessions(self):
        sessions = [{**self.session_payload('eyes closed', key='ec'),
                     'started_at': '2026-11-01 10:00'},
                    {**self.session_payload('eyes open', key='eo'),
                     'started_at': '2026-11-02 10:00'}]
        dialog = qeeg_ui.CompareDialog(sessions, self.panel)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.ec_combo.count(), 2)
        self.assertEqual(dialog.eo_combo.count(), 2)
        self.assertEqual(dialog.eo_combo.currentIndex(), 1)   # defaults to the EO session
        left, right = dialog.selection()
        self.assertEqual(left['picked_condition'], 'eyes closed')
        self.assertEqual(right['picked_condition'], 'eyes open')
        self.assertIn('eyes closed', dialog.ec_combo.itemText(0))

    def test_compare_pick_supplies_a_condition_not_stored_with_the_session(self):
        sessions = [{**self.session_payload('', key='ec'), 'started_at': '2026-11-01 10:00'},
                    {**self.session_payload('', key='eo'), 'started_at': '2026-11-02 10:00'}]
        dialog = qeeg_ui.CompareDialog(sessions, self.panel)
        self.addCleanup(dialog.close)
        left, right = dialog.selection()
        result = self.panel.compare_sessions(left, right)
        self.addCleanup(lambda: self.panel.comparison_view and
                        self.panel.comparison_view.close())
        self.assertIsNotNone(result)
        self.assertTrue(result['ok'], result.get('message'))
        self.assertIn('eyes closed', result['header'])
        self.assertIn('eyes open', result['header'])
        # A stored condition that contradicts the slot is refused, not relabelled.
        contradictory = dict(left)
        contradictory['condition'] = 'eyes open'
        self.assertIsNone(self.panel.compare_sessions(contradictory, right))
        self.assertIn('recorded as eyes open', self.panel.compare_label.text())

    def test_compare_delta_strip_is_eo_minus_ec(self):
        ec_samples, channels = band_session(theta_amp=3.0, beta_amp=6.0)
        eo_samples, _channels = band_session(theta_amp=12.0, beta_amp=6.0)
        reference = flat_norm_reference(channels)
        ec = self.session_payload('eyes closed', samples=ec_samples, key='ec')
        eo = self.session_payload('eyes open', samples=eo_samples, key='eo')
        result = qeeg_ui.compare_sessions(ec, eo, 'norm', reference)
        self.assertTrue(result['ok'], result.get('message'))
        delta = result['deltas'][('absolute', 'Theta')]
        self.assertIsNotNone(delta)
        self.assertGreater(delta, 0.0)          # EO theta above EC theta
        # The function always orders EC first, so swapping the arguments must
        # not change the strip (it is EO − EC, not "second minus first").
        swapped = qeeg_ui.compare_sessions(eo, ec, 'norm', reference)
        self.assertAlmostEqual(swapped['deltas'][('absolute', 'Theta')], delta)
        # Reverse the physiology: now EC has the higher theta, so the strip flips.
        ec_high = self.session_payload('eyes closed', samples=eo_samples, key='ec2')
        eo_low = self.session_payload('eyes open', samples=ec_samples, key='eo2')
        negative = qeeg_ui.compare_sessions(ec_high, eo_low, 'norm', reference)
        self.assertLess(negative['deltas'][('absolute', 'Theta')], 0.0)
        view = qeeg_ui.ComparisonView(result)
        self.addCleanup(view.close)
        self.assertEqual(len(view.sheet_ec.cells), 25)
        self.assertEqual(len(view.sheet_eo.cells), 25)
        view.resize(1720, 1240)
        view.show()
        QApplication.processEvents()
        with tempfile.TemporaryDirectory() as tmp:
            target = view.export_png(Path(tmp) / 'ec_eo.png')
            data = target.read_bytes()
        self.assertGreater(len(data), 10_000)
        self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')

    # -------------------------------------------------------- norm file load
    @unittest.skipUnless(qeeg_ui.norms is not None, 'norms.py is not installed')
    def test_norm_file_load_shows_source_n_and_age_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'neuroguide_export.json'
            write_norm_file(path, source='neuroguide', scope='21ch')
            self.assertTrue(self.panel.load_norm_file(path))
        self.assertEqual(self.panel.norm_meta['source'], 'neuroguide')
        self.assertEqual(self.panel.norm_meta['n'], 22)
        self.assertEqual(self.panel.norm_meta['age_band'], '30-39')
        text = self.panel.reference_label.text()
        self.assertIn('kind: norm', text)
        self.assertIn('imported reference (neuroguide)', text)
        self.assertIn('n=22', text)
        self.assertIn('age 30-39', text)
        self.assertNotIn('vs local norm', text)
        self.assertIn('not a normative clinical database', text)
        label = self.panel.norm_label.text()
        self.assertIn('source: neuroguide', label)
        self.assertIn('n=22', label)
        self.assertIn('age 30-39', label)
        self.assertEqual(self.panel.norm_scope_warning, '')
        self.assertNotIn('local norm', self.panel.status_label.text().lower())
        # A local norm is the only source allowed to call itself local.
        local_line = qeeg_ui.describe_reference({'kind': 'norm', 'label': 'x', 'n': 20,
                                                 'age_band': '18-25', 'source': 'local'})
        self.assertIn('vs local norm', local_line)

    @unittest.skipUnless(qeeg_ui.norms is not None, 'norms.py is not installed')
    def test_norm_file_device_scope_mismatch_is_amber(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nineteen.json'
            write_norm_file(path, source='bsi', scope='19ch')
            self.assertTrue(self.panel.load_norm_file(path))
        self.assertIn('device scope mismatch', self.panel.norm_scope_warning)
        self.assertIn('19ch norm on 21ch recording', self.panel.norm_scope_warning)
        self.assertIn('device scope mismatch', self.panel.norm_label.text())
        self.assertIn('FFF3D6', self.panel.norm_label.styleSheet())
        self.assertIn('imported reference (bsi)', self.panel.reference_label.text())
        # The mismatch is carried into the report metadata, not just the widget.
        self.assertIn('device scope mismatch',
                      self.panel.report_metadata()['norm_scope_warning'])

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
        self.assertIn('Theta / Beta', text)
        self.assertIn('Channel mean θ/β', text)
        self.assertIn('read it with the rest of the sheet', text)
        self.assertIn('Region roll-up (mean z)', text)
        self.assertIn('Check the raw traces before trusting a map', text)
        self.assertNotIn('NeuroGuide', text)


if __name__ == '__main__':
    unittest.main()
