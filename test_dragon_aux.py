"""DragonEEG 24-channel path: TEST ONLY fixtures, no hardware, no BLE.

The fixture below is the documented 21 scalp sites plus the three auxiliary
poly channels a DragonEEG reports: an ECG-like D1, a flat D2 and a 50 Hz D3.
It proves that analysis and the head map use scalp channels only, that the aux
channels stay in the raw trace view and in storage, and that a single bad scalp
electrode no longer erases the whole band table.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import time
import unittest
from unittest import mock

import numpy as np
from PyQt6.QtWidgets import QApplication

import devices
import eeg_analysis as eeg
import simfeed
import topomap

FS = 250
SCALP = tuple(devices.DRAGON_DOCUMENTED_EEG_CHANNELS)          # 21 documented sites
AUX = ('D1', 'D2', 'D3')                                      # ECG-like / flat / 50 Hz
ALL = SCALP + AUX                                             # 24-piece channel set


def contacts(channels, age_s=0.):
    """Passing resistance (well below 1 MOhm) for every channel name."""
    return dict.fromkeys(channels, 200_000.), age_s


def fixture_24(n=1250, bad_scalp=None):
    """(rows, channels): 21 clean 10 Hz scalp channels plus the three aux channels.

    D1 carries sharp ECG-like spikes, D2 is flat and D3 carries 50 Hz line noise,
    so any code path that treats an aux channel as EEG fails the quality gates.
    """
    t = np.arange(n) / FS
    alpha = 20. * np.sin(2 * np.pi * 10. * t)
    scalp = np.repeat(alpha[:, None], len(SCALP), axis=1)
    scalp += np.random.default_rng(20261103).normal(0., .5, scalp.shape)
    if bad_scalp is not None:
        scalp[:, SCALP.index(bad_scalp)] = 0.                  # one flat scalp electrode
    ecg = np.zeros(n)
    for beat in range(1, int(n / FS) + 1):
        index = min(int(beat * FS), n - 1)
        ecg[index] = 800.                                      # QRS-like spike
    flat = np.zeros(n)
    line = 30. * np.sin(2 * np.pi * 50. * t)
    return np.column_stack([scalp, ecg, flat, line]), ALL


def scalp_rows(rows, channels=ALL):
    return rows[:, [channels.index(name) for name in SCALP]]


class AggregatePolicyTests(unittest.TestCase):
    def test_four_electrode_device_still_requires_every_electrode(self):
        self.assertEqual(devices.aggregate_min_valid(('O1', 'O2', 'T3', 'T4')), 4)

    def test_dragon_tolerates_a_few_bad_electrodes_but_never_fewer_than_four(self):
        self.assertEqual(devices.aggregate_min_valid(SCALP), 13)          # 60% of 21
        self.assertEqual(devices.aggregate_min_valid(SCALP[:5]), 4)


class DragonAuxAnalysisTests(unittest.TestCase):
    def analyze(self, samples, names, **kwargs):
        ohms, age = contacts(names)
        options = dict(contact_ohms=ohms, contact_age_s=age, notch_hz=60)
        options.update(kwargs)
        return eeg.analyze_window(samples, fs=FS, channels=names, **options)

    def test_scalp_only_analysis_is_valid_on_a_24_channel_device(self):
        rows, channels = fixture_24()
        report = self.analyze(scalp_rows(rows, channels), SCALP,
                              min_valid_channels=devices.aggregate_min_valid(SCALP))
        self.assertTrue(report['valid'], report['reason'])
        self.assertEqual(set(report['channels']), set(SCALP))
        self.assertEqual(report['dominant'], 'Alpha')
        self.assertEqual(report['channels_valid'], 21)
        self.assertEqual(report['channels_total'], 21)

    def test_analysing_the_aux_channels_would_block_the_aggregate(self):
        rows, channels = fixture_24()
        report = self.analyze(rows, channels)
        self.assertFalse(report['valid'])
        self.assertIn('D2', report['reason'])          # flatline
        self.assertIn('D3', report['reason'])          # 50 Hz contamination

    def test_one_bad_scalp_electrode_keeps_the_band_table(self):
        rows, channels = fixture_24(bad_scalp='F7')
        report = self.analyze(scalp_rows(rows, channels), SCALP,
                              min_valid_channels=devices.aggregate_min_valid(SCALP))
        self.assertTrue(report['valid'], report['reason'])
        self.assertEqual(report['channels_valid'], 20)
        self.assertEqual(report['channels_total'], 21)
        self.assertIn('20/21 channels valid', report['reason'])
        self.assertIn('F7', report['reason'])
        self.assertIsNotNone(report['bands']['Alpha']['power_uv2'])

    def test_stable_estimate_uses_scalp_channels_only(self):
        rows, channels = fixture_24(n=3125)
        ohms, age = contacts(SCALP)
        stable = eeg.analyze_stable(scalp_rows(rows, channels), fs=FS, channels=SCALP,
                                    contact_ohms=ohms, contact_age_s=age, notch_hz=60,
                                    min_valid_channels=devices.aggregate_min_valid(SCALP))
        self.assertTrue(stable['valid'], stable['reason'])
        self.assertEqual(set(stable['channels']), set(SCALP))
        self.assertEqual(stable['channels_valid'], 21)

    def test_topomap_over_the_21_scalp_channels_is_finite(self):
        rows, channels = fixture_24()
        report = self.analyze(scalp_rows(rows, channels), SCALP)
        self.assertTrue(report['valid'], report['reason'])
        powers = {name: report['channels'][name]['relative_pct'] for name in SCALP}
        field = topomap.topomap_band(powers, 'Alpha', list(SCALP), grid_n=64)
        inside = np.isfinite(field)
        self.assertTrue(inside.any())
        self.assertEqual(field.shape, (64, 64))
        self.assertGreater(field[inside].min(), 90.)     # every channel is ~pure alpha

    def test_topomap_rejects_aux_names_so_callers_must_filter(self):
        values = {name: {'Alpha': 10.} for name in ALL}
        with self.assertRaises(ValueError):
            topomap.topomap_band(values, 'Alpha', list(ALL), grid_n=32)
        field = topomap.topomap_band(values, 'Alpha', list(SCALP), grid_n=32)
        self.assertTrue(np.isfinite(field).any())


class DragonAuxWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        from main import NeuroSyncApp
        self.main_module = __import__('main')
        self.window = NeuroSyncApp(autostart=False)
        self.addCleanup(self.window.close)
        info = dict(simfeed.device_info_for('dragon'))
        info.update(channels=list(ALL), scalp_channels=list(SCALP),
                    poly_channels=list(AUX), fs=FS,
                    channel_source='TEST ONLY 24-channel fixture table',
                    device_label='TEST ONLY DragonEEG fixture (21 scalp + 3 aux)')
        self.window.apply_device_info(info)

    def feed(self, bad_scalp=None):
        rows, channels = fixture_24(bad_scalp=bad_scalp)
        self.window.handle_batch({'samples_uv': rows.tolist(),
                                  'packnums': [int(n) for n in np.arange(len(rows)) // 2],
                                  'arrived_monotonic': time.monotonic(),
                                  'sample_count': len(rows), 'gap_events': 0})
        self.window.handle_contact({'ohms': dict.fromkeys(channels, 200_000.),
                                    'measured_monotonic': time.monotonic()})

    def test_window_keeps_aux_in_the_raw_view_but_analyses_scalp_only(self):
        self.feed()
        self.window.analyze_now()
        report = self.window.last_report
        self.assertTrue(report['valid'], report['reason'])
        self.assertEqual(set(report['channels']), set(SCALP))
        self.assertFalse(set(AUX) & set(report['channels']))
        self.assertEqual(len(self.window.channels), 24)            # raw trace view keeps all
        self.assertEqual(set(self.window.raw_curves), set(ALL))
        self.assertIn('relative power on 21 schematic electrode positions',
                      self.window.map_status.text())
        self.assertIsNotNone(self.window.map_image.image)

    def test_update_topomap_ignores_aux_names_and_never_raises(self):
        self.feed()
        self.window.analyze_now()
        report = dict(self.window.last_report)
        report['channels'] = dict(report['channels'])
        for name in AUX:                    # a report that still carries aux names
            report['channels'][name] = {'valid': True, 'relative_pct': {'Alpha': 99.0}}
        self.window.last_report = report
        self.window.update_topomap()        # ValueError before the fix
        self.assertIn('relative power on 21 schematic electrode positions',
                      self.window.map_status.text())

    def test_window_reports_n_of_m_channels_valid(self):
        self.feed(bad_scalp='F7')
        self.window.analyze_now()
        report = self.window.last_report
        self.assertTrue(report['valid'], report['reason'])
        self.assertEqual(report['channels_valid'], 20)
        self.assertIn('20/21 channels valid', self.window.quality_label.text())
        self.assertIn('F7', self.window.quality_label.text())

    def test_map_failure_is_reported_instead_of_escaping_the_timer(self):
        self.feed()
        self.window.analyze_now()
        with mock.patch.object(self.main_module.topomap, 'topomap_band',
                               side_effect=RuntimeError('TEST ONLY map failure')):
            self.window.update_topomap()
        self.assertIn('Head map unavailable', self.window.map_status.text())
        self.assertIsNotNone(self.window.last_report)

    def test_saved_session_keeps_the_aux_channels_in_storage(self):
        import storage
        import tempfile
        from pathlib import Path
        self.feed()
        self.window.subject = {'name': 'TEST ONLY Dragon', 'phone': '000',
                               'key': storage.subject_key('TEST ONLY Dragon', '000')}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.main_module, 'SUBJECT_ROOT', Path(tmp)), \
                mock.patch.object(self.window, '_session_charts', return_value={}):
            self.window.save_session_clicked()
            sessions = storage.list_sessions(Path(tmp), 'TEST ONLY Dragon', '000')
            self.assertEqual(len(sessions), 1)
            loaded = storage.load_session(sessions[0]['path'])
        summary = loaded['summary']
        self.assertEqual(summary['channels'], list(ALL))          # 24 raw channels stored
        self.assertEqual(summary['poly_channels'], list(AUX))
        self.assertEqual(len(summary['scalp_channels']), 21)
        self.assertEqual(set(summary['analysis']['channels']), set(SCALP))
        self.assertEqual(len(loaded['raw_header']), 3 + len(ALL))


if __name__ == '__main__':
    unittest.main()
