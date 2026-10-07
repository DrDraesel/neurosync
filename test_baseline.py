"""Tests for baseline.py: per-subject baseline capture, storage and delta reporting.

Every numeric fixture below is TEST ONLY (synthetic sines and hand-written
records); nothing here originates from a headset. The Qt tests run offscreen
with a temporary recordings root and never touch the real recordings folder.
"""
import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np

import baseline
import eeg_analysis as eeg
import storage
from eeg_analysis import BANDS, CHANNELS

try:
    from PyQt6.QtWidgets import QApplication, QDialog
    import main
except ImportError:  # pragma: no cover - Qt is a hard dependency in practice
    main = None
    QApplication = None
    QDialog = None

NAME = 'Jane Doe'
PHONE = '+1 (555) 010-2030'
KEY = 'jane_doe_15550102030'
FORBIDDEN_CLAIMS = ('mood', 'emotion', 'anxiety', 'depress', 'iq', 'stress level',
                    'attention level', 'sleep stage', 'personality', 'you are')


def test_only_sine(hz=10., amplitude=10., n=1250, fs=250):
    """TEST ONLY: four identical sine-wave channels in microvolts."""
    t = np.arange(n) / fs
    return np.repeat((amplitude * np.sin(2 * np.pi * hz * t))[:, None], 4, axis=1)


def test_only_window_report(hz=10., n=1250, fs=250):
    """TEST ONLY: a real five-second analysis of the synthetic sine above."""
    return eeg.analyze_window(test_only_sine(hz=hz, n=n, fs=fs), fs=fs, notch_hz=None,
                              contact_ohms=dict.fromkeys(CHANNELS, 500_000.), contact_age_s=0.)


def test_only_stable_report(n=3750, fs=250):
    """TEST ONLY: a real multi-epoch estimate of the synthetic sine above."""
    return eeg.analyze_stable(test_only_sine(n=n, fs=fs), fs=fs, notch_hz=None,
                              contact_ohms=dict.fromkeys(CHANNELS, 500_000.), contact_age_s=0.)


def synthetic_report(relative=None, peak_hz=9.8, dominant='Theta', valid=True, channels=CHANNELS):
    """TEST ONLY hand-written analysis report shaped like eeg_analysis output."""
    relative = relative if relative is not None else {
        'Delta': 6., 'Theta': 20., 'Alpha': 39., 'Beta': 25., 'Gamma': 10.}
    return {
        'valid': valid, 'ready': True, 'reason': 'ok' if valid else 'TEST ONLY: O1 flatline',
        'samples': 1250, 'duration_s': 5.0,
        'bands': {name: {'low_hz': low, 'high_hz': high, 'power_uv2': 2.0,
                         'relative_pct': relative[name]}
                  for name, (low, high) in BANDS.items() if name in relative},
        'channels': {name: {'valid': True, 'flags': [], 'dominant': 'Alpha', 'peak_hz': 9.5,
                            'powers_uv2': {}, 'relative_pct': {}} for name in channels},
        'dominant': dominant if valid else None,
        'dominant_pct': relative.get(dominant) if valid else None,
        'peak_hz': peak_hz if valid else None,
        'conclusion': 'TEST ONLY',
    }


def synthetic_record(relative=None, peak_hz=10.2, dominant='Alpha', condition='eyes closed',
                     channels=CHANNELS, current=None):
    """TEST ONLY hand-written baseline record (numbers are fabricated fixtures)."""
    relative = relative if relative is not None else {
        'Delta': 5., 'Theta': 10., 'Alpha': 43., 'Beta': 30., 'Gamma': 12.}
    profile = {
        'source': 'stable', 'source_label': baseline.SOURCE_LABELS['stable'],
        'bands': {name: {'power_uv2': 1.0 + index, 'relative_pct': value}
                  for index, (name, value) in enumerate(relative.items())},
        'dominant': dominant,
        'dominant_pct': relative.get(dominant),
        'peak_hz': peak_hz,
        'channels': {name: {'dominant': 'Alpha', 'peak_hz': 10.0, 'valid': True, 'flags': []}
                     for name in channels},
        'quality': {'valid': True, 'reason': 'ok', 'samples': 3750, 'duration_s': 15.0,
                    'epochs_total': 7, 'epochs_used': 7, 'seconds_used': 14.0},
    }
    payload = {
        'baseline_version': baseline.BASELINE_VERSION, 'kind': baseline.KIND,
        'set_at': '2026-09-28T10:00:00+00:00', 'condition': condition,
        'subject_name': NAME, 'subject_phone': PHONE, 'subject_key': KEY,
        'device_type': 'brainbit_classic', 'device_label': 'BrainBit Classic',
        'channel_source': 'SDK channel table', 'channels': list(channels), 'fs_hz': 250,
        'profile': profile, 'app_version': baseline.APP_VERSION,
    }
    if current is not None:
        payload['session_id'] = current
    return baseline.validate_baseline(payload)


def session_summary(report=None, stable=None, **extra):
    """TEST ONLY saved-summary shape for the History/list paths."""
    summary = {'started_at': '2026-09-28T12:00:00+00:00', 'device_type': 'brainbit_classic',
               'device_label': 'BrainBit Classic', 'channels': list(CHANNELS), 'duration_s': 15.0,
               'channel_source': 'SDK channel table', 'fs_hz': 250}
    if report is not None:
        summary['analysis'] = report
    if stable is not None:
        summary['stable_estimate'] = stable
    summary.update(extra)
    return summary


def save_test_only_session(root, summary):
    return storage.save_session(root, NAME, PHONE, summary,
                                ['sample_index', 'raw_O1_uV'], [[0, 1.0]])


class BaselineModuleTests(unittest.TestCase):
    def test_public_api_surface(self):
        self.assertEqual(baseline.BASELINE_FILE, 'baseline.json')
        self.assertEqual(baseline.CONDITIONS, ('eyes closed', 'eyes open', 'other'))
        self.assertIsInstance(baseline.APP_VERSION, str)
        self.assertTrue(baseline.APP_VERSION)
        self.assertEqual(set(baseline.SOURCES), {'stable', 'five_second'})
        for name in ('normalize_condition', 'profile_from_report', 'build_baseline',
                     'validate_baseline', 'save_baseline', 'load_baseline', 'compare',
                     'one_line', 'profile_line', 'interpretive_note', 'best_estimate',
                     'vs_baseline_for_summary', 'band_table', 'reason_text'):
            self.assertTrue(callable(getattr(baseline, name)), name)


class BaselineCaptureTests(unittest.TestCase):
    def test_condition_is_required_and_stored_exactly_as_picked(self):
        report = test_only_window_report()
        for condition in baseline.CONDITIONS:
            with self.subTest(condition=condition):
                record = baseline.build_baseline(report, condition)
                self.assertEqual(record['condition'], condition)
        self.assertEqual(baseline.build_baseline(report, '  eyes open  ')['condition'], 'eyes open')
        for bad in (None, '', '   ', 'relaxed', 'EYES CLOSED', 5, True, ['eyes closed']):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    baseline.build_baseline(report, bad)
        self.assertEqual(baseline.normalize_condition('other'), 'other')
        with self.assertRaises(ValueError):
            baseline.normalize_condition('unknown')

    def test_capture_requires_a_passing_and_complete_estimate(self):
        with self.assertRaises(ValueError):
            baseline.build_baseline(None, 'eyes closed')
        with self.assertRaises(ValueError):
            baseline.build_baseline([1, 2, 3], 'eyes closed')
        with self.assertRaises(ValueError):
            baseline.build_baseline(synthetic_report(valid=False), 'eyes closed')
        incomplete = synthetic_report()
        del incomplete['bands']['Gamma']
        with self.assertRaises(ValueError):
            baseline.build_baseline(incomplete, 'eyes closed')
        for band in BANDS:
            with self.subTest(band=band):
                broken = synthetic_report()
                broken['bands'][band]['relative_pct'] = None
                with self.assertRaises(ValueError):
                    baseline.build_baseline(broken, 'eyes closed')
                broken = synthetic_report()
                broken['bands'][band]['power_uv2'] = float('nan')
                with self.assertRaises(ValueError):
                    baseline.build_baseline(broken, 'eyes closed')
        with self.assertRaises(ValueError):
            baseline.profile_from_report(test_only_window_report(), source='guesswork')
        with self.assertRaises(ValueError):
            baseline.build_baseline(test_only_window_report(), 'eyes closed',
                                    captured_at='2026-09-28')

    def test_profile_from_a_real_five_second_analysis(self):
        report = test_only_window_report()
        record = baseline.build_baseline(
            report, 'eyes closed', subject_name=NAME, subject_phone=PHONE,
            device={'device_key': 'brainbit_classic', 'device_label': 'BrainBit Classic',
                    'channel_source': 'SDK channel table'},
            channels=list(CHANNELS), fs_hz=250, source='five_second')
        self.assertEqual(record['condition'], 'eyes closed')
        self.assertEqual(record['subject_key'], KEY)
        self.assertEqual(record['device_type'], 'brainbit_classic')
        self.assertEqual(record['device_label'], 'BrainBit Classic')
        self.assertEqual(record['channels'], list(CHANNELS))
        self.assertEqual(record['app_version'], baseline.APP_VERSION)
        self.assertEqual(record['kind'], baseline.KIND)
        self.assertEqual(record['captured_from'], 'live')
        self.assertTrue(record['baseline_note'])
        self.assertTrue(record['condition_note'])
        self.assertTrue(record['limits'])
        self.assertEqual(datetime.fromisoformat(record['set_at']).utcoffset().total_seconds(), 0)
        profile = record['profile']
        self.assertEqual(profile['source'], 'five_second')
        self.assertEqual(profile['source_label'], baseline.SOURCE_LABELS['five_second'])
        self.assertEqual(profile['dominant'], 'Alpha')
        self.assertAlmostEqual(profile['peak_hz'], 10., delta=.5)
        self.assertAlmostEqual(sum(band['relative_pct'] for band in profile['bands'].values()),
                               100., delta=.1)
        for name, band in profile['bands'].items():
            self.assertIn(name, BANDS)
            if name == 'Alpha':
                # a 10 Hz sine puts (numerically) all of its power in 8-13 Hz
                self.assertAlmostEqual(band['power_uv2'], 50., delta=2.)
                self.assertAlmostEqual(band['relative_pct'], 100., delta=.1)
            else:
                # every other band carries only leakage-level power (~1e-28 uV^2)
                self.assertAlmostEqual(band['power_uv2'], 0., delta=.5)
                self.assertAlmostEqual(band['relative_pct'], 0., delta=.1)
        self.assertEqual(set(profile['channels']), set(CHANNELS))
        for channel in profile['channels'].values():
            self.assertEqual(channel['dominant'], 'Alpha')
            self.assertAlmostEqual(channel['peak_hz'], 10., delta=.5)
            self.assertTrue(channel['valid'])
        self.assertEqual(profile['quality']['samples'], 1250)
        self.assertEqual(profile['quality']['duration_s'], 5.)
        json.dumps(record, allow_nan=False)

    def test_profile_from_a_real_stable_estimate_records_its_source(self):
        report = test_only_stable_report()
        self.assertTrue(report['valid'])
        record = baseline.build_baseline(report, 'eyes open', subject_name=NAME,
                                        subject_phone=PHONE, channels=list(CHANNELS),
                                        fs_hz=250, source='stable')
        self.assertEqual(record['profile']['source'], 'stable')
        self.assertEqual(record['profile']['quality']['epochs_used'], 7)
        self.assertAlmostEqual(record['profile']['quality']['seconds_used'], 14.)
        self.assertEqual(record['condition'], 'eyes open')
        self.assertIn('eyes open', baseline.profile_line(record))

    def test_build_never_mutates_the_caller_report(self):
        report = test_only_window_report()
        before = json.dumps(report, allow_nan=False)
        baseline.build_baseline(report, 'eyes closed', channels=list(CHANNELS))
        self.assertEqual(json.dumps(report, allow_nan=False), before)


class BaselineStorageTests(unittest.TestCase):
    def test_roundtrip_replace_and_atomicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            record = synthetic_record()
            path = baseline.save_baseline(root, NAME, PHONE, record)
            self.assertEqual(path, root / KEY / baseline.BASELINE_FILE)
            self.assertTrue(storage._inside(root, path))
            self.assertEqual(list((root / KEY).glob('*.tmp')), [])
            on_disk = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(on_disk['condition'], 'eyes closed')
            self.assertEqual(on_disk['profile']['bands']['Alpha']['relative_pct'], 43.)

            info = baseline.load_baseline(root, NAME, PHONE)
            self.assertTrue(info['present'])
            self.assertEqual(info['error'], '')
            self.assertEqual(info['record']['condition'], 'eyes closed')
            self.assertEqual(info['record']['profile']['bands'], record['profile']['bands'])
            self.assertEqual(info['record']['profile']['channels'], record['profile']['channels'])
            self.assertEqual(info['record']['profile']['peak_hz'], 10.2)
            json.dumps(info['record'], allow_nan=False)

            second = synthetic_record(condition='eyes open', peak_hz=9.4)
            baseline.save_baseline(root, NAME, PHONE, second)
            self.assertEqual(sorted(entry.name for entry in (root / KEY).iterdir()),
                             [baseline.BASELINE_FILE])
            reloaded = baseline.load_baseline(root, NAME, PHONE)['record']
            self.assertEqual(reloaded['condition'], 'eyes open')
            self.assertEqual(reloaded['profile']['peak_hz'], 9.4)

    def test_baseline_does_not_touch_saved_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            folder = save_test_only_session(root, session_summary(report=test_only_window_report()))
            baseline.save_baseline(root, NAME, PHONE, synthetic_record())
            self.assertTrue((folder / storage.SUMMARY_FILE).is_file())
            self.assertTrue((folder / storage.RAW_FILE).is_file())
            self.assertEqual(len(storage.list_sessions(root, NAME, PHONE)), 1)
            subjects = storage.list_subjects(root)
            self.assertEqual(subjects[0]['sessions'], 1)

    def test_missing_and_unusable_files_are_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            info = baseline.load_baseline(root, NAME, PHONE)
            self.assertFalse(info['present'])
            self.assertIsNone(info['record'])
            self.assertEqual(info['error'], '')

            folder = root / KEY
            folder.mkdir(parents=True)
            target = folder / baseline.BASELINE_FILE
            target.write_text('{ this is not json', encoding='utf-8')
            info = baseline.load_baseline(root, NAME, PHONE)
            self.assertTrue(info['present'])
            self.assertIsNone(info['record'])
            self.assertTrue(info['error'])

            target.write_text('[1, 2, 3]', encoding='utf-8')
            self.assertIsNone(baseline.load_baseline(root, NAME, PHONE)['record'])

            payload = synthetic_record()
            del payload['condition']
            target.write_text(json.dumps(payload), encoding='utf-8')
            info = baseline.load_baseline(root, NAME, PHONE)
            self.assertIsNone(info['record'])
            self.assertIn('not a usable baseline', info['error'])

            payload = synthetic_record()
            payload['profile']['bands']['Alpha']['relative_pct'] = None
            target.write_text(json.dumps(payload), encoding='utf-8')
            self.assertIsNone(baseline.load_baseline(root, NAME, PHONE)['record'])

            payload = synthetic_record()
            payload['set_at'] = ''
            target.write_text(json.dumps(payload), encoding='utf-8')
            self.assertIsNone(baseline.load_baseline(root, NAME, PHONE)['record'])

    def test_invalid_records_are_refused_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            for bad in (None, ['x'], {}, {'condition': 'eyes closed'},
                        {'set_at': '2026-09-28T10:00:00+00:00', 'condition': 'unknown',
                         'profile': {'bands': {}}}):
                with self.subTest(bad=repr(bad)):
                    with self.assertRaises(ValueError):
                        baseline.save_baseline(root, NAME, PHONE, bad)
            self.assertFalse(root.exists())

    def test_hostile_subject_values_stay_inside_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            path = baseline.save_baseline(root, '../../etc', '..', synthetic_record())
            self.assertTrue(storage._inside(root, path))
            self.assertTrue(path.is_file())
            self.assertEqual(path.parent.parent, root)
            self.assertNotIn('..', path.parent.name)
            self.assertIsNone(baseline.load_baseline(root, 'somebody else', '9')['record'])

    def test_write_json_atomic_refuses_bad_payloads_without_partial_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'record.json'
            for bad in (None, [1], 'text', 5, {'x': object()}, {'x': float('nan')},
                        {'x': float('inf')}):
                with self.subTest(bad=repr(bad)):
                    with self.assertRaises(ValueError):
                        storage.write_json_atomic(target, bad)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])
            storage.write_json_atomic(target, {'ok': 1})
            self.assertEqual(json.loads(target.read_text(encoding='utf-8')), {'ok': 1})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])
            self.assertEqual(storage.read_json_object(target), {'ok': 1})
            self.assertIsNone(storage.read_json_object(Path(tmp) / 'missing.json'))


class BaselineDeltaTests(unittest.TestCase):
    def test_per_band_peak_and_dominant_deltas(self):
        record = synthetic_record()
        report = synthetic_report()
        delta = baseline.compare(record, report, source='five_second')
        self.assertTrue(delta['available'])
        self.assertEqual(delta['reason'], 'ok')
        self.assertEqual(delta['current_source'], 'five_second')
        self.assertEqual(delta['relative_pct_delta'],
                         {'Delta': 1., 'Theta': 10., 'Alpha': -4., 'Beta': -5., 'Gamma': -2.})
        self.assertAlmostEqual(delta['peak_hz_delta'], -0.4, places=3)
        self.assertEqual(delta['baseline_peak_hz'], 10.2)
        self.assertEqual(delta['current_peak_hz'], 9.8)
        self.assertEqual(delta['dominant_from'], 'Alpha')
        self.assertEqual(delta['dominant_to'], 'Theta')
        self.assertIs(delta['dominant_changed'], True)
        self.assertEqual(delta['largest_change'], {'band': 'Theta', 'delta_pp': 10.})
        self.assertEqual(delta['baseline_relative_pct']['Alpha'], 43.)
        self.assertEqual(delta['current_relative_pct']['Alpha'], 39.)
        self.assertEqual(delta['baseline_condition'], 'eyes closed')
        self.assertFalse(delta['channels_changed'])
        self.assertIn('vs baseline', baseline.one_line(delta))
        self.assertEqual(baseline.short_dominant_change(delta), 'Alpha -> Theta')
        self.assertEqual(baseline.band_table(delta)[2], ('Alpha', 43., 39., -4.))
        json.dumps(delta, allow_nan=False)

    def test_dominant_unchanged_is_reported_as_unchanged(self):
        delta = baseline.compare(synthetic_record(), synthetic_report(dominant='Alpha'),
                                 source='stable')
        self.assertIs(delta['dominant_changed'], False)
        self.assertEqual(delta['dominant_from'], 'Alpha')
        self.assertEqual(delta['dominant_to'], 'Alpha')
        self.assertEqual(baseline.short_dominant_change(delta), 'Alpha -> Alpha')
        self.assertIn('dominant unchanged (Alpha)', baseline.one_line(delta))

    def test_mixed_dominant_is_carried_through(self):
        delta = baseline.compare(synthetic_record(dominant='mixed'), synthetic_report(),
                                 source='stable')
        self.assertEqual(delta['dominant_from'], 'mixed')
        self.assertIs(delta['dominant_changed'], True)
        self.assertIn('mixed -> Theta', baseline.one_line(delta))
        self.assertIn('mixed measured bands', baseline.profile_line(synthetic_record(dominant='mixed')))

    def test_missing_band_and_none_values_leave_none_deltas(self):
        report = synthetic_report()
        del report['bands']['Alpha']
        delta = baseline.compare(synthetic_record(), report, source='stable')
        self.assertTrue(delta['available'])
        self.assertIsNone(delta['relative_pct_delta']['Alpha'])
        self.assertEqual(delta['relative_pct_delta']['Theta'], 10.)
        self.assertIsNone(delta['current_relative_pct']['Alpha'])
        self.assertEqual(delta['largest_change']['band'], 'Theta')

        report = synthetic_report()
        report['bands']['Beta']['relative_pct'] = None
        delta = baseline.compare(synthetic_record(), report, source='stable')
        self.assertIsNone(delta['relative_pct_delta']['Beta'])

        baseline_missing_peak = synthetic_record()
        baseline_missing_peak['profile']['peak_hz'] = None
        delta = baseline.compare(baseline.validate_baseline(baseline_missing_peak),
                                 synthetic_report(), source='stable')
        self.assertIsNone(delta['peak_hz_delta'])
        self.assertIn('vs baseline', baseline.one_line(delta))

    def test_no_baseline_or_no_usable_current_estimate(self):
        delta = baseline.compare(None, synthetic_report())
        self.assertFalse(delta['available'])
        self.assertEqual(delta['reason'], 'no_baseline')
        self.assertIn('no baseline set', baseline.one_line(delta))
        self.assertEqual(baseline.compare({'condition': 'eyes closed'}, synthetic_report())['reason'],
                         'no_baseline')
        delta = baseline.compare(synthetic_record(), None, source='stable')
        self.assertFalse(delta['available'])
        self.assertEqual(delta['reason'], 'no_current_estimate')
        delta = baseline.compare(synthetic_record(), synthetic_report(valid=False), source='stable')
        self.assertFalse(delta['available'])
        self.assertEqual(delta['reason'], 'current_estimate_not_valid')
        self.assertIsNone(delta['dominant_changed'])
        self.assertEqual(delta['current_relative_pct']['Theta'], 20.)
        self.assertIn('quality gates', baseline.one_line(delta))
        empty = synthetic_report()
        empty['bands'] = {}
        delta = baseline.compare(synthetic_record(), empty, source='stable')
        self.assertFalse(delta['available'])
        self.assertEqual(delta['reason'], 'no_comparable_band_numbers')
        self.assertEqual(baseline.band_table(delta)[0], ('Delta', None, None, None))

    def test_channel_and_device_changes_are_flagged(self):
        report = synthetic_report(channels=('O1', 'O2'))
        delta = baseline.compare(synthetic_record(), report, source='stable',
                                 device={'device_key': 'dragon_eeg', 'device_label': 'DragonEEG'})
        self.assertIs(delta['channels_changed'], True)
        self.assertIs(delta['device_changed'], True)
        same = baseline.compare(synthetic_record(), synthetic_report(), source='stable',
                                device={'device_key': 'brainbit_classic',
                                        'device_label': 'BrainBit Classic'})
        self.assertIs(same['device_changed'], False)
        unknown = baseline.compare(synthetic_record(), synthetic_report(), source='stable')
        self.assertIsNone(unknown['device_changed'])

    def test_delta_is_json_safe_and_never_raises_on_junk(self):
        for bad in (None, 'text', 5, [], {'profile': 'no'}, {'profile': {}}):
            with self.subTest(bad=repr(bad)):
                delta = baseline.compare(bad, synthetic_report(), source='stable')
                json.dumps(delta, allow_nan=False)
                self.assertFalse(delta['available'])
        for bad_report in (None, 'text', 5, [], {'valid': True}, {'valid': True, 'bands': 'no'}):
            with self.subTest(bad_report=repr(bad_report)):
                delta = baseline.compare(synthetic_record(), bad_report, source='stable')
                json.dumps(delta, allow_nan=False)
        self.assertEqual(baseline.one_line(None), '')
        self.assertEqual(baseline.reason_text('nonsense'), 'not comparable')


class BaselineSummaryTests(unittest.TestCase):
    def test_best_estimate_prefers_stable_then_five_second(self):
        stable = test_only_stable_report()
        window = test_only_window_report()
        report, source = baseline.best_estimate(session_summary(window, stable))
        self.assertIs(report, stable)
        self.assertEqual(source, 'stable')
        report, source = baseline.best_estimate(session_summary(window))
        self.assertIs(report, window)
        self.assertEqual(source, 'five_second')
        self.assertEqual(baseline.best_estimate(session_summary()), (None, None))
        self.assertEqual(baseline.best_estimate(None), (None, None))
        self.assertEqual(baseline.best_estimate({'stable_estimate': {'valid': False},
                                                 'analysis': {'valid': False}}), (None, None))

    def test_vs_baseline_for_summary_only_with_a_baseline(self):
        self.assertIsNone(baseline.vs_baseline_for_summary(None, test_only_window_report()))
        self.assertIsNone(baseline.vs_baseline_for_summary({}, test_only_window_report()))
        payload = baseline.vs_baseline_for_summary(synthetic_record(), synthetic_report(),
                                                   source='stable')
        self.assertTrue(payload['available'])
        self.assertEqual(payload['baseline_condition'], 'eyes closed')
        json.dumps(payload, allow_nan=False)
        saved = session_summary(test_only_window_report())
        saved['vs_baseline'] = payload
        self.assertIn('vs_baseline', json.loads(json.dumps(saved, allow_nan=False)))

    def test_saved_session_summary_carries_vs_baseline_when_a_baseline_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            baseline.save_baseline(root, NAME, PHONE, synthetic_record())
            summary = session_summary(stable=test_only_stable_report())
            summary['vs_baseline'] = baseline.vs_baseline_for_summary(
                baseline.load_baseline(root, NAME, PHONE)['record'],
                test_only_stable_report(), source='stable')
            folder = save_test_only_session(root, summary)
            loaded = storage.load_session(folder)['summary']
            self.assertIn('vs_baseline', loaded)
            self.assertTrue(loaded['vs_baseline']['available'])
            self.assertEqual(loaded['vs_baseline']['baseline_condition'], 'eyes closed')
            self.assertEqual(loaded['vs_baseline']['current_source'], 'stable')


class BaselineWordingTests(unittest.TestCase):
    def assert_no_person_claims(self, text):
        lowered = text.lower()
        for word in FORBIDDEN_CLAIMS:
            self.assertNotIn(word, lowered, f'{word!r} must not appear in {text!r}')

    def test_profile_line_names_the_pattern_condition_and_source(self):
        line = baseline.profile_line(synthetic_record())
        self.assert_no_person_claims(line)
        self.assertIn('Baseline: Alpha-dominant', line)
        self.assertIn('peak 10.2 Hz', line)
        self.assertIn('43%', line)
        self.assertIn('(eyes closed)', line)
        self.assertIn('set 2026-09-28', line)
        self.assertIn('BrainBit Classic', line)
        self.assertIn('stable multi-epoch estimate', line)
        self.assertEqual(baseline.profile_line(None), 'No baseline set')

    def test_one_line_shows_deltas_direction_and_transition(self):
        line = baseline.one_line(baseline.compare(synthetic_record(), synthetic_report(),
                                                 source='stable'))
        self.assert_no_person_claims(line)
        self.assertTrue(line.startswith('vs baseline: '))
        self.assertIn('largest band change Theta +10.0 pp', line)
        self.assertIn('peak 10.2 -> 9.8 Hz (-0.4 Hz)', line)
        self.assertIn('Alpha -> Theta', line)
        unavailable = baseline.one_line(baseline.compare(None, synthetic_report()))
        self.assertIn('no baseline set for this subject', unavailable)

    def test_interpretive_note_uses_only_the_allowed_pattern(self):
        note = baseline.interpretive_note(synthetic_record())
        self.assert_no_person_claims(note)
        self.assertTrue(note.startswith('Alpha-dominant rhythm, peak 10.2 Hz, 43%'))
        self.assertIn('EEG pattern description, not an assessment of the person', note)
        self.assertIn('in the literature, eyes-closed alpha dominance is commonly associated with '
                      'relaxed wakefulness.', note)
        self.assertEqual(baseline.interpretive_note(synthetic_record(condition='eyes open')),
                         'Alpha-dominant rhythm, peak 10.2 Hz, 43% - EEG pattern description, '
                         'not an assessment of the person.')
        self.assertEqual(baseline.interpretive_note(synthetic_record(condition='other')),
                         'Alpha-dominant rhythm, peak 10.2 Hz, 43% - EEG pattern description, '
                         'not an assessment of the person.')
        # The dominant band's own share is part of the honest measured description
        # (here Theta holds only 10% of 1-45 Hz), exactly as the Alpha cases above.
        theta_note = baseline.interpretive_note(synthetic_record(dominant='Theta',
                                                                 condition='eyes closed'))
        self.assertEqual(theta_note,
                         'Theta-dominant rhythm, peak 10.2 Hz, 10% - EEG pattern description, '
                         'not an assessment of the person.')
        self.assertNotIn('relaxed wakefulness', theta_note)
        self.assertEqual(baseline.interpretive_note(None), '')

    def test_stored_notes_carry_the_limits_and_the_condition_caveat(self):
        record = synthetic_record()
        self.assertIn('not a diagnosis', record['baseline_note'])
        self.assertIn('not a diagnosis', record['limits'])
        self.assertIn('same condition', record['condition_note'])
        self.assertIn('not a diagnosis', baseline.DELTA_NOTE)
        self.assertIn('not a mental-state', baseline.DELTA_NOTE)


@unittest.skipIf(main is None, 'Qt/main.py unavailable')
class BaselineUiTests(unittest.TestCase):
    """Offscreen UI tests: temporary recordings root, no hardware, no message boxes."""

    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'subjects'
        patcher = mock.patch.object(main, 'SUBJECT_ROOT', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.window = main.NeuroSyncApp(autostart=False)
        self.addCleanup(self.window.close)
        self.window.subject = {'name': NAME, 'phone': PHONE}
        self.window.subject_label.setText(f'Subject: {NAME}')
        self.window.load_baseline()

    def feed_fixture(self, batches=1):
        """TEST ONLY: feed synthetic 10 Hz sine batches (never device data)."""
        for _ in range(batches):
            rows = test_only_sine()
            self.window.handle_batch({
                'samples_uv': rows.tolist(),
                'packnums': [int(number) for number in np.arange(len(rows)) // 2],
                'arrived_monotonic': time.monotonic(), 'sample_count': len(rows),
                'gap_events': 0})
        self.window.handle_contact({'ohms': dict.fromkeys(CHANNELS, 100_000.),
                                    'measured_monotonic': time.monotonic()})

    def capture_baseline(self, condition='eyes closed'):
        self.feed_fixture(batches=3)
        self.window.analyze_stable_now()
        self.assertTrue(self.window.last_stable['valid'], self.window.last_stable['reason'])
        self.window.baseline_condition.setCurrentText(condition)
        with mock.patch.object(main.QMessageBox, 'information'), \
                mock.patch.object(main.QMessageBox, 'warning'):
            self.window.set_baseline_from_current()
        return self.root / KEY / baseline.BASELINE_FILE

    def test_panel_without_baseline_is_safe_and_explicit(self):
        self.assertIn('No baseline set', self.window.baseline_status.text())
        self.assertIn('no baseline set for this subject', self.window.baseline_delta_label.text())
        self.assertEqual(self.window.baseline_table.item(0, 3).text(), '—')
        self.assertIn('not a diagnosis', self.window.baseline_note.text())
        self.assertIn('same condition', self.window.baseline_note.text())
        self.window.analyze_now()
        self.assertIsNone(self.window.baseline)

    def test_panel_without_a_subject_asks_for_one(self):
        self.window.subject = None
        self.window.load_baseline()
        self.assertIn('set a subject first', self.window.baseline_status.text())

    def test_capture_from_the_stable_estimate_then_saved_summary_has_deltas(self):
        # The window is deliberately never shown: saving must also work on an
        # offscreen/never-rendered window whose trend widgets are still unshown.
        path = self.capture_baseline('eyes closed')
        self.assertTrue(path.is_file())
        self.assertEqual(list(path.parent.glob('*.tmp')), [])
        record = baseline.load_baseline(self.root, NAME, PHONE)['record']
        self.assertEqual(record['condition'], 'eyes closed')
        self.assertEqual(record['profile']['source'], 'stable')
        self.assertEqual(record['captured_from'], 'live')
        self.assertEqual(record['app_version'], baseline.APP_VERSION)
        self.assertTrue(record['subject_key'].startswith('jane_doe'))
        self.assertIn('Baseline: Alpha-dominant', self.window.baseline_status.text())
        self.assertIn('(eyes closed)', self.window.baseline_status.text())
        self.assertIn('not an assessment of the person', self.window.baseline_note.text())
        self.assertTrue(self.window.baseline_delta_label.text().startswith('vs baseline: '))
        self.assertNotIn('no baseline set', self.window.baseline_delta_label.text())
        self.assertEqual(self.window.baseline_table.item(2, 0).text(), 'Alpha')
        self.assertIn('%', self.window.baseline_table.item(2, 1).text())

        with mock.patch.object(main.QMessageBox, 'warning'):
            self.window.save_session_clicked()
        folders = [entry for entry in (self.root / KEY).iterdir() if entry.is_dir()]
        self.assertEqual(len(folders), 1)
        summary = storage.read_json_object(folders[0] / storage.SUMMARY_FILE)
        self.assertIn('vs_baseline', summary)
        self.assertTrue(summary['vs_baseline']['available'])
        self.assertEqual(summary['vs_baseline']['baseline_condition'], 'eyes closed')
        self.assertEqual(summary['vs_baseline']['current_source'], 'stable')
        self.assertIsNotNone(summary['vs_baseline']['peak_hz_delta'])
        self.assertEqual(set(summary['vs_baseline']['relative_pct_delta']), set(BANDS))

    def test_refresh_trend_skips_drawing_when_its_widget_is_gone(self):
        """A deleted trend widget skips the redraw (no RuntimeError); a live one draws.

        TEST ONLY: sip.delete removes the C++ object the way Qt does when a plot
        is torn down while the Python wrapper survives — the failure that crashed
        a save on such a window. The same data must plot normally while the
        widget is alive, and be skipped (silently) once it is gone.
        """
        save_test_only_session(self.root, session_summary(stable=test_only_stable_report()))
        self.window.refresh_trend()
        self.assertEqual(set(self.window.trend_curves), set(BANDS))
        main.sip.delete(self.window.trend_plot)   # TEST ONLY: the C++ widget disappears
        self.window.refresh_trend()               # must not raise RuntimeError
        self.assertEqual(self.window.trend_curves, {})

    def test_capture_without_a_passing_estimate_is_refused(self):
        self.window.analyze_now()
        with mock.patch.object(main.QMessageBox, 'warning') as warning:
            self.window.set_baseline_from_current()
        self.assertTrue(warning.called)
        self.assertFalse((self.root / KEY / baseline.BASELINE_FILE).exists())
        self.assertIsNone(self.window.baseline)

    def test_unusable_baseline_file_is_reported_and_never_crashes(self):
        folder = self.root / KEY
        folder.mkdir(parents=True)
        (folder / baseline.BASELINE_FILE).write_text('{ broken', encoding='utf-8')
        self.window.load_baseline()
        self.assertIsNone(self.window.baseline)
        self.assertTrue(self.window.baseline_error)
        self.assertIn('not usable', self.window.baseline_status.text())
        self.window.analyze_now()
        self.assertIn('vs baseline: ', self.window.baseline_delta_label.text())

    def test_replacing_the_baseline_updates_the_panel_condition(self):
        self.capture_baseline('eyes closed')
        self.window.baseline_condition.setCurrentText('eyes open')
        with mock.patch.object(main.QMessageBox, 'information'), \
                mock.patch.object(main.QMessageBox, 'warning'):
            self.window.set_baseline_from_current()
        self.assertIn('(eyes open)', self.window.baseline_status.text())
        self.assertEqual(baseline.load_baseline(self.root, NAME, PHONE)['record']['condition'],
                         'eyes open')


@unittest.skipIf(main is None, 'Qt/main.py unavailable')
class BaselineHistoryTests(unittest.TestCase):
    """History dialog: baseline mark, per-row delta line, capture from a session."""

    class FakeBaselineDialog(QDialog):
        """TEST ONLY: accepts immediately with a fixed condition."""
        condition_value = 'eyes open'

        def __init__(self, *args, **kwargs):
            super().__init__(None)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def condition(self):
            return type(self).condition_value

    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'subjects'
        self.first = save_test_only_session(
            self.root, session_summary(stable=test_only_stable_report()))
        self.second = save_test_only_session(
            self.root, session_summary(report=test_only_window_report()))

    def dialog(self):
        dialog = main.HistoryDialog(self.root)
        self.addCleanup(dialog.close)
        dialog.subject_list.setCurrentRow(0)
        return dialog

    def test_rows_show_deltas_and_mark_the_baseline_source_session(self):
        dialog = self.dialog()
        self.assertIn('No baseline set for this subject', dialog.baseline_label.text())
        record = baseline.build_baseline(
            test_only_stable_report(), 'eyes closed', subject_name=NAME, subject_phone=PHONE,
            channels=list(CHANNELS), fs_hz=250, source='stable',
            session_id=self.first.name, captured_from='saved_session')
        baseline.save_baseline(self.root, NAME, PHONE, record)
        dialog.baseline_label.setText('')
        dialog.show_sessions()
        texts = [dialog.session_list.item(row).text()
                 for row in range(dialog.session_list.count())]
        self.assertEqual(len(texts), 2)
        # list_sessions is newest-first, so derive the marked row from the dialog's
        # own session list instead of assuming a fixed order.
        source_row = next(row for row, session in enumerate(dialog.sessions)
                          if session['path'] == self.first)
        self.assertIn('★ baseline source', texts[source_row])
        for row, text in enumerate(texts):
            if row != source_row:
                self.assertNotIn('★ baseline source', text)
        for text in texts:
            self.assertIn('vs baseline: ', text)
        self.assertIn('Baseline: Alpha-dominant', dialog.baseline_label.text())
        self.assertIn('same condition', dialog.baseline_label.text())
        dialog.session_list.setCurrentRow(0)
        self.assertIn('BASELINE (predominant-wave profile of this subject)',
                      dialog.details.toPlainText())
        self.assertIn('same condition', dialog.details.toPlainText())

    def test_capture_from_a_selected_saved_session(self):
        dialog = self.dialog()
        stored = storage.read_json_object(self.second / storage.SUMMARY_FILE)
        stored['vs_baseline'] = {'available': True, 'reason': 'ok',
                                 'baseline_condition': 'eyes closed',
                                 'relative_pct_delta': {name: 0. for name in BANDS},
                                 'baseline_relative_pct': {name: 0. for name in BANDS},
                                 'current_relative_pct': {name: 0. for name in BANDS},
                                 'dominant_from': 'Alpha', 'dominant_to': 'Alpha',
                                 'dominant_changed': False, 'peak_hz_delta': 0.,
                                 'largest_change': {'band': 'Alpha', 'delta_pp': 0.},
                                 'baseline_set_at': '2026-09-28T10:00:00+00:00',
                                 'baseline_peak_hz': 10., 'current_peak_hz': 10.}
        (self.second / storage.SUMMARY_FILE).write_text(json.dumps(stored), encoding='utf-8')
        # list_sessions is newest-first: derive the row of self.second from the dialog.
        second_row = next(index for index, session in enumerate(dialog.sessions)
                          if session['path'] == self.second)
        dialog.session_list.setCurrentRow(second_row)
        with mock.patch.object(main, 'BaselineDialog', self.FakeBaselineDialog), \
                mock.patch.object(main.QMessageBox, 'information'), \
                mock.patch.object(main.QMessageBox, 'warning'):
            dialog.set_baseline_from_selected()
        record = baseline.load_baseline(self.root, NAME, PHONE)['record']
        self.assertEqual(record['condition'], 'eyes open')
        self.assertEqual(record['session_id'], self.second.name)
        self.assertEqual(record['captured_from'], 'saved_session')
        self.assertEqual(record['profile']['source'], 'five_second')
        self.assertEqual(record['device_label'], 'BrainBit Classic')
        texts = [dialog.session_list.item(row).text() for row in range(dialog.session_list.count())]
        self.assertTrue(any('★ baseline source' in text for text in texts))
        # the injected stored-at-save-time block lives on self.second (derived row).
        dialog.session_list.setCurrentRow(next(index for index, session in enumerate(dialog.sessions)
                                               if session['path'] == self.second))
        self.assertIn('Stored at save time:', dialog.details.toPlainText())

    def test_capture_from_a_session_without_a_passing_estimate_is_refused(self):
        empty = save_test_only_session(self.root, session_summary())
        dialog = self.dialog()
        for row, session in enumerate(dialog.sessions):
            if session['path'] == empty:
                dialog.session_list.setCurrentRow(row)
                break
        with mock.patch.object(main, 'BaselineDialog', self.FakeBaselineDialog), \
                mock.patch.object(main.QMessageBox, 'warning') as warning:
            dialog.set_baseline_from_selected()
        self.assertTrue(warning.called)
        self.assertFalse(baseline.load_baseline(self.root, NAME, PHONE)['present'])

    def test_missing_baseline_and_no_sessions_are_safe(self):
        empty_root = Path(self.tmp.name) / 'empty'
        dialog = main.HistoryDialog(empty_root)
        self.addCleanup(dialog.close)
        self.assertEqual(dialog.baseline_label.text(), 'No baseline for this subject.')
        with mock.patch.object(main.QMessageBox, 'warning'):
            dialog.set_baseline_from_selected()
        dialog.subject_list.setCurrentRow(0)


if __name__ == '__main__':
    unittest.main()
