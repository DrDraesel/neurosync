"""Tests for norms.py: CSV import and local-norm accumulation.

TEST ONLY. Every recording below is a synthetic sine plus seeded noise written
through ``storage.save_session`` into a temporary folder; no headset, no patient
data and no licensed export are involved. These tests pin the contract in
``docs/QEEG-NORMS-SPEC.md`` section 1, not any clinical claim.
"""
import csv
import hashlib
import json
import pathlib
import statistics
import tempfile
import unittest

import numpy as np

import norms
import qeeg

FS = 250.0
N = 500                     # 2 s at 250 Hz: the minimum analysis window
CHANNELS = ('O1', 'O2', 'T3', 'T4')
HEADER = (['sample_index', 'relative_time_s', 'packet_number']
          + [f'raw_{name}_uV' for name in CHANNELS])

LONG_CSV = """channel,band,row,mean,sd,n
F3,Alpha,absolute,50.5,10.2,24
C3,Alpha,absolute,48.0,9.5,24
A1,Alpha,absolute,40.0,5.0,24
F3,Alpha,relative,32.1,4.2,24
"""

WIDE_CSV = """channel,band,absolute_mean,absolute_sd,relative_mean,relative_sd
F3,Alpha,50.5,10.2,32.1,4.2
F4,Beta,20.0,5.0,10.0,2.0
"""

Z_ONLY_CSV = """channel,band,row,z
F3,Alpha,absolute,2.1
C3,Alpha,absolute,1.4
"""


def signal(seed):
    """TEST ONLY: four deterministic 2 s traces with a peak per channel."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FS
    columns = [15.0 * np.sin(2 * np.pi * (8 + 2 * index) * t)
               + rng.normal(0.0, 1.5, N) for index in range(len(CHANNELS))]
    return np.column_stack(columns)


def raw_rows(matrix):
    return [[index, index / FS, index, *matrix[index].tolist()]
            for index in range(N)]


def summary_for(age=30, condition='eyes closed', simulated=False, valid=True):
    return {
        'valid': valid,
        'condition': condition,
        'age': age,
        'age_band': '26-35',
        'simulated': simulated,
        'fs_hz': FS,
        'channels': list(CHANNELS),
        'scalp_channels': list(CHANNELS),
        # A deliberately absurd cached band table: build_local_norm must
        # recompute from raw_*.csv and never read this.
        'bands': {'Alpha': {'absolute': {'O1': 999999.0}}},
    }


def write_subject(root, name, *, seeds, age=30, condition='eyes closed',
                  simulated=False, minute=0):
    """Write one subject's session folders by hand (summary + raw CSV).

    The folder shape matches ``storage.save_session`` exactly
    (``root/<subject_key>/<session>/summary.json`` + ``raw.csv``) without the
    atomic-rename step, so these tests do not depend on storage's temp-file
    behaviour or on Windows file locking.
    """
    key = f'{name}_nophone'
    for offset, seed in enumerate(seeds):
        folder = pathlib.Path(root) / key / f'session_{minute:04d}_{offset:02d}'
        folder.mkdir(parents=True, exist_ok=True)
        summary = summary_for(age=age, condition=condition, simulated=simulated)
        summary['session_id'] = folder.name
        (folder / 'summary.json').write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        with open(folder / 'raw.csv', 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(raw_rows(signal(seed)))


def value_of(seed, row='absolute', band='Alpha', site='O1'):
    metrics = qeeg.compute_metrics(signal(seed), FS, CHANNELS)
    return qeeg.grid(metrics)['rows'][row][band][site]['value']


class ImportCSVTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = pathlib.Path(self._tmp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding='utf-8')
        return path

    def test_long_import_builds_correct_n_mean_sd(self):
        path = self.write('export.csv', LONG_CSV)
        norm = norms.import_csv(path, source='neuroguide', age_band='26-35',
                                condition='eyes closed', device_scope='21ch')
        self.assertIsInstance(norm, norms.NormFile)
        self.assertEqual(norm.schema, 'lab.norm/1')
        self.assertEqual(norm.source, 'neuroguide')
        self.assertEqual(norm.n, 24)
        self.assertEqual(norm.age_band, '26-35')
        self.assertEqual(norm.condition, 'eyes_closed')
        self.assertEqual(norm.device_scope, '21ch')
        self.assertEqual(norm.rows['absolute']['Alpha']['F3'],
                         {'mean': 50.5, 'sd': 10.2})
        self.assertEqual(norm.rows['absolute']['Alpha']['C3'],
                         {'mean': 48.0, 'sd': 9.5})
        self.assertEqual(norm.rows['relative']['Alpha']['F3'],
                         {'mean': 32.1, 'sd': 4.2})
        self.assertNotIn('A1', norm.rows['absolute']['Alpha'])
        self.assertEqual(norm.provenance['shape'], 'long')
        self.assertEqual(norm.provenance['sha256'],
                         hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(any('A1' in note for note in norm.provenance['notes']))

    def test_z_only_import_is_refused_with_the_spec_message(self):
        path = self.write('z.csv', Z_ONLY_CSV)
        with self.assertRaises(ValueError) as caught:
            norms.import_csv(path, age_band='26-35', condition='eyes closed')
        self.assertEqual(str(caught.exception),
                         'z-only exports cannot seed a reference — export mean and SD')

    def test_wide_import_builds_both_rows(self):
        path = self.write('wide.csv', WIDE_CSV)
        norm = norms.import_csv(path, source='bsi', age_band='26-35',
                                condition='eyes open', device_scope='19ch')
        self.assertEqual(norm.source, 'bsi')
        self.assertEqual(norm.condition, 'eyes_open')
        self.assertEqual(norm.provenance['shape'], 'wide-mean-sd')
        self.assertEqual(norm.rows['absolute']['Alpha']['F3'],
                         {'mean': 50.5, 'sd': 10.2})
        self.assertEqual(norm.rows['relative']['Alpha']['F3'],
                         {'mean': 32.1, 'sd': 4.2})
        self.assertEqual(norm.rows['absolute']['Beta']['F4'],
                         {'mean': 20.0, 'sd': 5.0})
        # No n column: n stays 0 and the provenance says the count was absent.
        self.assertEqual(norm.n, 0)
        self.assertTrue(any('n=0' in note for note in norm.provenance['notes']))

    def test_pairs_survive_and_unknown_rows_are_dropped(self):
        path = self.write('pairs.csv',
                          'channel,band,row,mean,sd\n'
                          'F3/F4,Alpha,coherence,0.62,0.08\n'
                          'F3-F4,Alpha,asymmetry,4.5,2.0\n'
                          'F3,Alpha,peak_hz,10.0,1.0\n')
        norm = norms.import_csv(path, age_band='26-35', condition='eyes closed')
        self.assertEqual(norm.rows['coherence']['Alpha']['F3/F4'],
                         {'mean': 0.62, 'sd': 0.08})
        self.assertEqual(norm.rows['asymmetry']['Alpha']['F3/F4'],
                         {'mean': 4.5, 'sd': 2.0})
        self.assertNotIn('peak_hz', norm.rows)
        self.assertTrue(any('peak_hz' in note
                            for note in norm.provenance['notes']))

    def test_missing_age_band_is_refused(self):
        path = self.write('export.csv', LONG_CSV)
        with self.assertRaises(ValueError):
            norms.import_csv(path, condition='eyes closed')

    def test_unrecognised_header_is_refused(self):
        path = self.write('odd.csv', 'channel,band,foo\nF3,Alpha,1\n')
        with self.assertRaises(ValueError):
            norms.import_csv(path, age_band='26-35', condition='eyes closed')


class LocalBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls._tmp.name)
        cls.root20 = base / 'twenty'
        cls.root19 = base / 'nineteen'
        cls.root21 = base / 'with_simulated'

        # 20 subjects; subject00 has a second session in the same condition.
        write_subject(cls.root20, 'subject00', seeds=[0, 100], minute=0)
        for index in range(1, 20):
            write_subject(cls.root20, f'subject{index:02d}', seeds=[index],
                          minute=index * 3)

        for index in range(19):
            write_subject(cls.root19, f'subject{index:02d}', seeds=[index],
                          minute=index * 3)

        for index in range(20):
            write_subject(cls.root21, f'subject{index:02d}', seeds=[index],
                          minute=index * 3)
        write_subject(cls.root21, 'simulated00', seeds=[500], simulated=True,
                      minute=200)

        cls.norm20, cls.reason20 = norms.build_local_norm(
            cls.root20, min_n=20, condition='eyes closed', age_band='26-35',
            device_scope='4ch')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_nineteen_subjects_are_refused_with_a_reason(self):
        norm, reason = norms.build_local_norm(
            self.root19, min_n=20, condition='eyes closed', age_band='26-35',
            device_scope='4ch')
        self.assertIsNone(norm)
        self.assertIn('19', reason)
        self.assertIn('20', reason)
        self.assertIn('at least 20', reason)

    def test_twenty_subjects_build_a_local_norm(self):
        self.assertEqual(self.reason20, '')
        self.assertIsNotNone(self.norm20)
        norm = self.norm20
        self.assertEqual(norm.source, 'local')
        self.assertEqual(norm.n, 20)
        self.assertEqual(norm.device_scope, '4ch')
        self.assertEqual(len(norm.provenance['subjects']), 20)
        self.assertTrue(norm.provenance['qeeg_version'].startswith('qeeg.py@'))
        self.assertIn('absolute', norm.rows)
        cell = norm.rows['absolute']['Alpha']['O1']
        self.assertGreater(cell['mean'], 0.0)
        self.assertGreaterEqual(cell['sd'], 0.0)

    def test_two_sessions_of_one_subject_count_once(self):
        norm = self.norm20
        self.assertEqual(norm.n, 20)
        entry = next(item for item in norm.provenance['subjects']
                     if item['subject'] == 'subject00_nophone')
        self.assertEqual(len(entry['sessions']), 2)
        self.assertEqual(norm.n, len(norm.provenance['subjects']))

    def test_cached_band_numbers_are_never_trusted(self):
        # Every summary claims Alpha/O1 = 999999 uV2. The built norm must match
        # a fresh qeeg computation from the raw CSVs instead.
        subject_values = [statistics.fmean([value_of(0), value_of(100)])]
        subject_values += [value_of(index) for index in range(1, 20)]
        expected_mean = statistics.fmean(subject_values)
        expected_sd = statistics.stdev(subject_values)
        cell = self.norm20.rows['absolute']['Alpha']['O1']
        self.assertAlmostEqual(cell['mean'], expected_mean, places=6)
        self.assertAlmostEqual(cell['sd'], expected_sd, places=6)
        self.assertLess(cell['mean'], 10000.0)

    def test_simulated_sessions_are_excluded(self):
        norm, reason = norms.build_local_norm(
            self.root21, min_n=20, condition='eyes closed', age_band='26-35',
            device_scope='4ch')
        self.assertEqual(reason, '')
        self.assertIsNotNone(norm)
        self.assertEqual(norm.n, 20)
        names = [entry['subject'] for entry in norm.provenance['subjects']]
        self.assertNotIn('simulated00_nophone', names)

    def test_device_scope_mismatch_is_reported_not_silently_accepted(self):
        norm, reason = norms.build_local_norm(
            self.root20, min_n=20, condition='eyes closed', age_band='26-35',
            device_scope='21ch')
        self.assertIsNone(norm)
        self.assertIn('device scope mismatch', reason)

    def test_condition_and_age_band_are_required(self):
        for kwargs in ({'age_band': '26-35'}, {'condition': 'eyes closed'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                norms.build_local_norm(self.root20, min_n=20, **kwargs)

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'norm.json'
            written = norms.save(self.norm20, path)
            self.assertEqual(written, str(path))
            loaded = norms.load(path)
        self.assertIsInstance(loaded, norms.NormFile)
        self.assertEqual(loaded.n, 20)
        self.assertEqual(len(loaded.provenance['subjects']), 20)
        self.assertEqual(loaded.rows, self.norm20.rows)
        self.assertEqual(loaded.condition, 'eyes_closed')

    def test_to_reference_scores_a_grid(self):
        reference = norms.to_reference(self.norm20)
        self.assertEqual(reference['kind'], 'norm')
        self.assertEqual(reference['n'], 20)
        metrics = qeeg.compute_metrics(signal(0), FS, CHANNELS)
        scored = qeeg.grid(metrics, reference)
        self.assertIsNotNone(scored['rows']['absolute']['Alpha']['O1']['z'])

    def test_scope_warning(self):
        norm = norms.NormFile(
            source='neuroguide', label='export', n=24, age_band='26-35',
            condition='eyes_closed', device_scope='21ch',
            rows={'absolute': {'Alpha': {'F3': {'mean': 1.0, 'sd': 0.5}}}})
        warning = norms.scope_warning(norm, CHANNELS)
        self.assertIn('mismatch', warning)
        self.assertIn('21ch', warning)
        self.assertIn('4ch', warning)
        self.assertEqual(norms.scope_warning(norm, 21), '')
        self.assertEqual(norms.scope_warning(norm, 4), warning)
        norm.device_scope = 'any'
        self.assertEqual(norms.scope_warning(norm, 4), '')
        norm.device_scope = '4ch'
        self.assertEqual(norms.scope_warning(norm, 4), '')

    def test_footer_line(self):
        line = norms.reference_line(self.norm20)
        self.assertEqual(
            line, 'reference: local · n=20 · age 26-35 · eyes_closed')
        self.assertIn('device scope mismatch',
                      norms.footer_text(self.norm20, 21))

    def generic_payload(self):
        return {
            'schema': 'lab.norm/1', 'source': 'generic', 'label': 'demo',
            'n': 24, 'age_band': '26-35', 'condition': 'eyes_closed',
            'device_scope': '19ch', 'created_at': '2026-01-01T00:00:00+00:00',
            'rows': {'absolute': {'Alpha': {'F3': {'mean': 1.0, 'sd': 0.5}}}},
            'provenance': {'notes': []},
        }

    def test_load_refuses_local_norm_without_subjects(self):
        payload = self.generic_payload()
        payload['source'] = 'local'
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'bad.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError) as caught:
                norms.load(path)
        self.assertIn('subjects', str(caught.exception))

    def test_load_refuses_missing_required_fields(self):
        for field in ('n', 'age_band', 'source'):
            payload = self.generic_payload()
            del payload[field]
            with self.subTest(field=field), \
                    tempfile.TemporaryDirectory() as folder:
                path = pathlib.Path(folder) / 'bad.json'
                path.write_text(json.dumps(payload), encoding='utf-8')
                with self.assertRaises(ValueError):
                    norms.load(path)

    def test_load_refuses_wrong_schema(self):
        payload = self.generic_payload()
        payload['schema'] = 'lab.norm/2'
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'bad.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                norms.load(path)


if __name__ == '__main__':
    unittest.main()
