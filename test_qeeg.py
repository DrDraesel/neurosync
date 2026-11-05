"""Tests for qeeg.py: band powers, asymmetry, coherence, phase, grid, references.

TEST ONLY. Every signal below is a synthetic sine or seeded Gaussian noise; no
headset, no patient data and no normative database are involved. These tests
pin the engineering definitions in docs/QEEG-GRID-SPEC.md, not any clinical
claim.
"""
import json
import os
import re
import tempfile
import unittest

import numpy as np

import qeeg

FS = 250.0
N = 2500                    # 10 s at 250 Hz -> nperseg 1024, 3 Welch segments
CHANNELS = ('F3', 'F4', 'C3', 'C4')
BAND_MID = {name: (low + high) / 2.0 for name, (low, high) in qeeg.BANDS.items()}


def sine(hz, amplitude=10.0, n=N, fs=FS, phase=0.0):
    t = np.arange(n) / fs
    return amplitude * np.sin(2 * np.pi * hz * t + phase)


def stack(*signals):
    return np.column_stack(signals)


def metrics_for(signals, channels=CHANNELS, fs=FS):
    return qeeg.compute_metrics(stack(*signals), fs, channels)


class TestConstants(unittest.TestCase):
    def test_bands_match_the_contract(self):
        self.assertEqual(list(qeeg.BANDS), ['Delta', 'Theta', 'Alpha', 'Beta', 'HighBeta'])
        self.assertEqual(qeeg.BANDS['Delta'], (1, 4))
        self.assertEqual(qeeg.BANDS['Theta'], (4, 8))
        self.assertEqual(qeeg.BANDS['Alpha'], (8, 13))
        self.assertEqual(qeeg.BANDS['Beta'], (13, 30))
        self.assertEqual(qeeg.BANDS['HighBeta'], (30, 45))

    def test_rows_match_the_contract(self):
        self.assertEqual(qeeg.ROWS,
                         ('absolute', 'relative', 'asymmetry', 'coherence', 'phase'))


class TestBandPower(unittest.TestCase):
    def test_sine_lands_in_its_band_and_peak_is_within_10_percent(self):
        for band, hz in BAND_MID.items():
            with self.subTest(band=band):
                metrics = metrics_for([sine(hz)] * 4)
                absolute = {name: metrics['bands'][name]['absolute']['F3']
                            for name in qeeg.BANDS}
                self.assertEqual(max(absolute, key=absolute.get), band)
                peak = metrics['bands'][band]['peak_hz']['F3']
                self.assertIsNotNone(peak)
                self.assertLessEqual(abs(peak - hz) / hz, 0.10)
                # 10 uV sine -> A^2/2 = 50 uV^2 in its own band.
                self.assertAlmostEqual(absolute[band], 50.0, delta=1.0)

    def test_relative_power_sums_to_100_percent(self):
        mixed = (sine(2.5, 6) + sine(6, 8) + sine(10, 10)
                 + sine(20, 4) + sine(37.5, 3))
        metrics = metrics_for([mixed] * 4)
        for channel in CHANNELS:
            total = sum(metrics['bands'][band]['relative'][channel]
                        for band in qeeg.BANDS)
            self.assertAlmostEqual(total, 100.0, delta=0.5)

    def test_unmeasured_band_reports_none_peak(self):
        # Nyquist is 50 Hz at fs=100, so a 60-80 Hz band was never measured.
        bands = {'Delta': (1, 4), 'Above': (60, 80)}
        metrics = qeeg.compute_metrics(
            np.column_stack([sine(2.5, n=1000, fs=100.0)] * 3), 100.0,
            ('F3', 'F4', 'C3'), bands=bands)
        self.assertIsNone(metrics['bands']['Above']['peak_hz']['F3'])
        self.assertEqual(metrics['bands']['Above']['absolute']['F3'], 0.0)

    def test_metrics_shape_and_epochs(self):
        metrics = metrics_for([sine(10.5)] * 4)
        self.assertEqual(metrics['n_samples'], N)
        self.assertEqual(metrics['n_epochs'], 3)
        self.assertAlmostEqual(metrics['window_s'], 10.0)
        self.assertEqual(tuple(metrics['usable_channels']), CHANNELS)
        self.assertEqual(metrics['notes'], [])
        for band in qeeg.BANDS:
            self.assertEqual(set(metrics['bands'][band]),
                             {'absolute', 'relative', 'peak_hz'})


class TestAsymmetry(unittest.TestCase):
    def test_left_louder_is_positive(self):
        # 2x amplitude = 4x power -> (400-100)/250*100 = +120 %.
        metrics = metrics_for([sine(10, 20), sine(10, 10),
                               sine(10, 10), sine(10, 10)])
        value = metrics['asymmetry']['Alpha']['F3/F4']
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, 120.0, delta=1.0)
        self.assertAlmostEqual(metrics['asymmetry']['Alpha']['C3/C4'], 0.0, delta=1e-6)

    def test_only_homologous_pairs_are_reported(self):
        metrics = metrics_for([sine(10)] * 4)
        self.assertEqual(set(metrics['asymmetry']['Alpha']), {'F3/F4', 'C3/C4'})

    def test_pair_is_skipped_when_the_homolog_is_absent(self):
        metrics = qeeg.compute_metrics(
            stack(sine(10), sine(10), sine(10)), FS, ('O1', 'C3', 'C4'))
        self.assertTrue(all('O1/O2' not in pair
                            for pair in metrics['asymmetry']['Alpha']))


class TestCoherence(unittest.TestCase):
    def test_identical_channels_have_coherence_one(self):
        alpha = sine(10.5)
        noise = np.random.default_rng(3).normal(size=N)
        metrics = qeeg.compute_metrics(stack(alpha, alpha, noise), FS,
                                       ('F3', 'F4', 'C3'))
        self.assertGreater(metrics['pairs']['Alpha']['F3/F4']['coherence'], 0.999)
        self.assertGreater(metrics['coherence']['Alpha']['F3'], 0.5)

    def test_independent_noise_has_coherence_near_zero(self):
        rng = np.random.default_rng(11)
        n = 15000           # 60 s -> ~28 Welch averages, bias ~1/28 per bin
        noise_a = rng.normal(size=n)
        noise_b = rng.normal(size=n)
        alpha = 10 * np.sin(2 * np.pi * 10.5 * np.arange(n) / FS)
        metrics = qeeg.compute_metrics(stack(alpha, alpha, noise_a, noise_b),
                                       FS, ('F3', 'F4', 'C3', 'P3'))
        self.assertLess(metrics['pairs']['Alpha']['C3/P3']['coherence'], 0.15)
        self.assertGreater(metrics['pairs']['Alpha']['F3/F4']['coherence'], 0.999)

    def test_electrode_coherence_is_the_pair_mean(self):
        metrics = metrics_for([sine(10.5), sine(10.5, 5), sine(10.5, 2),
                               sine(20, 3)])
        for channel in CHANNELS:
            pair_values = [row['coherence']
                           for pair, row in metrics['pairs']['Alpha'].items()
                           if channel in pair.split('/')]
            self.assertTrue(pair_values)
            self.assertAlmostEqual(metrics['coherence']['Alpha'][channel],
                                   float(np.mean(pair_values)), places=9)


class TestPhase(unittest.TestCase):
    def test_delay_on_the_right_gives_positive_left_minus_right(self):
        left = sine(10, phase=0.0)
        right = sine(10, phase=-2 * np.pi * 10 * 0.010)   # 10 ms late
        metrics = qeeg.compute_metrics(stack(left, right, left), FS,
                                       ('C3', 'C4', 'P3'))
        # 360 deg * 10 Hz * 0.010 s = 36 deg, left leading right.
        self.assertAlmostEqual(metrics['pairs']['Alpha']['C3/C4']['phase_deg'],
                               36.0, delta=2.0)
        self.assertGreater(metrics['pairs']['Alpha']['C3/C4']['phase_deg'], 0.0)

    def test_delay_on_the_left_flips_the_sign(self):
        left = sine(10, phase=-2 * np.pi * 10 * 0.010)
        right = sine(10, phase=0.0)
        metrics = qeeg.compute_metrics(stack(left, right, right), FS,
                                       ('C3', 'C4', 'P3'))
        self.assertAlmostEqual(metrics['pairs']['Alpha']['C3/C4']['phase_deg'],
                               -36.0, delta=2.0)

    def test_electrode_phase_is_the_signed_pair_mean_and_abs_is_reported(self):
        left = sine(10)
        right = sine(10, phase=-2 * np.pi * 10 * 0.010)
        metrics = qeeg.compute_metrics(stack(left, right, left), FS,
                                       ('C3', 'C4', 'P3'))
        for channel in ('C3', 'C4', 'P3'):
            values = [row['phase_deg']
                      for pair, row in metrics['pairs']['Alpha'].items()
                      if channel in pair.split('/')]
            self.assertAlmostEqual(metrics['phase']['Alpha'][channel],
                                   float(np.mean(values)), places=9)
            self.assertAlmostEqual(metrics['phase_abs']['Alpha'][channel],
                                   float(np.mean(np.abs(values))), places=9)


class TestZScore(unittest.TestCase):
    def test_zero_or_missing_sd_is_none(self):
        self.assertIsNone(qeeg.zscore(1.0, 0.0, 0.0))
        self.assertIsNone(qeeg.zscore(1.0, 0.0, None))
        self.assertIsNone(qeeg.zscore(None, 0.0, 1.0))
        self.assertIsNone(qeeg.zscore(float('nan'), 0.0, 1.0))

    def test_plain_value(self):
        self.assertAlmostEqual(qeeg.zscore(12.0, 10.0, 2.0), 1.0)


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.metrics = metrics_for([sine(10, 10), sine(10, 20),
                                    sine(10, 30), sine(10, 40)])

    def test_reference_none_never_colours(self):
        grid = qeeg.grid(self.metrics)
        self.assertEqual(grid['reference']['kind'], 'none')
        self.assertEqual(grid['limits'], {'z': [-3.0, 3.0]})
        self.assertEqual(grid['channels'], list(CHANNELS))
        self.assertEqual(grid['rows_order'], list(qeeg.ROWS))
        coloured = 0
        values = 0
        for row in qeeg.ROWS:
            for band in qeeg.BANDS:
                for channel in CHANNELS:
                    cell = grid['rows'][row][band][channel]
                    self.assertIn('value', cell)
                    self.assertIn('z', cell)
                    self.assertIsNone(cell['z'])
                    coloured += cell['z'] is not None
                    values += cell['value'] is not None
        self.assertEqual(coloured, 0)
        self.assertGreater(values, 0)

    def test_session_reference_produces_zero_mean_z(self):
        reference = qeeg.reference_from_session(self.metrics)
        self.assertEqual(reference['kind'], 'session')
        self.assertEqual(reference['n'], len(CHANNELS))
        grid = qeeg.grid(self.metrics, reference)
        zs = [grid['rows']['absolute']['Alpha'][channel]['z'] for channel in CHANNELS]
        self.assertTrue(all(z is not None for z in zs))
        self.assertAlmostEqual(float(np.mean(zs)), 0.0, places=9)
        self.assertLess(zs[0], 0.0)   # quietest channel
        self.assertGreater(zs[-1], 0.0)

    def test_midline_channel_has_none_asymmetry_not_zero(self):
        metrics = qeeg.compute_metrics(stack(sine(10), sine(10), sine(10)),
                                       FS, ('C3', 'C4', 'Cz'))
        grid = qeeg.grid(metrics)
        cell = grid['rows']['asymmetry']['Alpha']['Cz']
        self.assertEqual(cell, {'value': None, 'z': None})
        self.assertIsNotNone(grid['rows']['absolute']['Alpha']['Cz']['value'])

    def test_channel_missing_from_reference_keeps_value_but_no_z(self):
        reference = qeeg.reference_from_session(self.metrics)
        other = qeeg.compute_metrics(
            stack(sine(10), sine(10), sine(10), sine(10)), FS,
            ('F3', 'F4', 'C3', 'Cz'))
        grid = qeeg.grid(other, reference)
        cell = grid['rows']['absolute']['Alpha']['Cz']
        self.assertIsNotNone(cell['value'])
        self.assertIsNone(cell['z'])
        self.assertTrue(any('no z-score' in note for note in grid['notes']))

    def test_constant_across_channels_has_no_z(self):
        metrics = metrics_for([sine(10.5)] * 4)       # SD across channels = 0
        grid = qeeg.grid(metrics, qeeg.reference_from_session(metrics))
        for channel in CHANNELS:
            self.assertIsNone(grid['rows']['absolute']['Alpha'][channel]['z'])

    def test_baseline_label_is_canonical(self):
        reference = qeeg.reference_from_baseline(self.metrics, '2026-01-01 eyes closed', 2)
        grid = qeeg.grid(self.metrics, reference)
        self.assertIn('saved baseline', grid['reference']['label'])
        self.assertIn('2026-01-01 eyes closed', grid['reference']['label'])
        self.assertIn('not a normative clinical database', grid['notes'][0].lower())


class TestGuards(unittest.TestCase):
    def test_fewer_than_three_channels_returns_notes_and_no_heads(self):
        metrics = qeeg.compute_metrics(stack(sine(10), sine(10)), FS, ('C3', 'C4'))
        self.assertEqual(metrics['usable_channels'], [])
        self.assertTrue(metrics['notes'])
        self.assertIn('3 channels', metrics['notes'][0])
        grid = qeeg.grid(metrics)
        self.assertEqual(grid['channels'], [])
        self.assertEqual(grid['rows']['absolute']['Delta'], {})
        self.assertEqual(len(grid['rows']), 5)
        self.assertTrue(any('no heads are drawn' in note for note in grid['notes']))

    def test_less_than_two_seconds_returns_notes(self):
        short = np.column_stack([sine(10, n=250)] * 3)
        metrics = qeeg.compute_metrics(short, FS, ('F3', 'F4', 'C3'))
        self.assertEqual(metrics['usable_channels'], [])
        self.assertTrue(any('2 s' in note for note in metrics['notes']))

    def test_flat_or_non_finite_channel_is_excluded_not_zero_filled(self):
        flat = np.column_stack([sine(10), np.zeros(N), sine(10), sine(10)])
        metrics = qeeg.compute_metrics(flat, FS, ('F3', 'F4', 'C3', 'C4'))
        self.assertNotIn('F4', metrics['usable_channels'])
        self.assertEqual(len(metrics['usable_channels']), 3)
        self.assertTrue(any('F4' in note for note in metrics['notes']))
        self.assertNotIn('F4', metrics['bands']['Alpha']['absolute'])

    def test_two_clean_channels_after_exclusion_gives_no_grid(self):
        half_dead = np.column_stack([sine(10), np.zeros(N), sine(10)])
        metrics = qeeg.compute_metrics(half_dead, FS, ('F3', 'F4', 'C3'))
        self.assertEqual(metrics['usable_channels'], [])
        self.assertTrue(any('2 clean' in note for note in metrics['notes']))


class TestColor(unittest.TestCase):
    def test_green_is_exactly_zero_and_ends_clamp(self):
        green = qeeg.color_for_z(0.0)
        self.assertEqual(green, '#1aa84f')
        self.assertEqual(qeeg.color_for_z(-3.0), qeeg.color_for_z(-99.0))
        self.assertEqual(qeeg.color_for_z(3.0), qeeg.color_for_z(99.0))
        self.assertNotEqual(qeeg.color_for_z(-3.0), green)
        self.assertNotEqual(qeeg.color_for_z(3.0), green)
        self.assertEqual(qeeg.color_for_z(1.5), '#e8c51a')

    def test_colours_are_hex_and_none_is_grey(self):
        for z in (-2.5, -0.1, 0.0, 0.2, 2.9):
            self.assertRegex(qeeg.color_for_z(z), r'^#[0-9a-f]{6}$')
        self.assertEqual(qeeg.color_for_z(None), '#b3bcc6')

    def test_limits_are_validated(self):
        with self.assertRaises(ValueError):
            qeeg.color_for_z(0.0, (3, -3))


class TestRatios(unittest.TestCase):
    def test_balanced_five_band_signal_gives_ratios_near_one(self):
        mixed = (sine(2.5) + sine(6) + sine(10) + sine(20) + sine(37.5))
        result = qeeg.ratios(metrics_for([mixed] * 4))
        self.assertEqual(set(result), {'theta_beta', 'alpha_theta', 'slow_fast'})
        self.assertAlmostEqual(result['theta_beta'], 1.0, delta=0.15)
        self.assertAlmostEqual(result['alpha_theta'], 1.0, delta=0.15)
        self.assertAlmostEqual(result['slow_fast'], 2.0 / 3.0, delta=0.1)

    def test_missing_bands_return_none(self):
        self.assertEqual(qeeg.ratios({'usable_channels': [], 'bands': {}}),
                         {'theta_beta': None, 'alpha_theta': None, 'slow_fast': None})


class TestReferenceIO(unittest.TestCase):
    def setUp(self):
        self.metrics = metrics_for([sine(10, 10), sine(10, 20),
                                    sine(10, 30), sine(10, 40)])
        self.directory = tempfile.mkdtemp(prefix='qeeg_ref_')

    def path(self, name):
        return os.path.join(self.directory, name)

    def test_round_trip_preserves_the_statistics(self):
        reference = qeeg.reference_from_session(self.metrics)
        saved = qeeg.save_reference(reference, self.path('session.json'))
        loaded = qeeg.load_reference(saved)
        self.assertEqual(loaded['schema'], 'lab.norm/1')
        self.assertEqual(loaded['kind'], 'session')
        self.assertEqual(loaded['n'], len(CHANNELS))
        self.assertEqual(loaded['mean'], reference['mean'])
        self.assertEqual(loaded['sd'], reference['sd'])
        self.assertEqual(loaded['age_band'], reference['age_band'])
        self.assertTrue(loaded['created_at'])

    def test_file_contains_the_explicit_schema(self):
        qeeg.save_reference(qeeg.reference_from_session(self.metrics),
                            self.path('session.json'))
        with open(self.path('session.json'), encoding='utf-8') as handle:
            raw = json.load(handle)
        self.assertEqual(set(raw), {'schema', 'kind', 'label', 'n', 'age_band',
                                    'bands', 'rows', 'mean', 'sd', 'created_at'})

    def test_load_refuses_a_file_without_n(self):
        with open(self.path('bad.json'), 'w', encoding='utf-8') as handle:
            json.dump({'schema': 'lab.norm/1', 'kind': 'session', 'label': 'x',
                       'mean': {}}, handle)
        with self.assertRaises(ValueError) as caught:
            qeeg.load_reference(self.path('bad.json'))
        self.assertIn("'n'", str(caught.exception))

    def test_load_refuses_a_file_without_kind(self):
        with open(self.path('bad.json'), 'w', encoding='utf-8') as handle:
            json.dump({'schema': 'lab.norm/1', 'n': 4, 'mean': {}}, handle)
        with self.assertRaises(ValueError):
            qeeg.load_reference(self.path('bad.json'))

    def test_norm_with_too_few_subjects_is_refused(self):
        reference = qeeg.reference_from_session(self.metrics)
        reference['kind'] = 'norm'
        reference['n'] = 5
        reference['age_band'] = '30-39'
        with self.assertRaises(ValueError):
            qeeg.save_reference(reference, self.path('norm.json'))

    def test_norm_label_uses_the_allowed_wording(self):
        reference = qeeg.reference_from_session(self.metrics)
        reference.update(kind='norm', n=20, age_band='30-39',
                         label='local clinic')
        grid = qeeg.grid(self.metrics, reference)
        self.assertIn('vs local norm n=20, age 30-39', grid['reference']['label'])
        self.assertNotIn('NeuroGuide', grid['reference']['label'])


if __name__ == '__main__':
    unittest.main()
