"""TEST ONLY synthetic EEG fixtures; never use these as device measurements."""
import json
import unittest

import numpy as np
from scipy import signal

import eeg_analysis as eeg

CHANNELS = ('O1', 'O2', 'T3', 'T4')
FREQUENCIES = {'Delta': 2.5, 'Theta': 6., 'Alpha': 10., 'Beta': 20., 'Gamma': 36.}


def test_only_sine(hz=10., amplitude=10., n=1250, fs=250):
    """TEST ONLY: four identical sine-wave channels in microvolts."""
    t = np.arange(n) / fs
    return np.repeat((amplitude * np.sin(2 * np.pi * hz * t))[:, None], 4, axis=1)


class EEGAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.contact = dict.fromkeys(CHANNELS, 500_000.)

    def analyze(self, samples=None, **kwargs):
        options = dict(contact_ohms=self.contact, contact_age_s=0., notch_hz=None)
        options.update(kwargs)
        return eeg.analyze_window(test_only_sine() if samples is None else samples, **options)

    def assert_suppressed(self, result):
        self.assertFalse(result['valid'])
        self.assertIsNone(result['dominant'])
        self.assertIsNone(result['dominant_pct'])
        self.assertIsNone(result['peak_hz'])
        self.assertTrue(result['reason'])
        self.assertNotIn('Largest measured band', result['conclusion'])
        for value in result['bands'].values():
            self.assertIsNone(value['power_uv2'])
            self.assertIsNone(value['relative_pct'])
        json.dumps(result, allow_nan=False)

    def test_public_contract_and_serialization(self):
        self.assertEqual(eeg.CHANNELS, CHANNELS)
        self.assertEqual(dict(eeg.BANDS), {'Delta': (1, 4), 'Theta': (4, 8),
                         'Alpha': (8, 13), 'Beta': (13, 30), 'Gamma': (30, 45)})
        self.assertEqual(list(eeg.COLORS), list(FREQUENCIES))
        self.assertEqual(len(set(eeg.COLORS.values())), 5)
        for color in eeg.COLORS.values():
            self.assertRegex(color, r'^#[0-9a-fA-F]{6}$')
        result = self.analyze()
        self.assertEqual(set(result), {'ready', 'valid', 'reason', 'samples', 'duration_s',
                                      'bands', 'channels', 'dominant', 'dominant_pct',
                                      'peak_hz', 'conclusion'})
        self.assertEqual(result['samples'], 1250)
        self.assertEqual(result['duration_s'], 5.)
        self.assertTrue(result['ready'])
        self.assertTrue(result['valid'])
        self.assertIn('descriptive only', result['conclusion'])
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_all_five_test_only_band_dominants(self):
        for band, hz in FREQUENCIES.items():
            with self.subTest(band=band):
                result = self.analyze(test_only_sine(hz))
                self.assertTrue(result['valid'], result['reason'])
                self.assertEqual(result['dominant'], band)
                self.assertGreater(result['dominant_pct'], 95.)
                self.assertAlmostEqual(result['peak_hz'], hz, delta=.5)
                for channel in result['channels'].values():
                    self.assertEqual(channel['dominant'], band)
                    self.assertEqual(channel['flags'], [])

    def test_calibrated_sine_power_is_amplitude_squared_over_two(self):
        result = self.analyze(test_only_sine(amplitude=12.))
        self.assertAlmostEqual(result['bands']['Alpha']['power_uv2'], 72., delta=.2)
        self.assertAlmostEqual(sum(b['relative_pct'] for b in result['bands'].values()), 100.)
        for channel in result['channels'].values():
            self.assertAlmostEqual(sum(channel['relative_pct'].values()), 100.)

    def test_microvolt_scaling_is_quadratic(self):
        a = self.analyze(test_only_sine(amplitude=3.))
        b = self.analyze(test_only_sine(amplitude=9.))
        self.assertAlmostEqual(b['bands']['Alpha']['power_uv2'] /
                               a['bands']['Alpha']['power_uv2'], 9.)

    def test_short_window_is_not_ready(self):
        result = self.analyze(test_only_sine(n=1249))
        self.assertFalse(result['ready'])
        self.assertEqual(result['samples'], 1249)
        self.assert_suppressed(result)

    def test_only_latest_five_seconds_are_analyzed(self):
        samples = np.concatenate([test_only_sine(6.), test_only_sine(10.)])
        result = self.analyze(samples)
        self.assertEqual(result['dominant'], 'Alpha')
        self.assertEqual(result['samples'], 1250)
        self.assertEqual(result['duration_s'], 5.)

    def test_sampling_frequency_controls_window_and_frequency_axis(self):
        result = self.analyze(test_only_sine(fs=256, n=1280), fs=256)
        self.assertTrue(result['valid'])
        self.assertEqual(result['samples'], 1280)
        self.assertAlmostEqual(result['peak_hz'], 10., delta=.3)

    def test_band_edges_partition_power_without_halfopen_loss(self):
        for hz in (4., 8., 13., 30.):
            with self.subTest(hz=hz):
                result = self.analyze(test_only_sine(hz, amplitude=10.))
                self.assertAlmostEqual(sum(b['power_uv2'] for b in result['bands'].values()),
                                       50., delta=.1)
                self.assertEqual(result['dominant'], 'mixed')

    def test_near_tie_is_mixed(self):
        samples = test_only_sine(10., 10.) + test_only_sine(20., 9.8)
        result = self.analyze(samples)
        self.assertTrue(result['valid'])
        self.assertEqual(result['dominant'], 'mixed')
        self.assertIn('mixed', result['conclusion'])
        self.assertTrue(all(c['dominant'] == 'mixed' for c in result['channels'].values()))

    def test_aggregate_uses_mean_absolute_power_not_mean_percentages(self):
        samples = test_only_sine(6., 3.)
        samples[:, 0] = test_only_sine(10., 12.)[:, 0]
        result = self.analyze(samples)
        self.assertEqual(result['dominant'], 'Alpha')
        self.assertEqual(result['channels']['O1']['dominant'], 'Alpha')
        self.assertEqual(result['channels']['O2']['dominant'], 'Theta')
        self.assertAlmostEqual(result['bands']['Alpha']['power_uv2'], 18., delta=.1)
        self.assertAlmostEqual(result['bands']['Theta']['power_uv2'], 3.375, delta=.1)

    def test_absent_or_stale_contact_blocks_aggregate(self):
        for options in ({'contact_ohms': None}, {'contact_age_s': None},
                        {'contact_age_s': 120.01}, {'contact_age_s': -1},
                        {'contact_age_s': float('nan')}):
            with self.subTest(options=options):
                self.assert_suppressed(self.analyze(**options))
        result = self.analyze(contact_ohms=None)
        self.assertIn('contact_unverified', result['channels']['O1']['flags'])
        self.assertTrue(result['channels']['O1']['powers_uv2'])

    def test_invalid_or_high_contact_in_one_channel_blocks_all(self):
        for value in (0., -1., float('nan'), float('inf'), 1_000_001., None):
            with self.subTest(value=value):
                contacts = dict(self.contact, T4=value)
                result = self.analyze(contact_ohms=contacts)
                self.assert_suppressed(result)
                self.assertTrue(result['channels']['O1']['valid'])
                self.assertFalse(result['channels']['T4']['valid'])
        result = self.analyze(contact_ohms={'O1': 1.})
        self.assert_suppressed(result)

    def test_contact_limits_are_inclusive_and_sequences_supported(self):
        result = self.analyze(contact_ohms=[1_000_000.] * 4, contact_age_s=120.)
        self.assertTrue(result['valid'], result['reason'])
        result = self.analyze(contact_age_s=dict.fromkeys(CHANNELS, 0.))
        self.assertTrue(result['valid'], result['reason'])

    def test_discontinuity_blocks_aggregate(self):
        result = self.analyze(continuous=False)
        self.assert_suppressed(result)
        self.assertIn('discontinuous', result['channels']['O1']['flags'])

    def test_missing_malformed_and_nonfinite_samples_are_safe(self):
        for samples in (None, [], np.empty((0, 4)), np.zeros((1250, 3)),
                        np.zeros(1250), [['not numeric']] * 1250):
            with self.subTest(kind=str(type(samples))):
                result = eeg.analyze_window(samples)
                self.assert_suppressed(result)
        for value in (float('nan'), float('inf'), -float('inf')):
            samples = test_only_sine()
            samples[30, 2] = value
            self.assert_suppressed(self.analyze(samples))
        samples = np.concatenate([test_only_sine(), test_only_sine()])
        samples[0, 0] = np.nan
        self.assert_suppressed(self.analyze(samples))

    def test_ragged_input_returns_suppressed_results_without_exception(self):
        samples = [[1., 2., 3., 4.], [1., 2.]]
        self.assert_suppressed(eeg.analyze_window(samples))
        self.assertEqual(eeg.band_traces(samples), dict.fromkeys(FREQUENCIES, []))

    def test_zero_and_low_amplitude_are_flatline(self):
        for amplitude in (0., .1):
            result = self.analyze(test_only_sine(amplitude=amplitude))
            self.assert_suppressed(result)
            self.assertIn('flatline', result['channels']['O1']['flags'])

    def test_peak_to_peak_artifact(self):
        result = self.analyze(test_only_sine(2.5, 260.))
        self.assert_suppressed(result)
        self.assertIn('peak_to_peak', result['channels']['O1']['flags'])

    def test_abrupt_step_artifact(self):
        samples = test_only_sine()
        samples[500:, 0] += 150.
        result = self.analyze(samples)
        self.assert_suppressed(result)
        self.assertIn('abrupt_step', result['channels']['O1']['flags'])

    def test_saturated_or_offset_signal(self):
        result = self.analyze(test_only_sine() + 10001.)
        self.assert_suppressed(result)
        self.assertIn('saturated_or_offset', result['channels']['O1']['flags'])

    def test_line_noise_is_checked_before_notch(self):
        for hz in (50., 60.):
            for notch in (None, 50., 60.):
                with self.subTest(hz=hz, notch=notch):
                    result = self.analyze(test_only_sine() + test_only_sine(hz, 20.),
                                          notch_hz=notch)
                    self.assert_suppressed(result)
                    self.assertIn('mains_contamination', result['channels']['O1']['flags'])

    def test_notch_accepts_only_off_50_or_60(self):
        for notch in (None, 0, 50, 60):
            self.assertTrue(self.analyze(notch_hz=notch)['valid'])
        with self.assertRaises(ValueError):
            self.analyze(notch_hz=55)

    def test_invalid_sampling_parameters_raise_value_error(self):
        for fs in (0, -250, np.nan, np.inf, 90):
            with self.subTest(fs=fs), self.assertRaises(ValueError):
                self.analyze(fs=fs)

    def test_analysis_does_not_mutate_input(self):
        samples = test_only_sine()
        original = samples.copy()
        self.analyze(samples, notch_hz=60)
        np.testing.assert_array_equal(samples, original)

    def test_display_filters_separate_all_bands_without_normalization(self):
        samples = sum(test_only_sine(hz, 2., n=5000) for hz in FREQUENCIES.values())
        original = samples.copy()
        traces = eeg.band_traces(samples, notch_hz=None)
        self.assertEqual(list(traces), list(FREQUENCIES))
        for band, hz in FREQUENCIES.items():
            curve = np.array(traces[band])
            self.assertEqual(curve.size, 5000)
            frequencies, psd = signal.welch(curve[1250:-1250], fs=250, nperseg=500)
            self.assertAlmostEqual(frequencies[np.argmax(psd)], hz, delta=.5)
            self.assertGreater(np.std(curve[1250:-1250]), .9)
            self.assertLess(np.std(curve[1250:-1250]), 2.)
            doubled = eeg.band_traces(samples * 2, notch_hz=None)[band]
            np.testing.assert_allclose(doubled, curve * 2, atol=1e-10)
        np.testing.assert_array_equal(samples, original)
        json.dumps(traces, allow_nan=False)

    def test_display_selects_channel_and_preserves_length(self):
        samples = test_only_sine(6., n=1250)
        samples[:, 2] = test_only_sine(10.)[:, 0]
        traces = eeg.band_traces(samples, channel=2, notch_hz=60)
        self.assertTrue(all(len(curve) == 1250 for curve in traces.values()))
        self.assertGreater(np.std(traces['Alpha'][250:-250]), 5.)
        self.assertLess(np.std(traces['Theta'][250:-250]), .2)

    def test_display_short_missing_nonfinite_return_empty_lists(self):
        bad = test_only_sine()
        bad[10, 0] = np.nan
        for samples in (None, [], test_only_sine(n=2), bad):
            with self.subTest(kind=str(type(samples))):
                result = eeg.band_traces(samples)
                self.assertEqual(result, dict.fromkeys(FREQUENCIES, []))
        with self.assertRaises(ValueError):
            eeg.band_traces(test_only_sine(), channel=4)


class StableEstimateTests(unittest.TestCase):
    """TEST ONLY fixtures for the multi-epoch stable estimate."""

    def setUp(self):
        self.contact = dict.fromkeys(CHANNELS, 500_000.)

    def analyze(self, samples=None, **kwargs):
        options = dict(contact_ohms=self.contact, contact_age_s=0., notch_hz=None)
        options.update(kwargs)
        return eeg.analyze_stable(test_only_sine(n=7500) if samples is None
                                  else samples, **options)

    def test_contract_and_clean_sine(self):
        result = self.analyze()
        self.assertEqual(set(result), {'valid', 'reason', 'epochs_total', 'epochs_used',
                                       'epoch_seconds', 'seconds_total', 'seconds_used',
                                       'bands', 'channels', 'dominant', 'dominant_pct',
                                       'peak_hz', 'conclusion'})
        self.assertTrue(result['valid'], result['reason'])
        self.assertEqual(result['epochs_total'], 15)
        self.assertEqual(result['epochs_used'], 15)
        self.assertEqual(result['seconds_total'], 30.)
        self.assertEqual(result['dominant'], 'Alpha')
        self.assertAlmostEqual(result['bands']['Alpha']['power_uv2'], 50., delta=.2)
        self.assertAlmostEqual(result['bands']['Alpha']['sem_uv2'], 0., places=6)
        self.assertAlmostEqual(result['peak_hz'], 10., delta=.5)
        self.assertIn('descriptive only', result['conclusion'])
        for channel in result['channels'].values():
            self.assertEqual(channel['dominant'], 'Alpha')
            self.assertEqual(channel['flags'], [])
            self.assertEqual(channel['epochs_used'], 15)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_artifact_epoch_is_excluded_not_spread(self):
        samples = test_only_sine(n=7500)
        samples[2500:3000, 0] += 150.
        result = self.analyze(samples)
        self.assertTrue(result['valid'], result['reason'])
        # The offset's entry and exit steps reject the two adjacent epochs.
        self.assertEqual(result['channels']['O1']['epochs_used'], 13)
        self.assertEqual(result['channels']['O1']['excluded'], {'abrupt_step': 2})
        self.assertEqual(result['channels']['O2']['epochs_used'], 15)
        self.assertEqual(result['epochs_used'], 13)
        self.assertAlmostEqual(result['bands']['Alpha']['power_uv2'], 50., delta=.2)

    def test_insufficient_clean_epochs_suppresses_aggregate(self):
        samples = test_only_sine(n=7500)
        samples[0:5000, 3] = test_only_sine(2.5, 260., n=5000)[:, 0]
        result = self.analyze(samples)
        self.assertFalse(result['valid'])
        self.assertIn('insufficient_clean_epochs', result['channels']['T4']['flags'])
        self.assertEqual(result['channels']['T4']['epochs_used'], 5)
        self.assertEqual(result['channels']['T4']['excluded'], {'peak_to_peak': 10})
        self.assertTrue(result['channels']['O1']['valid'])
        self.assertAlmostEqual(result['channels']['O1']['powers_uv2']['Alpha'], 50., delta=.2)
        self.assertIsNone(result['bands']['Alpha']['power_uv2'])
        self.assertIn('insufficient_clean_epochs', result['reason'])
        json.dumps(result, allow_nan=False)

    def test_contact_and_discontinuity_gate_the_estimate(self):
        result = self.analyze(contact_ohms=None)
        self.assertFalse(result['valid'])
        self.assertIn('contact_unverified', result['channels']['O1']['flags'])
        result = self.analyze(contact_age_s=180.)
        self.assertFalse(result['valid'])
        self.assertIn('contact_stale', result['channels']['O2']['flags'])
        result = self.analyze(continuous=False)
        self.assertFalse(result['valid'])
        self.assertIn('discontinuous', result['channels']['T3']['flags'])

    def test_short_buffer_and_malformed_inputs_are_safe(self):
        result = self.analyze(test_only_sine(n=2999))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'insufficient_samples')
        self.assertEqual(result['epochs_total'], 5)
        for samples in (None, [], np.zeros((7500, 3)), np.zeros(7500)):
            with self.subTest(kind=str(type(samples))):
                result = eeg.analyze_stable(samples)
                self.assertFalse(result['valid'])
                self.assertTrue(result['reason'])
                json.dumps(result, allow_nan=False)
        samples = test_only_sine(n=7500)
        samples[100, 1] = np.nan
        result = eeg.analyze_stable(samples)
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'nonfinite_input')

    def test_invalid_configuration_raises(self):
        for kwargs in ({'epoch_seconds': .5}, {'epoch_seconds': 0}, {'min_epochs': 1},
                       {'min_epochs': 99999}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.analyze(**kwargs)

    def test_sem_reflects_epoch_variability(self):
        samples = np.zeros((7500, 4))
        for epoch in range(15):
            amplitude = 10. if epoch % 2 == 0 else 8.
            samples[epoch * 500:(epoch + 1) * 500] = test_only_sine(10., amplitude, n=500)
        result = self.analyze(samples)
        self.assertTrue(result['valid'], result['reason'])
        self.assertAlmostEqual(result['bands']['Alpha']['power_uv2'], 41., delta=1.)
        self.assertGreater(result['bands']['Alpha']['sem_uv2'], .5)
        self.assertLess(result['bands']['Alpha']['sem_uv2'], 5.)

    def test_analysis_does_not_mutate_input(self):
        samples = test_only_sine(n=7500)
        original = samples.copy()
        self.analyze(samples, notch_hz=60)
        np.testing.assert_array_equal(samples, original)


if __name__ == '__main__':
    unittest.main()
