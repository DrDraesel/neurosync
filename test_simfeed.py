"""Simulator feed tests (no Qt, no hardware): shape, values, labels."""
import sys
import unittest
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import devices  # noqa: E402
import simfeed  # noqa: E402


class TestSimulatorSignal(unittest.TestCase):
    def test_block_shape_and_labels(self):
        sim = simfeed.Simulator(devices.CLASSIC_CHANNELS, fs=250)
        block = sim.block(10)
        self.assertEqual(block.shape, (10, 4))
        self.assertTrue(np.isfinite(block).all())
        self.assertEqual(sim.generated, 10)

    def test_alpha_dominates_posterior_channel(self):
        sim = simfeed.Simulator(devices.CLASSIC_CHANNELS, fs=250)
        block = sim.block(250 * 8)
        from eeg_analysis import BANDS
        import eeg_analysis
        # crude DFT-based check: the 10 Hz component must dominate O1
        freqs = np.fft.rfftfreq(block.shape[0], 1 / 250)
        spectrum = np.abs(np.fft.rfft(block[:, 0] - block[:, 0].mean()))
        alpha_peak = spectrum[(freqs >= 8) & (freqs <= 12)].max()
        theta_peak = spectrum[(freqs >= 4) & (freqs < 8)].max()
        delta_peak = spectrum[(freqs >= 1) & (freqs < 4)].max()
        self.assertGreater(alpha_peak, theta_peak)
        self.assertGreater(alpha_peak, delta_peak)
        self.assertIn('Alpha', BANDS)

    def test_posterior_weight_higher_than_frontal(self):
        self.assertGreater(simfeed._channel_weight('O1'), simfeed._channel_weight('Fp1'))
        self.assertGreater(simfeed._channel_weight('P4'), simfeed._channel_weight('F3'))


class TestSimulatedClientMetadata(unittest.TestCase):
    def test_dragon_channels_cover_full_montage(self):
        channels = simfeed.channels_for('dragon')
        self.assertEqual(len(channels), len(devices.DRAGON_DOCUMENTED_EEG_CHANNELS))
        self.assertIn('Cz', channels)

    def test_device_info_is_labelled_simulated(self):
        info = simfeed.device_info_for('brainbit')
        self.assertTrue(info['simulated'])
        self.assertTrue(info['device_label'].startswith('SIMULATED'))
        self.assertEqual(info['channel_source'], 'built-in simulator (not a device)')
        self.assertEqual(tuple(info['channels']), tuple(devices.CLASSIC_CHANNELS))
        # keys consumed by main.handle_devices / the device combo
        self.assertIn('supported', info)
        self.assertTrue(info['supported'])
        self.assertIn('address', info)

    def test_contact_passes_the_app_gate(self):
        contact = simfeed.contact_for(devices.CLASSIC_CHANNELS)
        self.assertTrue(contact['simulated'])
        for value in contact['ohms'].values():
            self.assertGreater(value, 0)
            self.assertLessEqual(value, 1e6)
        self.assertIn('A1', contact['aux'])


if __name__ == '__main__':
    unittest.main()
