"""TEST ONLY fixtures for the device catalogue. No BLE, no hardware, no Qt.

Everything here is a fake SensorInfo / EEGChannelInfo described by the SDK's own
enumerations; nothing in this file is emitted by a live headset.
"""
import unittest
from types import SimpleNamespace

from neurosdk.cmn_types import (EEGChannelId, EEGChannelMode, EEGChannelType,
                                SensorFamily, SensorGain, SensorSamplingFrequency)

import devices
import topomap


def channel_info(channel_id, num, name=''):
    """One SDK channel-table row (EEGChannelInfo shape)."""
    return SimpleNamespace(Id=channel_id, ChType=EEGChannelType.EEGChTypeSingleA1,
                           Name=name, Num=num)


# The SDK's own identifiers for the Dragon/NeuroEEG set: 21 scalp sites (the
# complete 10-20 set, midline Fpz/Fz/Cz/Pz/Oz included) plus three auxiliary
# identifiers that the manufacturer describes as poly channels (ECG/EMG/EOG).
DRAGON_TABLE = [
    channel_info(EEGChannelId.EEGChIdD1, 21, 'D1'),
    channel_info(EEGChannelId.EEGChIdFp1, 1),
    channel_info(EEGChannelId.EEGChIdF7, 2),
    channel_info(EEGChannelId.EEGChIdF3, 3),
    channel_info(EEGChannelId.EEGChIdFZ, 4),
    channel_info(EEGChannelId.EEGChIdFpZ, 5),
    channel_info(EEGChannelId.EEGChIdFp2, 6),
    channel_info(EEGChannelId.EEGChIdF8, 7),
    channel_info(EEGChannelId.EEGChIdF4, 8),
    channel_info(EEGChannelId.EEGChIdT3, 9),
    channel_info(EEGChannelId.EEGChIdC3, 10),
    channel_info(EEGChannelId.EEGChIdCZ, 11),
    channel_info(EEGChannelId.EEGChIdC4, 12),
    channel_info(EEGChannelId.EEGChIdT4, 13),
    channel_info(EEGChannelId.EEGChIdT5, 14),
    channel_info(EEGChannelId.EEGChIdP3, 15),
    channel_info(EEGChannelId.EEGChIdPZ, 16),
    channel_info(EEGChannelId.EEGChIdP4, 17),
    channel_info(EEGChannelId.EEGChIdT6, 18),
    channel_info(EEGChannelId.EEGChIdO1, 19),
    channel_info(EEGChannelId.EEGChIdOZ, 20),
    channel_info(EEGChannelId.EEGChIdO2, 22),
    channel_info(EEGChannelId.EEGChIdD2, 23, 'D2'),
    channel_info(EEGChannelId.EEGChIdD3, 24, 'D3'),
]

DRAGON_SCALP = ('Fp1', 'F7', 'F3', 'Fz', 'Fpz', 'Fp2', 'F8', 'F4', 'T3', 'C3',
                'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2')


class DeviceCatalogueTests(unittest.TestCase):
    def test_scan_covers_classic_black_2_and_dragon_families(self):
        families = set(devices.SCAN_FAMILIES)
        for family in (SensorFamily.LEBrainBit, SensorFamily.LEBrainBitBlack,
                       SensorFamily.LEBrainBit2, SensorFamily.LENeuroEEG):
            self.assertIn(family, families)
        self.assertEqual(len(devices.SCAN_FAMILIES), len(set(devices.SCAN_FAMILIES)))

    def test_family_mapping(self):
        self.assertIs(devices.spec_for_family(SensorFamily.LEBrainBit), devices.CLASSIC)
        self.assertIs(devices.spec_for_family(SensorFamily.LEBrainBitBlack), devices.CLASSIC)
        self.assertIs(devices.spec_for_family(SensorFamily.LENeuroEEG), devices.DRAGON)
        self.assertIs(devices.spec_for_family(SensorFamily.LEBrainBit2), devices.BRAINBIT2)
        self.assertIs(devices.spec_for_family(SensorFamily.LEBrainBitPro), devices.BRAINBIT2)
        self.assertIsNone(devices.spec_for_family(SensorFamily.LEHeadband))

    def test_brainbit_channel_map_is_the_verified_four(self):
        spec = devices.spec_for_family(SensorFamily.LEBrainBit)
        self.assertEqual(spec.channels, ('O1', 'O2', 'T3', 'T4'))
        self.assertEqual(spec.fs, 250)
        self.assertEqual(spec.transport, 'classic')
        self.assertFalse(spec.needs_amplifier_setup)
        self.assertEqual(devices.CLASSIC_CHANNELS, topomap.BRAINBIT_CHANNELS)
        for name in spec.channels:
            self.assertIn(name, topomap.ELECTRODE_POSITIONS)

    def test_dragon_spec_requires_amplifier_setup(self):
        spec = devices.spec_for_family(SensorFamily.LENeuroEEG)
        self.assertEqual(spec.transport, 'neuroeeg')
        self.assertTrue(spec.needs_amplifier_setup)
        self.assertTrue(spec.resistance)
        self.assertIn(spec.fs, devices.SUPPORTED_FS)

    def test_dragon_channel_map_uses_device_nums_not_logical_ids(self):
        names = devices.channel_names(DRAGON_TABLE)
        expected = tuple(sorted(DRAGON_TABLE, key=lambda item: item.Num))
        self.assertEqual(len(names), len(expected))
        for index, item in enumerate(expected):
            self.assertEqual(names[index], devices.channel_name(item.Id))
        self.assertEqual(names[0], 'Fp1')
        self.assertEqual(names[3], 'Fz')
        self.assertEqual(set(names) - set(DRAGON_SCALP), {'D1', 'D2', 'D3'})
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(set(devices.poly_channels(names)), {'D1', 'D2', 'D3'})
        self.assertEqual(set(devices.scalp_channels(names)), set(DRAGON_SCALP))

    def test_scalp_set_is_the_complete_10_20_layout(self):
        self.assertEqual(len(DRAGON_SCALP), 21)
        self.assertEqual(set(devices.DRAGON_DOCUMENTED_EEG_CHANNELS), set(DRAGON_SCALP))
        for name in DRAGON_SCALP:
            self.assertIn(name, topomap.ELECTRODE_POSITIONS)

    def test_documented_fallback_only_names_what_the_sdk_can_report(self):
        reportable = {devices.channel_name(member) for member in EEGChannelId}
        reportable.discard(None)
        for name in devices.DRAGON_DOCUMENTED_EEG_CHANNELS:
            self.assertIn(name, reportable,
                          f'{name} cannot be reported by this SDK; the fallback must stay honest')

    def test_dragon_channel_map_falls_back_to_the_documented_list(self):
        self.assertEqual(devices.channel_names(None), devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
        self.assertEqual(devices.channel_names([]), devices.DRAGON_DOCUMENTED_EEG_CHANNELS)
        self.assertEqual(len(devices.DRAGON_DOCUMENTED_EEG_CHANNELS), 21)
        # Every documented name has a schematic head position.
        for name in devices.DRAGON_DOCUMENTED_EEG_CHANNELS:
            self.assertIn(name, topomap.ELECTRODE_POSITIONS)

    def test_unknown_identifier_uses_device_text_and_never_guesses(self):
        table = [channel_info(0x7F, 1, 'X9'), channel_info(0x7F, 2, '')]
        self.assertEqual(devices.channel_names(table), ('X9',))

    def test_sdk_identifier_naming_is_standard_10_20_spelling(self):
        self.assertEqual(devices.channel_name(EEGChannelId.EEGChIdOZ), 'Oz')
        self.assertEqual(devices.channel_name(EEGChannelId.EEGChIdFpZ), 'Fpz')
        self.assertEqual(devices.channel_name(EEGChannelId.EEGChIdT3), 'T3')
        self.assertEqual(devices.channel_name(EEGChannelId.EEGChIdD3), 'D3')
        self.assertIsNone(devices.channel_name(EEGChannelId.EEGChIdUnknown))

    def test_describe_sensor_info_marks_family_and_support(self):
        dragon_info = SimpleNamespace(Address='AA:BB:CC:DD:EE:01', Name='DragonEEG',
                                      SerialNumber='DR1', RSSI=-55, SensModel=1,
                                      PairingRequired=True,
                                      SensFamily=SensorFamily.LENeuroEEG)
        described = devices.describe_sensor_info(dragon_info)
        self.assertEqual(described['device_key'], 'dragon_eeg')
        self.assertTrue(described['supported'])
        self.assertTrue(described['pairing_required'])
        self.assertEqual(described['channels'], list(devices.DRAGON_DOCUMENTED_EEG_CHANNELS))
        self.assertEqual(described['address'], 'AA:BB:CC:DD:EE:01')
        self.assertIn('amplifier', described['notes'].lower())

    def test_unknown_family_is_listed_but_not_driven_as_a_brainbit(self):
        info = SimpleNamespace(Address='11:22:33:44:55:66', Name='Mystery', SerialNumber='',
                               RSSI=-80, SensModel=0, PairingRequired=False, SensFamily=None)
        described = devices.describe_sensor_info(info)
        self.assertFalse(described['supported'])
        self.assertEqual(described['device_key'], 'unknown')
        self.assertEqual(described['channels'], [])
        # Only the caller's saved target address may supply a fallback spec.
        fallback = devices.describe_sensor_info(info, devices.CLASSIC)
        self.assertTrue(fallback['supported'])

    def test_amplifier_parameter_block_for_a_dragon(self):
        template = SimpleNamespace(ReferentResistMesureAllow=False,
                                   Frequency=SensorSamplingFrequency.FrequencyHz1000,
                                   ReferentMode='RefHeadTop', ChannelMode=[], ChannelGain=[],
                                   RespirationOn=False)
        param = devices.neuroeeg_amplifier_param(template, 24, fs=250)
        self.assertEqual(param.Frequency, SensorSamplingFrequency.FrequencyHz250)
        self.assertEqual(len(param.ChannelMode), 24)
        self.assertEqual(len(param.ChannelGain), 24)
        self.assertTrue(all(mode is EEGChannelMode.EEGChModeSignalResist
                            for mode in param.ChannelMode))
        self.assertTrue(all(gain is SensorGain.Gain6 for gain in param.ChannelGain))
        self.assertTrue(param.ReferentResistMesureAllow)
        # The device's own template is never mutated.
        self.assertFalse(template.ReferentResistMesureAllow)
        self.assertEqual(template.Frequency, SensorSamplingFrequency.FrequencyHz1000)

    def test_amplifier_parameter_validation(self):
        with self.assertRaises(ValueError):
            devices.neuroeeg_amplifier_param(None, 0)
        with self.assertRaises(ValueError):
            devices.neuroeeg_amplifier_param(None, 24, fs=333)
        with self.assertRaises(ValueError):
            devices.neuroeeg_amplifier_param(None, 24, gain='Gain6')
        with self.assertRaises(ValueError):
            devices.neuroeeg_amplifier_param(None, 24, mode=3)

    def test_amplifier_read_back_check_detects_the_zero_stream_causes(self):
        good = SimpleNamespace(Frequency=SensorSamplingFrequency.FrequencyHz250,
                               ChannelMode=[EEGChannelMode.EEGChModeSignalResist] * 4,
                               ChannelGain=[SensorGain.Gain6] * 4)
        self.assertEqual(devices.neuroeeg_amplifier_problems(good, 4, 250), [])
        wrong_rate = SimpleNamespace(Frequency=SensorSamplingFrequency.FrequencyHz1000,
                                     ChannelMode=[EEGChannelMode.EEGChModeSignalResist] * 4,
                                     ChannelGain=[SensorGain.Gain6] * 4)
        self.assertTrue(any('frequency' in problem
                            for problem in devices.neuroeeg_amplifier_problems(wrong_rate, 4, 250)))
        off = SimpleNamespace(Frequency=SensorSamplingFrequency.FrequencyHz250,
                              ChannelMode=[EEGChannelMode.EEGChModeOff] * 4,
                              ChannelGain=[SensorGain.Gain6] * 4)
        self.assertTrue(any('switched off' in problem
                            for problem in devices.neuroeeg_amplifier_problems(off, 4, 250)))
        short = SimpleNamespace(Frequency=SensorSamplingFrequency.FrequencyHz250,
                                ChannelMode=[EEGChannelMode.EEGChModeSignalResist] * 2,
                                ChannelGain=[SensorGain.Gain6] * 2)
        self.assertTrue(any('shorter' in problem
                            for problem in devices.neuroeeg_amplifier_problems(short, 4, 250)))
        self.assertTrue(devices.neuroeeg_amplifier_problems(None, 4, 250))

    def test_brainbit2_parameter_block_shape(self):
        param = devices.brainbit2_amplifier_param(None, 4)
        self.assertEqual(len(param.ChGain), 4)
        self.assertEqual(len(param.ChSignalMode), 4)
        self.assertEqual(len(param.ChResistUse), 4)
        self.assertTrue(all(param.ChResistUse))
        with self.assertRaises(ValueError):
            devices.brainbit2_amplifier_param(None, 0)


if __name__ == '__main__':
    unittest.main()
