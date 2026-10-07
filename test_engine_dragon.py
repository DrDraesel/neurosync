"""TEST ONLY fixtures for the unified transport. No BLE, no hardware, no Qt app.

The fake scanner and fake DragonEEG sensor below only implement the parts of the
pyneurosdk2 contract this application uses; nothing here is emitted by a live
headset, and none of it is reachable from the production acquisition path.
"""
import unittest
from types import SimpleNamespace

from neurosdk.cmn_types import (EEGChannelMode, EEGChannelType, EEGChannelId,
                                EEGRefMode, SensorCommand, SensorFamily,
                                SensorGain, SensorSamplingFrequency, SensorState)

import devices
import engine
from test_devices import DRAGON_TABLE


class AmplifierParamFixture:
    """Mutable stand-in for NeuroEEGAmplifierParam."""

    def __init__(self, channel_count=24, frequency=SensorSamplingFrequency.FrequencyHz250,
                 mode=EEGChannelMode.EEGChModeSignal):
        self.ReferentResistMesureAllow = False
        self.Frequency = frequency
        self.ReferentMode = EEGRefMode.RefHeadTop
        self.ChannelMode = [mode] * channel_count
        self.ChannelGain = [SensorGain.Gain6] * channel_count
        self.RespirationOn = False


class DragonSensorFixture:
    """NeuroEEGSensor-shaped fake: amplifier block, channel table, packets."""
    name = 'DragonEEG'
    serial_number = 'DRAGON-TEST'
    batt_power = 88
    sampling_frequency = SensorSamplingFrequency.FrequencyHz250
    state = SensorState.StateInRange
    signalDataReceived = None
    resistDataReceived = None

    def __init__(self, table=None, frames=3, fail_readback=False, ohm_base=120_000.,
                 sample_volts=1e-6):
        self.table = list(DRAGON_TABLE if table is None else table)
        self.channels_count = len(self.table)
        self.supported_channels = self.table
        self.frames = frames
        self.fail_readback = fail_readback
        self.ohm_base = ohm_base
        self.sample_volts = sample_volts
        self.events = []
        self.disconnected = False
        self._param = None
        self.client = None

    # --- SDK-shaped members ------------------------------------------------
    @property
    def amplifier_param(self):
        if self._param is None:
            return AmplifierParamFixture(self.channels_count)
        return self._param

    @amplifier_param.setter
    def amplifier_param(self, param):
        self.events.append(('write_amplifier_param', len(param.ChannelMode)))
        if self.fail_readback:
            self._param = AmplifierParamFixture(self.channels_count,
                                                frequency=SensorSamplingFrequency.FrequencyHz1000,
                                                mode=EEGChannelMode.EEGChModeOff)
        else:
            self._param = param

    def exec_command(self, command):
        self.events.append(('command', command))
        if command == SensorCommand.StartSignal and self.signalDataReceived is not None:
            samples = []
            for frame in range(self.frames):
                samples.extend([self.sample_volts * (index + 1) for index in range(self.channels_count)])
            self.signalDataReceived(self, [SimpleNamespace(PackNum=1, Marker=0, Samples=samples)])
            if self.client is not None:
                self.client._running = False
        if command == SensorCommand.StartResist and self.resistDataReceived is not None:
            self.resistDataReceived(self, [SimpleNamespace(
                PackNum=1, A1=self.ohm_base, A2=self.ohm_base + 1000.,
                Bias=self.ohm_base + 2000.,
                Values=[self.ohm_base + 1000. * index for index in range(self.channels_count)])])

    def disconnect(self):
        self.events.append(('disconnect', None))
        self.disconnected = True

    def event_names(self):
        return [name for name, _ in self.events]

    def commands(self):
        return [value for name, value in self.events if name == 'command']


class ScannerFixture:
    """Scanner that records its family filters and hands out SensorInfo rows."""
    last_filters = None

    def __init__(self, sensors, sensor=None):
        self._sensors = list(sensors)
        self._sensor = sensor
        self.created = []
        self.stopped = False
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def sensors(self):
        return list(self._sensors)

    def create_sensor(self, info):
        self.created.append(info)
        if self._sensor is None:
            raise RuntimeError('TEST fixture has no sensor instance')
        return self._sensor


def dragon_info(address='AA:BB:CC:DD:EE:0A', pairing=True):
    return SimpleNamespace(SensFamily=SensorFamily.LENeuroEEG, SensModel=1,
                           Name='DragonEEG', Address=address, SerialNumber='DRAGON-TEST',
                           PairingRequired=pairing, RSSI=-52)


class ChannelPacketTrackerTests(unittest.TestCase):
    def packets(self, frames, channel_count=24, packnum=1):
        return [SimpleNamespace(PackNum=packnum, Marker=0,
                                Samples=[1e-6 * (index + 1)
                                         for index in range(frames * channel_count)])]

    def test_interleaved_frames_become_one_row_per_frame_in_microvolts(self):
        tracker = engine.ChannelPacketTracker(4)
        result = tracker.decode([SimpleNamespace(PackNum=7, Marker=0,
                                                 Samples=[1e-6, 2e-6, 3e-6, 4e-6,
                                                          5e-6, 6e-6, 7e-6, 8e-6])])
        self.assertFalse(result['gap'])
        self.assertEqual(result['samples_uv'], [[1., 2., 3., 4.], [5., 6., 7., 8.]])
        self.assertEqual(result['packnums'], [7, 7])
        self.assertEqual(tracker.gap_events, 0)

    def test_ragged_or_nonfinite_packet_is_a_discontinuity(self):
        tracker = engine.ChannelPacketTracker(4)
        ragged = tracker.decode([SimpleNamespace(PackNum=1, Marker=0,
                                                 Samples=[1e-6] * 6)])
        self.assertTrue(ragged['gap'])
        self.assertEqual(ragged['samples_uv'], [])
        broken = tracker.decode([SimpleNamespace(PackNum=2, Marker=0,
                                                 Samples=[1e-6, float('nan'), 3e-6, 4e-6])])
        self.assertTrue(broken['gap'])
        self.assertEqual(tracker.gap_events, 2)

    def test_counter_anomalies_are_counted_and_repeats_break_the_window(self):
        tracker = engine.ChannelPacketTracker(2, max_repeats=2)
        for packnum in (5, 5, 5):
            result = tracker.decode([SimpleNamespace(PackNum=packnum, Marker=0,
                                                     Samples=[1e-6, 2e-6])])
        self.assertEqual(tracker.counter_anomalies, 2)
        self.assertTrue(result['gap'])
        self.assertEqual(tracker.gap_events, 1)

    def test_channel_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            engine.ChannelPacketTracker(0)


class DragonConnectionTests(unittest.TestCase):
    def make_client(self, scanner, info, **kwargs):
        options = dict(scanner_factory=lambda filters: scanner, scan_seconds=.01,
                       target_address=info.Address, auto_contact=False)
        options.update(kwargs)
        return engine.EEGClient(**options)

    def test_scan_requests_both_brainbit_and_dragon_families(self):
        sensor = DragonSensorFixture()
        scanner = ScannerFixture([dragon_info()], sensor)
        client = self.make_client(scanner, dragon_info())
        seen = []
        scanner_factory = lambda filters: (seen.append(list(filters)) or scanner)
        client.scanner_factory = scanner_factory
        client.run()
        self.assertEqual(len(seen), 1)
        families = seen[0]
        self.assertIn(SensorFamily.LEBrainBit, families)
        self.assertIn(SensorFamily.LENeuroEEG, families)
        self.assertIn(SensorFamily.LEBrainBitBlack, families)
        self.assertTrue(scanner.stopped)

    def test_dragon_amplifier_is_configured_before_start_signal(self):
        sensor = DragonSensorFixture()
        scanner = ScannerFixture([dragon_info()], sensor)
        client = self.make_client(scanner, dragon_info())
        sensor.client = client
        rows, infos = [], []
        client.data_received.connect(rows.append)
        client.device_info.connect(infos.append)
        client.run()
        names = sensor.event_names()
        self.assertIn('write_amplifier_param', names)
        write_index = names.index('write_amplifier_param')
        self.assertEqual(sensor.commands()[0], SensorCommand.StartSignal)
        self.assertLess(write_index, names.index('command'))
        self.assertEqual(sensor.commands(), [SensorCommand.StartSignal, SensorCommand.StopSignal])
        self.assertEqual(names[-1], 'disconnect')
        self.assertTrue(sensor.disconnected)
        self.assertIsNone(sensor.signalDataReceived)
        self.assertIsNone(sensor.resistDataReceived)

    def test_dragon_stream_decodes_every_channel_and_row(self):
        sensor = DragonSensorFixture(frames=3)
        scanner = ScannerFixture([dragon_info()], sensor)
        client = self.make_client(scanner, dragon_info())
        sensor.client = client
        rows, infos, batches = [], [], []
        client.data_received.connect(rows.append)
        client.device_info.connect(infos.append)
        client.batch_received.connect(batches.append)
        client.run()
        # Same ordering the app derives: the device's own Num field.
        names = [devices.channel_name(item.Id)
                 for item in sorted(sensor.table, key=lambda entry: entry.Num)]
        # One row per frame, every channel present, volts converted once.
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(rows[0]), 24)
        self.assertAlmostEqual(rows[0][0], 1., places=9)
        self.assertAlmostEqual(rows[0][23], 24., places=9)
        self.assertAlmostEqual(rows[1][0], 1., places=9)
        self.assertEqual(batches[0]['sample_count'], 3)
        self.assertEqual(batches[0]['channels'], names)
        self.assertEqual(client.channels, tuple(names))
        self.assertEqual(client.channel_source, 'read from NeuroEEGSensor.supported_channels')

    def test_device_info_reports_dragon_configuration_and_channels(self):
        sensor = DragonSensorFixture()
        scanner = ScannerFixture([dragon_info()], sensor)
        client = self.make_client(scanner, dragon_info())
        sensor.client = client
        infos = []
        client.device_info.connect(infos.append)
        client.run()
        info = infos[0]
        self.assertEqual(info['device_key'], 'dragon_eeg')
        self.assertEqual(info['fs'], 250)
        self.assertEqual(len(info['channels']), 24)
        self.assertEqual(len(info['scalp_channels']), 21)
        self.assertEqual(len(info['poly_channels']), 3)
        self.assertEqual(info['pairing_required'], True)
        self.assertEqual(info['configuration']['transport'], 'neuroeeg')
        self.assertIn('verified', info['configuration']['amplifier'])

    def test_amplifier_readback_failure_aborts_before_streaming(self):
        sensor = DragonSensorFixture(fail_readback=True)
        scanner = ScannerFixture([dragon_info()], sensor)
        client = self.make_client(scanner, dragon_info())
        sensor.client = client
        status = []
        client.status_changed.connect(status.append)
        client.run()
        self.assertNotIn(SensorCommand.StartSignal, sensor.commands())
        self.assertTrue(any('amplifier configuration failed' in text.lower() for text in status))

    def test_no_devices_found_is_reported_without_crashing(self):
        scanner = ScannerFixture([], None)
        client = self.make_client(scanner, dragon_info())
        status = []
        client.status_changed.connect(status.append)
        client.run()
        self.assertEqual(scanner.created, [])
        self.assertTrue(any('no devices found' in text.lower() for text in status))
        self.assertTrue(scanner.stopped)

    def test_dragon_resistance_reports_ohms_per_channel_with_references(self):
        sensor = DragonSensorFixture()
        client = engine.EEGClient(contact_seconds=0.05)
        client.device = sensor
        client.device_spec = devices.DRAGON
        client.role = 'neuroeeg'
        client.channels = tuple(devices.channel_names(sensor.table))
        client.mode = 'signal'
        readings = []
        client.contact_received.connect(readings.append)
        client._measure_contact(resume=True)
        self.assertEqual(sensor.commands(), [SensorCommand.StopSignal, SensorCommand.StartResist,
                                             SensorCommand.StopResist, SensorCommand.StartSignal])
        contact = readings[-1]
        self.assertEqual(len(contact['ohms']), len(client.channels))
        self.assertEqual(contact['ohms']['Fp1'], 120_000.)
        self.assertEqual(contact['ohms']['F7'], 121_000.)
        self.assertEqual(contact['aux'], {'A1': 120_000., 'A2': 121_000., 'Bias': 122_000.})
        self.assertEqual(contact['device_key'], 'dragon_eeg')
        self.assertIsNone(sensor.resistDataReceived)

    def test_nonfinite_resistance_is_reported_as_unavailable(self):
        sensor = DragonSensorFixture()
        client = engine.EEGClient(contact_seconds=0.05)
        client.device = sensor
        client.device_spec = devices.DRAGON
        client.role = 'neuroeeg'
        client.channels = ('O1', 'O2', 'T3', 'T4')
        readings = []
        client.contact_received.connect(readings.append)
        commands = []

        def exec_command(command):
            commands.append(command)
            if command == SensorCommand.StartResist:
                # One bad packet: NaN/zero/negative readings, one usable value.
                sensor.resistDataReceived(sensor, [SimpleNamespace(
                    PackNum=1, A1=float('nan'), A2=0., Bias=float('nan'),
                    Values=[float('nan'), 0., -5., 12_000.])])

        sensor.exec_command = exec_command
        client._measure_contact(resume=False)
        self.assertEqual(commands, [SensorCommand.StartResist, SensorCommand.StopResist])
        contact = readings[-1]
        self.assertIsNone(contact['ohms']['O1'])
        self.assertIsNone(contact['ohms']['O2'])
        self.assertIsNone(contact['ohms']['T3'])
        self.assertEqual(contact['ohms']['T4'], 12_000.)
        self.assertIsNone(contact['aux']['A1'])


class DeviceSelectionTests(unittest.TestCase):
    def describe(self, infos):
        client = engine.EEGClient()
        return client, client._describe(infos)

    def test_descriptions_keep_both_device_types(self):
        infos = {
            'DA:BB:A3:A0:73:4E': SimpleNamespace(SensFamily=SensorFamily.LEBrainBit,
                                                 SensModel=0, Name='BrainBit',
                                                 Address='DA:BB:A3:A0:73:4E',
                                                 SerialNumber='BB', PairingRequired=False, RSSI=-40),
            'AA:BB:CC:DD:EE:0A': dragon_info(),
        }
        _, described = self.describe(infos)
        keys = {item['device_key'] for item in described}
        self.assertEqual(keys, {'brainbit_classic', 'dragon_eeg'})
        classic = [item for item in described if item['device_key'] == 'brainbit_classic'][0]
        self.assertEqual(classic['channels'], ['O1', 'O2', 'T3', 'T4'])

    def test_selection_returns_the_device_the_user_picked(self):
        infos = {'AA:BB:CC:DD:EE:0A': dragon_info()}
        client, described = self.describe(infos)
        client.selection_required = True
        client.selection_timeout = 0.
        client.select_device('AA:BB:CC:DD:EE:0A')
        self.assertEqual(client._choose(infos, described), 'AA:BB:CC:DD:EE:0A')

    def test_selection_falls_back_to_the_saved_brainbit_address(self):
        infos = {'DA:BB:A3:A0:73:4E': SimpleNamespace(SensFamily=SensorFamily.LEBrainBit,
                                                     Address='DA:BB:A3:A0:73:4E',
                                                     Name='BrainBit', SensModel=0,
                                                     SerialNumber='BB', PairingRequired=False,
                                                     RSSI=-40)}
        client = engine.EEGClient(target_address='DA:BB:A3:A0:73:4E', selection_required=True,
                                  selection_timeout=0.)
        described = client._describe(infos)
        client._running = False          # no endless wait in the test
        chosen = client._choose(infos, described)
        self.assertEqual(chosen, 'DA:BB:A3:A0:73:4E')

    def test_saved_address_is_used_when_no_selection_is_requested(self):
        infos = {'DA:BB:A3:A0:73:4E': SimpleNamespace(SensFamily=SensorFamily.LEBrainBit,
                                                     Address='DA:BB:A3:A0:73:4E',
                                                     Name='BrainBit', SensModel=0,
                                                     SerialNumber='BB', PairingRequired=False,
                                                     RSSI=-40)}
        client, described = self.describe(infos)
        self.assertEqual(client._choose(infos, described), 'DA:BB:A3:A0:73:4E')

    def test_other_peoples_headsets_are_never_selected_silently(self):
        infos = {'11:22:33:44:55:66': SimpleNamespace(SensFamily=SensorFamily.LEBrainBit,
                                                     Address='11:22:33:44:55:66',
                                                     Name='Other', SensModel=0,
                                                     SerialNumber='X', PairingRequired=False,
                                                     RSSI=-70)}
        client, described = self.describe(infos)
        with self.assertRaises(RuntimeError) as raised:
            client._choose(infos, described)
        self.assertIn('not found', str(raised.exception).lower())

    def test_unsupported_family_is_refused_after_a_manual_selection(self):
        sensor = DragonSensorFixture()
        infos = {'AA:BB:CC:DD:EE:0A': SimpleNamespace(
            SensFamily=SensorFamily.LEHeadband, SensModel=0, Name='Band',
            Address='AA:BB:CC:DD:EE:0A', SerialNumber='HB', PairingRequired=False, RSSI=-60)}
        scanner = ScannerFixture(list(infos.values()), sensor)
        client = engine.EEGClient(scanner_factory=lambda filters: scanner, scan_seconds=.01,
                                  target_address='AA:BB:CC:DD:EE:0A')
        status = []
        client.status_changed.connect(status.append)
        client.run()
        self.assertEqual(scanner.created, [])
        self.assertTrue(any('not supported' in text.lower() for text in status))


if __name__ == '__main__':
    unittest.main()
