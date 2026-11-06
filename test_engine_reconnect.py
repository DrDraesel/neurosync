"""TEST ONLY fixtures for bounded auto-reconnect after stream loss.

No BLE, no hardware: fake scanners hand out doomed or healthy BrainBit
Classic sensors, one per connect attempt, and the reconnect backoff is
shortened to zero so the reconnect state machine runs at test speed.
Nothing here is emitted by a live headset, and none of it is reachable
from the production acquisition path.
"""
import threading
import time
import unittest
from types import SimpleNamespace

from neurosdk.cmn_types import (SensorCommand, SensorFamily,
                                SensorSamplingFrequency, SensorState)

import devices
import engine

ADDR = 'DA:BB:A3:A0:73:4E'


def classic_info(address=ADDR, family=SensorFamily.LEBrainBit):
    return SimpleNamespace(SensFamily=family, SensModel=0, Name='BrainBit',
                           Address=address, SerialNumber='BB-TEST',
                           PairingRequired=False, RSSI=-40)


class ClassicSensorFixture:
    """BrainBit Classic-shaped fake; ``healthy=False`` reports out-of-range.

    Both variants deliver one packet when StartSignal arrives (so rows are
    decoded exactly like the healthy connection path); a healthy sensor also
    ends the run the way the existing fixtures do.
    """

    name = 'BrainBit'
    serial_number = 'BB-TEST'
    batt_power = 92
    signalDataReceived = None
    resistDataReceived = None

    def __init__(self, healthy=True, frequency=None):
        self.healthy = healthy
        self.sampling_frequency = frequency or SensorSamplingFrequency.FrequencyHz250
        self.commands = []
        self.disconnected = False
        self.client = None

    @property
    def state(self):
        return SensorState.StateInRange if self.healthy else SensorState.StateOutOfRange

    def exec_command(self, command):
        self.commands.append(command)
        if command == SensorCommand.StartSignal and self.signalDataReceived is not None:
            self.signalDataReceived(self, [SimpleNamespace(
                PackNum=1, O1=1e-6, O2=2e-6, T3=3e-6, T4=4e-6)])
            if self.client is not None and self.healthy:
                self.client._running = False

    def disconnect(self):
        self.disconnected = True


class AttemptScanner:
    """One scan's worth of results; create_sensor hands out its own sensor."""

    def __init__(self, infos, sensor=None):
        self.infos = list(infos)
        self.sensor = sensor
        self.started = False
        self.stopped = False
        self.created = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def sensors(self):
        return list(self.infos)

    def create_sensor(self, info):
        self.created.append(info)
        if self.sensor is None:
            raise RuntimeError('TEST fixture has no sensor instance')
        return self.sensor


class AttemptFactory:
    """scanner_factory that returns one AttemptScanner per scanner request."""

    def __init__(self, attempts):
        # attempts: list of (infos, sensor) tuples, one per request.
        self.attempts = list(attempts)
        self.calls = 0
        self.scanners = []

    def __call__(self, filters):
        index = min(self.calls, len(self.attempts) - 1)
        infos, sensor = self.attempts[index]
        scanner = AttemptScanner(infos, sensor)
        self.scanners.append(scanner)
        self.calls += 1
        return scanner


class AutoReconnectTests(unittest.TestCase):
    def make_client(self, factory, **kwargs):
        options = dict(scanner_factory=factory, scan_seconds=.01,
                       target_address=ADDR, auto_contact=False,
                       max_reconnect=4, reconnect_backoff=(0., 0., 0., 0.))
        options.update(kwargs)
        return engine.EEGClient(**options)

    def test_defaults_match_the_reconnect_contract(self):
        self.assertEqual(engine.MAX_RECONNECT, 4)
        self.assertEqual(engine.RECONNECT_BACKOFF_S, (2., 4., 8., 12.))
        client = engine.EEGClient()
        self.assertEqual(client.max_reconnect, 4)
        self.assertEqual(client.reconnect_backoff, (2., 4., 8., 12.))

    def test_stream_loss_triggers_reconnect_then_resumes(self):
        doomed = ClassicSensorFixture(healthy=False)
        healthy = ClassicSensorFixture(healthy=True)
        factory = AttemptFactory([([classic_info()], doomed),
                                  ([classic_info()], healthy)])
        client = self.make_client(factory)
        doomed.client = client
        healthy.client = client
        rows, resets, status = [], [], []
        client.data_received.connect(rows.append)
        client.stream_reset.connect(resets.append)
        client.status_changed.connect(status.append)
        client.run()
        self.assertEqual(factory.calls, 2)
        self.assertIn('Connection lost — reconnecting (attempt 1 of 4)', resets)
        self.assertIn('Reconnected after connection loss', resets)
        self.assertTrue(any('Reconnected after connection loss' in text for text in status))
        self.assertTrue(any('(attempt 1 of 4)' in text for text in status))
        # rows from BOTH sessions still decoded in uV; sample_count cumulative
        self.assertEqual(rows, [[1., 2., 3., 4.], [1., 2., 3., 4.]])
        self.assertEqual(client.sample_count, 2)
        self.assertTrue(doomed.disconnected)
        self.assertTrue(healthy.disconnected)
        self.assertEqual(healthy.commands,
                         [SensorCommand.StartSignal, SensorCommand.StopSignal])
        self.assertEqual(client.mode, 'disconnected')
        # never a different device: every created sensor used the saved address
        for scanner in factory.scanners:
            for info in scanner.created:
                self.assertEqual(info.Address, ADDR)

    def test_success_on_attempt_2_resumes_streaming(self):
        first = ClassicSensorFixture(healthy=False)
        second = ClassicSensorFixture(healthy=False)
        healthy = ClassicSensorFixture(healthy=True)
        factory = AttemptFactory([([classic_info()], first),
                                  ([classic_info()], second),
                                  ([classic_info()], healthy)])
        client = self.make_client(factory)
        for sensor in (first, second, healthy):
            sensor.client = client
        rows, resets = [], []
        client.data_received.connect(rows.append)
        client.stream_reset.connect(resets.append)
        client.run()
        self.assertIn('Connection lost — reconnecting (attempt 1 of 4)', resets)
        self.assertIn('Connection lost — reconnecting (attempt 2 of 4)', resets)
        self.assertIn('Reconnected after connection loss', resets)
        # rows decoded from every session, including after the reconnect
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1], [1., 2., 3., 4.])
        self.assertEqual(client.sample_count, 3)
        self.assertEqual(factory.calls, 3)

    def test_exhausted_attempts_yield_error_mode_and_final_message(self):
        sensors = [ClassicSensorFixture(healthy=False) for _ in range(5)]
        factory = AttemptFactory([([classic_info()], sensor) for sensor in sensors])
        client = self.make_client(factory)
        for sensor in sensors:
            sensor.client = client
        resets, status = [], []
        client.stream_reset.connect(resets.append)
        client.status_changed.connect(status.append)
        client.run()
        self.assertEqual(factory.calls, 5)          # initial + 4 attempts
        for attempt in range(1, 5):
            self.assertIn(f'Connection lost — reconnecting (attempt {attempt} of 4)', resets)
        # each of the 4 attempts reconnected, then lost the stream again; the
        # 4th loss exhausts the budget and reports the error
        self.assertEqual(resets.count('Reconnected after connection loss'), 4)
        self.assertIn('Connection lost; auto-reconnect failed after 4 attempts — '
                      'power-cycle the headset and press Scan again', resets)
        self.assertEqual(client.mode, 'error')
        self.assertTrue(any(text.startswith('Connection error: Headset disconnected '
                                            'or out of range') for text in status))
        # every sensor (including the current one) was cleaned up
        for sensor in sensors:
            self.assertTrue(sensor.disconnected)

    def test_stop_during_backoff_aborts_without_reconnecting(self):
        doomed = ClassicSensorFixture(healthy=False)
        factory = AttemptFactory([([classic_info()], doomed),
                                  ([classic_info()], ClassicSensorFixture(healthy=True))])
        client = self.make_client(factory, reconnect_backoff=(30., 0., 0., 0.))
        doomed.client = client
        thread = threading.Thread(target=client.run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10.
        while client.mode != 'reconnecting' and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertEqual(client.mode, 'reconnecting')
        client.stop()
        thread.join(5.)
        self.assertFalse(thread.is_alive())
        self.assertEqual(factory.calls, 1)      # no rescan: no fresh scanner
        self.assertTrue(doomed.disconnected)
        self.assertEqual(client.mode, 'disconnected')

    def test_rescan_finding_nothing_counts_as_failed_attempt(self):
        doomed = ClassicSensorFixture(healthy=False)
        factory = AttemptFactory([([classic_info()], doomed)]
                                 + [([], None) for _ in range(4)])
        client = self.make_client(factory)
        doomed.client = client
        resets, status = [], []
        client.stream_reset.connect(resets.append)
        client.status_changed.connect(status.append)
        client.run()
        self.assertEqual(factory.calls, 5)
        for attempt in range(1, 5):
            self.assertIn(f'Connection lost — reconnecting (attempt {attempt} of 4)', resets)
        self.assertIn('Connection lost; auto-reconnect failed after 4 attempts — '
                      'power-cycle the headset and press Scan again', resets)
        self.assertEqual(client.mode, 'error')
        self.assertTrue(any('not seen during rescan' in text for text in status))
        # nothing was ever connected during the failed attempts
        for scanner in factory.scanners[1:]:
            self.assertEqual(scanner.created, [])

    def test_reconnect_never_switches_devices_or_families(self):
        doomed = ClassicSensorFixture(healthy=False)
        healthy = ClassicSensorFixture(healthy=True)
        factory = AttemptFactory([
            ([classic_info()], doomed),
            # same address re-appears as a DIFFERENT family -> refused
            ([classic_info(family=SensorFamily.LENeuroEEG)], None),
            ([classic_info()], healthy),
        ])
        client = self.make_client(factory)
        doomed.client = client
        healthy.client = client
        resets, status = [], []
        client.stream_reset.connect(resets.append)
        client.status_changed.connect(status.append)
        client.run()
        self.assertIn('Connection lost — reconnecting (attempt 1 of 4)', resets)
        self.assertIn('Connection lost — reconnecting (attempt 2 of 4)', resets)
        self.assertIn('Reconnected after connection loss', resets)
        self.assertTrue(any('refusing to switch devices' in text for text in status))
        # the refused scanner never created a sensor; the accepted one did
        self.assertEqual(factory.scanners[1].created, [])
        self.assertEqual(factory.scanners[2].created[0].Address, ADDR)
        self.assertEqual(client.device_spec.key, devices.CLASSIC.key)

    def test_midway_attempt_failure_cleans_up_partial_sensor(self):
        doomed = ClassicSensorFixture(healthy=False)
        bad = ClassicSensorFixture(healthy=False,
                                   frequency=SensorSamplingFrequency.FrequencyHz1000)
        healthy = ClassicSensorFixture(healthy=True)
        factory = AttemptFactory([([classic_info()], doomed),
                                  ([classic_info()], bad),
                                  ([classic_info()], healthy)])
        client = self.make_client(factory)
        for sensor in (doomed, bad, healthy):
            sensor.client = client
        resets, status = [], []
        client.stream_reset.connect(resets.append)
        client.status_changed.connect(status.append)
        client.run()
        # _prepare_device failed on attempt 1 (unsupported rate), so attempt 2
        # reconnected; the partially-created sensor was disconnected too
        self.assertTrue(any('Unsupported sample rate' in text for text in status))
        self.assertIn('Connection lost — reconnecting (attempt 2 of 4)', resets)
        self.assertIn('Reconnected after connection loss', resets)
        self.assertNotIn(SensorCommand.StartSignal, bad.commands)
        self.assertTrue(bad.disconnected)
        self.assertTrue(healthy.disconnected)
        self.assertEqual(client.mode, 'disconnected')


if __name__ == '__main__':
    unittest.main()
