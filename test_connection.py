import unittest
from types import SimpleNamespace
from neurosdk.cmn_types import SensorCommand, SensorState, SensorSamplingFrequency
from engine import EEGClient

class SensorFixture:
    name='BrainBit'; serial_number='TEST'; batt_power=92
    sampling_frequency=SensorSamplingFrequency.FrequencyHz250
    state=SensorState.StateInRange
    signalDataReceived=None
    def __init__(self): self.commands=[]; self.disconnected=False; self.client=None
    def exec_command(self, command):
        self.commands.append(command)
        if command == SensorCommand.StartSignal:
            self.signalDataReceived(self,[SimpleNamespace(PackNum=1,O1=1e-6,O2=2e-6,T3=3e-6,T4=4e-6)])
            self.client._running = False
    def disconnect(self): self.disconnected=True

class ScannerFixture:
    def __init__(self, sensor): self.sensor=sensor; self.created=False; self.stopped=False
    def start(self): pass
    def stop(self): self.stopped=True
    def sensors(self): return [SimpleNamespace(Address='DA:BB:A3:A0:73:4E',Name='BrainBit')]
    def create_sensor(self, info): self.created=True; return self.sensor

class ConnectionTests(unittest.TestCase):
    def make_client(self, scanner):
        # Attribute assertion gives an intentional RED for the missing implementation.
        self.assertTrue(hasattr(EEGClient,'status_changed'),'SDK connection status API is missing')
        return EEGClient(scanner_factory=lambda filters:scanner, scan_seconds=.05)
    def test_real_sdk_sequence_and_units(self):
        sensor=SensorFixture(); scanner=ScannerFixture(sensor)
        client=self.make_client(scanner); sensor.client=client
        rows=[]
        client.data_received.connect(rows.append)
        client.run()
        self.assertTrue(scanner.created)
        self.assertEqual(sensor.commands,[SensorCommand.StartSignal,SensorCommand.StopSignal])
        self.assertTrue(sensor.disconnected)
        self.assertIsNone(sensor.signalDataReceived)
        self.assertEqual(rows,[[1.,2.,3.,4.]])
        self.assertEqual(client.sample_count,1)
    def test_other_device_is_never_silently_selected(self):
        sensor=SensorFixture(); scanner=ScannerFixture(sensor)
        scanner.sensors=lambda:[SimpleNamespace(Address='00:00:00:00:00:00',Name='Other')]
        client=self.make_client(scanner)
        status=[]; client.status_changed.connect(status.append)
        client.run()
        self.assertFalse(scanner.created)
        self.assertTrue(any('not found' in x.lower() for x in status))
        self.assertTrue(scanner.stopped)

if __name__=='__main__': unittest.main()
