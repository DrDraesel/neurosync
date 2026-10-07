"""TEST ONLY SDK sequence fixtures; never emitted by the live application."""
import unittest
from types import SimpleNamespace
import engine

class ProtocolTests(unittest.TestCase):
    def make(self):
        self.assertTrue(hasattr(engine,'PacketTracker'),'Packet integrity tracker missing')
        return engine.PacketTracker()
    def packets(self, numbers):
        return [SimpleNamespace(PackNum=n,O1=1e-6,O2=2e-6,T3=3e-6,T4=4e-6) for n in numbers]
    def test_two_samples_per_packet_and_wrap(self):
        p=self.make()
        r=p.decode(self.packets([2047,2047,0,0,1,1]))
        self.assertFalse(r['gap'])
        self.assertEqual(len(r['samples_uv']),6)
        self.assertEqual(r['samples_uv'][0],[1,2,3,4])
    def test_gap_discards_pre_gap_window_and_keeps_post_gap_data(self):
        p=self.make()
        r=p.decode(self.packets([1,1,4,4]))
        self.assertTrue(r['gap'])
        self.assertEqual(r['packnums'],[4,4])
        self.assertEqual(p.gap_events,1)
    def test_nonfinite_breaks_continuity(self):
        p=self.make(); data=self.packets([1,1,2,2]); data[1].O1=float('nan')
        r=p.decode(data)
        self.assertTrue(r['gap']); self.assertEqual(len(r['samples_uv']),2)
    def test_third_repeated_packet_is_not_silently_accepted(self):
        p=self.make(); r=p.decode(self.packets([7,7,7]))
        self.assertTrue(r['gap'])
    def test_contact_protocol_stops_signal_then_checks_then_resumes(self):
        self.assertTrue(hasattr(engine.EEGClient,'check_contact'),'Contact measurement protocol missing')
        from neurosdk.cmn_types import SensorCommand
        class Sensor:
            signalDataReceived=None; resistDataReceived=None
            def __init__(self): self.commands=[]
            def exec_command(self,cmd):
                self.commands.append(cmd)
                if cmd==SensorCommand.StartResist:
                    self.resistDataReceived(self,SimpleNamespace(O1=100000,O2=200000,T3=300000,T4=400000))
        client=engine.EEGClient(contact_seconds=0)
        client.device=Sensor(); readings=[]
        client.contact_received.connect(readings.append)
        client._measure_contact(resume=True)
        self.assertEqual(client.device.commands,[SensorCommand.StopSignal,SensorCommand.StartResist,SensorCommand.StopResist,SensorCommand.StartSignal])
        self.assertEqual(readings[-1]['ohms']['O1'],100000)
        self.assertIsNone(client.device.resistDataReceived)
    def test_contact_request_is_queued_not_executed_on_gui_thread(self):
        self.assertTrue(hasattr(engine.EEGClient,'check_contact'),'Contact request API missing')
        client=engine.EEGClient(); client.check_contact()
        self.assertTrue(client.contact_requested.is_set())

if __name__=='__main__': unittest.main()
