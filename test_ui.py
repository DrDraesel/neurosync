"""UI fixtures below are TEST ONLY and never originate from the live headset."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import time
import tempfile
from pathlib import Path
from unittest import mock
import numpy as np
from PyQt6.QtWidgets import QApplication
try:
    from main import NeuroSyncApp
except ImportError:
    NeuroSyncApp=None
import ai_doctor

class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.qt=QApplication.instance() or QApplication([])
    def setUp(self):
        self.assertIsNotNone(NeuroSyncApp,'Updated four-channel analysis UI missing')
        self.window=NeuroSyncApp(autostart=False)
        self.addCleanup(self.window.close)
    def feed_fixture(self):
        t=np.arange(1250)/250
        rows=np.tile((20*np.sin(2*np.pi*10*t))[:,None],(1,4))
        self.window.handle_batch({'samples_uv':rows.tolist(),'packnums':[int(n) for n in np.arange(1250)//2],
                                  'arrived_monotonic':time.monotonic(),'sample_count':1250,'gap_events':0})
        self.window.handle_contact({'ohms':dict.fromkeys(('O1','O2','T3','T4'),100000),
                                    'measured_monotonic':time.monotonic()})
    def test_all_channels_and_bands_have_curves_and_colors(self):
        self.assertEqual(set(self.window.raw_curves),{'O1','O2','T3','T4'})
        self.assertEqual(set(self.window.band_curves),{'Delta','Theta','Alpha','Beta','Gamma'})
        colors={curve.opts['pen'].color().name() for curve in self.window.band_curves.values()}
        self.assertEqual(len(colors),5)
    def test_five_second_summary_uses_actual_buffer(self):
        self.feed_fixture(); self.window.analyze_now()
        self.assertTrue(self.window.last_report['valid'])
        self.assertEqual(self.window.last_report['dominant'],'Alpha')
        self.assertIn('Alpha',self.window.conclusion.text())
    def test_reset_and_stale_hide_previous_conclusion(self):
        self.feed_fixture(); self.window.analyze_now()
        self.window.reset_window('TEST gap')
        self.assertEqual(len(self.window.samples),0)
        self.assertIsNone(self.window.last_report)
        self.assertNotIn('Largest',self.window.conclusion.text())
    def test_unknown_contact_suppresses_interpretation(self):
        self.feed_fixture(); self.window.contact=None; self.window.analyze_now()
        self.assertFalse(self.window.last_report['valid'])
        self.assertIsNone(self.window.last_report['dominant'])
    def test_snapshot_exports_raw_and_quality_metadata(self):
        self.feed_fixture(); self.window.analyze_now()
        with tempfile.TemporaryDirectory() as tmp:
            self.window.export_snapshot(Path(tmp))
            self.assertEqual(len((Path(tmp)/'raw.csv').read_text().splitlines()),1251)
            self.assertTrue((Path(tmp)/'report.json').is_file())
    def test_stale_window_not_live(self):
        self.feed_fixture(); self.window.last_received=time.monotonic()-3
        self.window.analyze_now()
        self.assertFalse(self.window.last_report['valid'])
        self.assertIsNone(self.window.last_report['dominant'])
    def test_local_mode_labels_say_the_model_is_local(self):
        with mock.patch.object(ai_doctor,'AI_API_URL',''):
            self.assertIn('local model — nothing leaves this PC',self.window.chat_note.text())
            self.assertIn('the local model on this PC',self.window.ai_text.text())
            self.assertNotIn('API model',self.window.chat_note.text())
    def test_api_mode_labels_name_the_endpoint_and_drop_local_only_claims(self):
        with mock.patch.object(ai_doctor,'AI_API_URL','https://api.example.com/v1'):
            window=NeuroSyncApp(autostart=False)
            self.addCleanup(window.close)
            self.assertIn('API model — derived numbers are sent to api.example.com',
                          window.chat_note.text())
            self.assertIn('the API at api.example.com',window.ai_text.text())
            self.assertIn('derived numbers are sent to api.example.com',window.chat_tooltip)
            self.assertNotIn('never leaves this PC',window.ai_text.text())
            self.assertNotIn('Nothing is uploaded',window.chat_note.text())

if __name__=='__main__': unittest.main()
