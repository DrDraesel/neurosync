"""UI fixtures below are TEST ONLY and never originate from the live headset."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import time
import tempfile
from pathlib import Path
import numpy as np
from PyQt6.QtWidgets import QApplication
try:
    from main import NeuroSyncApp
except ImportError:
    NeuroSyncApp=None

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

if __name__=='__main__': unittest.main()
