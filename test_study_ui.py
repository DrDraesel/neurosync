"""UI fixtures below are TEST ONLY and never originate from the live headset.

UI-level checks for the full-duration STUDY flow: the toolbar buttons exist
and are gated on subject + live streaming, study rows accumulate past the 60 s
deque, Stop saves the FULL span with an honest summary and auto-runs the AI
doctor on the study numbers, and the cap auto-stops and notes itself.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PyQt6.QtWidgets import QApplication

import main
import storage
import study

CHANNELS = ('O1', 'O2', 'T3', 'T4')


class _StreamingClient:
    """TEST ONLY stand-in for a live BLE client in signal mode."""

    def __init__(self):
        self.mode = 'signal'

    def isRunning(self):
        return True

    def stop(self):
        return True


class _StubSignal:
    def __init__(self):
        self.slot = None

    def connect(self, slot):
        self.slot = slot


class _FakeAIWorker:
    """TEST ONLY: captures the payload instead of calling a model backend."""

    def __init__(self, payload, model, parent=None):
        self.payload = payload
        self.model = model
        self.note_ready = _StubSignal()
        self.started = False

    def start(self):
        self.started = True

    def isRunning(self):
        return False


class StudyUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = main.NeuroSyncApp(autostart=False)
        self.addCleanup(self.window.close)

    def sine_rows(self, n, fs=250, hz=10.0, channels=4, amplitude=20.0):
        t = np.arange(n) / fs
        column = amplitude * np.sin(2 * np.pi * hz * t)
        return np.tile(column[:, None], (1, channels))

    def feed(self, rows):
        self.window.handle_batch({
            'samples_uv': rows.tolist(),
            'packnums': [int(n) for n in np.arange(len(rows)) // 2],
            'arrived_monotonic': time.monotonic(),
            'sample_count': len(rows), 'gap_events': 0})
        self.window.handle_contact({'ohms': dict.fromkeys(CHANNELS, 100_000.),
                                    'measured_monotonic': time.monotonic()})

    def test_buttons_exist_and_are_gated(self):
        window = self.window
        self.assertEqual(window.start_study_button.text(), 'Start study (record)')
        self.assertEqual(window.stop_study_button.text(), 'Stop study')
        window.update_buttons()
        # fresh window: no subject, no stream -> both disabled
        self.assertFalse(window.start_study_button.isEnabled())
        self.assertFalse(window.stop_study_button.isEnabled())
        # a subject alone is not enough
        window.subject = {'name': 'TEST ONLY', 'phone': ''}
        window.update_buttons()
        self.assertFalse(window.start_study_button.isEnabled())
        # a live stream alone is not enough
        window.subject = None
        window.client = _StreamingClient()
        window.update_buttons()
        self.assertFalse(window.start_study_button.isEnabled())
        # subject + streaming unlocks Start; Stop stays disabled until started
        window.subject = {'name': 'TEST ONLY', 'phone': ''}
        window.update_buttons()
        self.assertTrue(window.start_study_button.isEnabled())
        self.assertFalse(window.stop_study_button.isEnabled())
        window.start_study()
        window.update_buttons()
        self.assertFalse(window.start_study_button.isEnabled())
        self.assertTrue(window.stop_study_button.isEnabled())
        self.assertIn('STUDY', window.study_status.text())
        self.assertIn('recording', window.study_status.text())

    def test_study_rows_accumulate_beyond_the_60s_deque(self):
        # 15001 rows at 250 Hz is 60.004 s: the deque keeps 15000, the study
        # must keep ALL of them (the core full-duration gap).
        self.window.subject = {'name': 'TEST ONLY Study', 'phone': ''}
        self.window.client = _StreamingClient()
        self.window.start_study()
        self.feed(self.sine_rows(15001))
        self.assertEqual(len(self.window.study_rows), 15001)
        self.assertEqual(len(self.window.samples), 15000)  # the deque, capped

    def test_stop_study_saves_the_full_span_and_auto_runs_ai(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(main, 'SUBJECT_ROOT', Path(tmp)), \
                mock.patch.object(main, 'AIDoctorWorker', _FakeAIWorker):
            self.window.subject = {'name': 'TEST ONLY Study', 'phone': '555'}
            self.window.client = _StreamingClient()
            self.window.start_study()
            n = 4000                       # 16 s of 10 Hz alpha
            self.feed(self.sine_rows(n))
            time.sleep(0.05)               # let the monotonic clock move
            self.window.stop_study(reason='user')

            sessions = storage.list_sessions(Path(tmp), 'TEST ONLY Study', '555')
            self.assertEqual(len(sessions), 1)
            folder = sessions[0]['path']
            summary = storage.read_json_object(folder / storage.SUMMARY_FILE)
            self.assertEqual(summary['raw_rows'], n)
            self.assertEqual(summary['recorded_span_s'], summary['duration_s'])
            self.assertGreater(summary['recorded_span_s'], 0)
            self.assertIn('full study recording', summary['raw_span_note'])
            self.assertNotIn('rolling', summary['raw_span_note'])
            self.assertEqual(summary['study']['kind'], 'study')
            self.assertEqual(summary['study']['stop_reason'], 'user')
            self.assertFalse(summary['study']['capped'])
            self.assertEqual(summary['study']['stream_resets'], 0)
            # the analysis came from the FULL study rows, not the live deque
            stable = summary['stable_estimate']
            self.assertEqual(stable['seconds_total'], 16.0)
            self.assertTrue(stable['valid'], stable['reason'])
            self.assertEqual(summary['analysis']['samples'], 1250)   # final 5 s
            # the raw CSV holds every row plus the header
            loaded = storage.load_session(folder)
            self.assertEqual(len(loaded['raw_rows']), n)
            self.assertIn('Study saved locally', self.window.statusBar().currentMessage())
            # the AI worker got the STUDY payload (not the live deque numbers) and ran
            worker = self.window.ai_worker
            self.assertIsInstance(worker, _FakeAIWorker)
            self.assertTrue(worker.started)
            self.assertEqual(worker.payload['five_second']['samples'], 1250)
            self.assertEqual(worker.payload['stable_estimate']['seconds_total'], 16.0)
            # when the note lands, it is written into the saved session with provenance
            self.window.handle_ai_note({'ok': True, 'model': 'TEST MODEL', 'text': 'note'})
            updated = storage.read_json_object(folder / storage.SUMMARY_FILE)
            note = updated['ai_doctor']
            self.assertEqual(note['model'], 'TEST MODEL')
            self.assertTrue(note['ok'])
            self.assertIn('generated_at', note)
            self.assertIn('provenance', note)
            self.assertIsNone(self.window.ai_pending_session_folder)

    def test_study_auto_stops_at_the_cap_and_notes_it(self):
        # Patch the cap down so the auto-stop path runs with a small buffer.
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(main, 'SUBJECT_ROOT', Path(tmp)), \
                mock.patch.object(main, 'AIDoctorWorker', _FakeAIWorker), \
                mock.patch.object(study, 'max_rows', return_value=2000):
            self.window.subject = {'name': 'TEST ONLY Cap', 'phone': ''}
            self.window.client = _StreamingClient()
            self.window.start_study()
            self.feed(self.sine_rows(2500))     # handle_batch hits the cap

            # the study auto-stopped and saved itself, with the clear cap note
            self.assertIsNone(self.window.study_started_monotonic)
            self.assertIn(study.CAP_MESSAGE, self.window.statusBar().currentMessage())
            self.assertIn('60-minute cap', self.window.study_status.text())
            sessions = storage.list_sessions(Path(tmp), 'TEST ONLY Cap', '')
            self.assertEqual(len(sessions), 1)
            summary = storage.read_json_object(sessions[0]['path'] / storage.SUMMARY_FILE)
            self.assertTrue(summary['study']['capped'])
            self.assertEqual(summary['study']['stop_reason'], 'cap')
            self.assertEqual(summary['raw_rows'], 2500)          # nothing truncated
            self.assertIn('60-minute cap', summary['raw_span_note'])

    def test_patient_report_exports_a_saved_study(self):
        import patient_report
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(main, 'SUBJECT_ROOT', Path(tmp)), \
                mock.patch.object(main, 'AIDoctorWorker', _FakeAIWorker):
            self.window.subject = {'name': 'TEST ONLY Study', 'phone': '555'}
            self.window.client = _StreamingClient()
            self.window.start_study()
            self.feed(self.sine_rows(4000))
            time.sleep(0.05)               # let the monotonic clock move
            self.window.stop_study(reason='user')
            out = Path(tmp) / 'patient_report.html'
            written = patient_report.build(Path(tmp), 'TEST ONLY Study', '555',
                                           out_path=out)
            self.assertEqual(written, out)
            self.assertTrue(out.is_file())
            html = out.read_text(encoding='utf-8')
            # the report renders the study session as a full-duration study
            self.assertIn('full-duration study', html)
            self.assertIn('TEST ONLY Study', html)


if __name__ == '__main__':
    unittest.main()
