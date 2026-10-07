"""Patient report + offline interpretation helper tests (no hardware)."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import ai_doctor  # noqa: E402
import patient_report  # noqa: E402
import storage  # noqa: E402


def _bands(alpha=45.0, theta=20.0):
    rest = (100.0 - alpha - theta) / 3.0
    return {'Delta': {'power_uv2': 5.0, 'relative_pct': rest},
            'Theta': {'power_uv2': 9.0, 'relative_pct': theta},
            'Alpha': {'power_uv2': 20.0, 'relative_pct': alpha},
            'Beta': {'power_uv2': 7.0, 'relative_pct': rest},
            'Gamma': {'power_uv2': 2.0, 'relative_pct': rest}}


def _channels():
    return {
        'O1': {'valid': True, 'dominant': 'Alpha', 'relative_pct': {'Alpha': 52.0},
               'powers_uv2': {'Alpha': 21.0}, 'flags': []},
        'O2': {'valid': True, 'dominant': 'Alpha', 'relative_pct': {'Alpha': 44.0},
               'powers_uv2': {'Alpha': 18.0}, 'flags': []},
        'T3': {'valid': True, 'dominant': 'Theta', 'relative_pct': {'Theta': 34.0},
               'powers_uv2': {'Theta': 8.0}, 'flags': []},
        'T4': {'valid': False, 'dominant': 'Theta', 'relative_pct': {'Theta': 30.0},
               'powers_uv2': {'Theta': 7.0}, 'flags': ['high_resistance']},
    }


def fake_report(alpha=45.0):
    return {'valid': True, 'ready': True, 'reason': 'all quality gates passed',
            'dominant': 'Alpha', 'dominant_pct': alpha, 'peak_hz': 10.2,
            'samples': 1250, 'bands': _bands(alpha=alpha), 'channels': _channels()}


class ReportFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'subjects'
        self.root.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def seed(self, name='Report Person', phone='555', *, simulated=False,
             ai_text='', with_charts=True):
        summary = {
            'started_at': datetime.now(timezone.utc).isoformat(),
            'device_type': 'brainbit_classic',
            'device_label': ('SIMULATED BrainBit Classic' if simulated
                             else 'BrainBit Classic / Black'),
            'device': {'simulated': True} if simulated else {},
            'simulated': simulated,
            'channels': ['O1', 'O2', 'T3', 'T4'],
            'fs_hz': 250, 'raw_rows': 1200, 'duration_s': 4.8, 'notch_hz': 60,
            'analysis': fake_report(),
            'stable_estimate': dict(fake_report(), bands=_bands(alpha=48.0),
                                    dominant='Alpha', dominant_pct=48.0,
                                    peak_hz=10.1, epochs_used=6, epochs_total=6),
            'ai_doctor': {'ok': True, 'text': ai_text} if ai_text else None,
            'contact': None, 'transport': {'samples': 1200},
            'limits': 'Engineering estimates only.',
        }
        header = ['sample_index', 'relative_time_s', 'packet_number',
                  'raw_O1_uV', 'raw_O2_uV', 'raw_T3_uV', 'raw_T4_uV']
        rows = [[i, i / 250.0, i] + [0.0] * 4 for i in range(30)]
        charts = {}
        if with_charts:
            charts['band_power_bars.png'] = b'\x89PNG\r\n\x1a\n' + b'0' * 64
        return storage.save_session(self.root, name, phone, summary, header, rows,
                                    charts=charts)


class TestPatientReport(ReportFixture):
    def test_report_bundles_sessions_charts_and_helper(self):
        self.seed(ai_text='Stored model description of measured numbers.')
        self.seed(phone='555')
        path = patient_report.build(self.root, 'Report Person', '555',
                                    out_path=Path(self.tmp.name) / 'out.html')
        text = path.read_text(encoding='utf-8')
        self.assertIn('NeuroSync patient report', text)
        self.assertIn('Report Person', text)
        self.assertIn('History — all saved sessions (2)', text)
        self.assertIn('How to read these numbers', text)
        self.assertIn('Predominant rhythm', text)
        self.assertIn('O1 vs O2', text)
        self.assertIn('What cannot be concluded from this', text)
        self.assertIn('Alpha', text)
        self.assertIn('data:image/png;base64,', text)          # embedded chart
        self.assertIn('Stored model description of measured numbers.', text)
        self.assertIn(ai_doctor.DISCLAIMER, text)

    def test_simulated_sessions_are_flagged_in_the_report(self):
        self.seed(simulated=True)
        path = patient_report.build(self.root, 'Report Person', '555',
                                    out_path=Path(self.tmp.name) / 'sim.html')
        text = path.read_text(encoding='utf-8')
        self.assertIn('SIMULATED DATA', text)
        self.assertIn('pipeline demonstration', text)

    def test_unknown_subject_raises(self):
        with self.assertRaises(ValueError):
            patient_report.build(self.root, 'Nobody', '000')


class TestLocalInterpretation(unittest.TestCase):
    def test_sections_cover_bands_pairs_and_limits(self):
        result = ai_doctor.local_interpretation(report=fake_report(), stable=None,
                                                device={'device_label': 'Test device',
                                                        'channels': ['O1', 'O2', 'T3', 'T4']})
        titles = [section['title'] for section in result['sections']]
        self.assertIn('Predominant rhythm', titles)
        self.assertIn('Band-by-band context', titles)
        self.assertIn('Left / right electrode comparison', titles)
        self.assertIn('What cannot be concluded from this', titles)
        pair_section = next(s for s in result['sections']
                            if s['title'] == 'Left / right electrode comparison')
        self.assertIn('O1 vs O2', pair_section['text'])
        self.assertIn('+8.0', pair_section['text'])            # 52 - 44 percentage points
        blue = next(s for s in result['sections'] if s['title'] == 'Band-by-band context')
        self.assertIn('muscle', blue['text'].lower())
        self.assertEqual(result['predominant']['band'], 'Alpha')

    def test_simulated_device_is_labelled_in_helper(self):
        result = ai_doctor.local_interpretation(
            report=fake_report(), stable=None,
            device={'device_label': 'SIMULATED BrainBit', 'simulated': True,
                    'channels': ['O1', 'O2']})
        first = result['sections'][0]['text']
        self.assertIn('SIMULATED', first)

    def test_payload_predominant_and_simulated_flags(self):
        payload = ai_doctor.build_payload(report=fake_report(), stable=None,
                                          device={'device_label': 'X', 'simulated': True,
                                                  'channels': ['O1']})
        self.assertTrue(payload['simulated'])
        self.assertEqual(payload['predominant']['band'], 'Alpha')
        self.assertEqual(payload['predominant']['peak_hz'], 10.2)
        self.assertIn('channels', payload['device'])


if __name__ == '__main__':
    unittest.main()
