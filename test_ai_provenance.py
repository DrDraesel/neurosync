"""AI provenance labels: local mode vs API mode. No network calls are made.

The app must never claim 'local model / nothing leaves this PC' while
AI_API_URL points at an endpoint, so these tests pin the one helper every UI
string and report line uses.
"""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import ai_doctor

BASE = Path(__file__).resolve().parent


def _run(env_value):
    """Run a fresh interpreter with AI_API_URL set (temp environment)."""
    env = {key: value for key, value in os.environ.items()
           if key not in ('AI_API_URL', 'AI_API_KEY', 'AI_API_MODEL')}
    env['AI_API_URL'] = env_value
    return subprocess.run(
        [sys.executable, '-c',
         'import ai_doctor; print(ai_doctor.provenance_label()); '
         'print(ai_doctor.backend_label()); print(ai_doctor.data_destination())'],
        cwd=str(BASE), env=env, capture_output=True, text=True, timeout=120)


class ProvenanceLabelTests(unittest.TestCase):
    def test_local_mode_label_says_nothing_leaves_the_pc(self):
        with mock.patch.object(ai_doctor, 'AI_API_URL', ''):
            self.assertFalse(ai_doctor.api_mode())
            self.assertEqual(ai_doctor.backend_label(), 'local model')
            self.assertEqual(ai_doctor.provenance_label(),
                             'local model — nothing leaves this PC')
            self.assertEqual(ai_doctor.data_destination(), 'the local model on this PC')

    def test_api_mode_label_names_the_endpoint_host(self):
        with mock.patch.object(ai_doctor, 'AI_API_URL', 'https://api.example.com/v1'):
            self.assertTrue(ai_doctor.api_mode())
            self.assertEqual(ai_doctor.backend_label(), 'API model (api.example.com)')
            self.assertEqual(ai_doctor.provenance_label(),
                             'API model — derived numbers are sent to api.example.com')
            self.assertEqual(ai_doctor.data_destination(), 'the API at api.example.com')

    def test_api_mode_label_comes_from_a_temp_environment(self):
        result = _run('https://api.example.com/v1')
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[0], 'API model — derived numbers are sent to api.example.com')
        self.assertEqual(lines[1], 'API model (api.example.com)')
        self.assertEqual(lines[2], 'the API at api.example.com')
        self.assertNotIn('nothing leaves this PC', result.stdout)

    def test_local_mode_label_comes_from_a_temp_environment(self):
        result = _run('')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[0],
                         'local model — nothing leaves this PC')

    def test_api_host_keeps_port_and_falls_back_to_the_raw_url(self):
        with mock.patch.object(ai_doctor, 'AI_API_URL', 'http://127.0.0.1:1234/v1'):
            self.assertEqual(ai_doctor.api_host(), '127.0.0.1:1234')
        with mock.patch.object(ai_doctor, 'AI_API_URL', 'api.example.com/v1'):
            self.assertEqual(ai_doctor.api_host(), 'api.example.com/v1')

    def test_prompts_do_not_claim_a_nominal_250_hz_rate(self):
        for prompt in (ai_doctor.SYSTEM_PROMPT, ai_doctor.LIVE_SYSTEM_PROMPT):
            self.assertNotIn('Nominal sampling 250 Hz', prompt)
            self.assertIn('250, 500 or 1000 Hz', prompt)


if __name__ == '__main__':
    unittest.main()
