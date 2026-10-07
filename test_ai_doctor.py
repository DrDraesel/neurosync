"""TEST ONLY fixtures for the local AI-doctor bridge; no network is ever used."""
import io
import json
import unittest
import urllib.error

import ai_doctor as ai


class FakeResponse:
    def __init__(self, payload_bytes):
        self._payload = payload_bytes

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_returning(payload):
    def opener(request, timeout=None):
        opener.requests.append(request)
        return FakeResponse(json.dumps(payload).encode('utf-8'))
    opener.requests = []
    return opener


def sample_payload():
    report = {'valid': True, 'ready': True, 'reason': 'ok', 'dominant': 'Alpha',
              'dominant_pct': 42.1234, 'peak_hz': 10.25, 'samples': 1250,
              'bands': {'Alpha': {'power_uv2': 12.3456, 'relative_pct': 42.1234}},
              'channels': {'O1': {'dominant': 'Alpha', 'valid': True, 'flags': []}}}
    stable = {'valid': False, 'reason': 'O1: flatline', 'dominant': None,
              'dominant_pct': None, 'seconds_used': 0.0, 'seconds_total': 30.0,
              'epochs_used': 0, 'epochs_total': 15,
              'bands': {'Alpha': {'power_uv2': None, 'relative_pct': None, 'sem_uv2': None}},
              'channels': {'O1': {'dominant': None, 'flags': ['flatline'],
                                  'epochs_used': 2, 'excluded': {'flatline': 13}}}}
    transport = {'samples': 9000, 'rate_hz': 250.04, 'gap_events': 1, 'battery': 88, 'mode': 'signal'}
    device = {'name': 'BrainBit', 'serial': 'TESTONLY', 'fs': 250, 'battery': 88, 'address': 'AA:BB'}
    return ai.build_payload(report=report, stable=stable, transport=transport,
                            device=device, contact={'O1': 120000.4, 'O2': None})


class AIDoctorTests(unittest.TestCase):
    def test_payload_keeps_only_derived_numbers(self):
        payload = sample_payload()
        text = json.dumps(payload, allow_nan=False)
        for raw_field in ('samples_uv', 'packnums', 'raw_', 'uV'):
            self.assertNotIn(raw_field, text)
        self.assertEqual(payload['five_second']['dominant_pct'], 42.1)
        self.assertEqual(payload['five_second']['peak_hz'], 10.25)
        self.assertEqual(payload['five_second']['bands']['Alpha']['power_uv2'], 12.346)
        self.assertEqual(payload['stable_estimate']['channels']['O1']['excluded_epochs'], {'flatline': 13})
        self.assertIsNone(payload['contact_ohms']['O2'])
        self.assertEqual(payload['transport']['received_hz'], 250.04)
        self.assertEqual(payload['device']['serial'], 'TESTONLY')

    def test_default_payload_handles_absent_inputs(self):
        payload = ai.build_payload()
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload['five_second'])
        self.assertIsNone(payload['stable_estimate'])
        self.assertIsNone(payload['device'])

    def test_prompt_rules_and_disclaimer(self):
        messages = ai.build_messages(sample_payload())
        self.assertEqual([message['role'] for message in messages], ['system', 'user'])
        system = messages[0]['content']
        for rule in ('Never diagnose', 'Never infer emotions', 'Never invent',
                     'not a medical interpretation'):
            self.assertIn(rule, system)
        self.assertTrue(system.rstrip().endswith(ai.DISCLAIMER))
        self.assertIn('Measurements JSON', messages[1]['content'])

    def test_successful_reply_and_request_shape(self):
        opener = opener_returning({'message': {'role': 'assistant',
                                               'content': '  Data status: invalid.\n' + ai.DISCLAIMER + '  '}})
        result = ai.ask({'five_second': {'valid': False}}, model='test-model', opener=opener)
        self.assertTrue(result['ok'])
        self.assertEqual(result['model'], 'test-model')
        self.assertTrue(result['text'].startswith('Data status: invalid.'))
        self.assertTrue(result['text'].endswith(ai.DISCLAIMER))
        request = opener.requests[0]
        self.assertEqual(request.get_method(), 'POST')
        self.assertTrue(request.full_url.endswith('/api/chat'))
        body = json.loads(request.data.decode('utf-8'))
        self.assertEqual(body['model'], 'test-model')
        self.assertFalse(body['stream'])
        self.assertFalse(body['think'])
        self.assertEqual([message['role'] for message in body['messages']], ['system', 'user'])

    def test_errors_never_raise_and_never_invent_text(self):
        def http_error(request, timeout=None):
            raise urllib.error.HTTPError('http://127.0.0.1:11434/api/chat', 404, 'Not Found',
                                         {}, io.BytesIO(b'{}'))
        result = ai.ask({}, opener=http_error)
        self.assertFalse(result['ok'])
        self.assertIn('404', result['error'])
        self.assertIn(ai.DEFAULT_MODEL, result['error'])
        self.assertNotIn('text', result)

        def refused(request, timeout=None):
            raise urllib.error.URLError('connection refused')
        result = ai.ask({}, opener=refused)
        self.assertFalse(result['ok'])
        self.assertIn('unreachable', result['error'])

        def unreadable(request, timeout=None):
            return FakeResponse(b'not json at all')
        result = ai.ask({}, opener=unreadable)
        self.assertFalse(result['ok'])
        self.assertIn('Unreadable', result['error'])

        opener = opener_returning({'message': {'role': 'assistant', 'content': '   '}})
        result = ai.ask({}, opener=opener)
        self.assertFalse(result['ok'])
        self.assertIn('empty', result['error'])

    def test_available_models_lists_local_names_and_survives_outage(self):
        opener = opener_returning({'models': [{'name': 'qwen3.8:latest'},
                                              {'name': 'llama3.2-vision:11b'}]})
        self.assertEqual(ai.available_models(opener=opener),
                         ['qwen3.8:latest', 'llama3.2-vision:11b'])

        def down(request, timeout=None):
            raise OSError('down')
        self.assertEqual(ai.available_models(opener=down), [])

    def test_payload_is_json_safe_with_hostile_inputs(self):
        payload = ai.build_payload(report={'valid': True, 'dominant_pct': float('nan'),
                                           'peak_hz': float('inf'), 'samples': 'x',
                                           'bands': {'Alpha': {'power_uv2': float('nan')}},
                                           'channels': {'O1': {'flags': [None, 3], 'excluded': 'bad'}}},
                                   contact={'O1': float('nan')})
        json.dumps(payload, allow_nan=False)
        self.assertIsNone(payload['five_second']['dominant_pct'])
        self.assertIsNone(payload['five_second']['peak_hz'])
        self.assertIsNone(payload['five_second']['bands']['Alpha']['power_uv2'])
        self.assertIsNone(payload['contact_ohms']['O1'])


if __name__ == '__main__':
    unittest.main()
