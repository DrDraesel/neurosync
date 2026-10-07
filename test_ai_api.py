"""API-backend tests for ai_doctor: OpenAI-compatible mode, fake responses only.

No network and no real API key is used — every response is faked, mirroring the
test_ai_doctor.py / test_live_chat.py conventions.
"""
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path

import ai_doctor as ai


class FakeResponse:
    """Non-streaming response stand-in."""

    def __init__(self, payload_bytes):
        self._payload = payload_bytes

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeStream:
    """Line-iterable response stand-in for SSE / NDJSON streams."""

    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b"".join(self._lines)

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


def opener_lines(lines):
    def opener(request, timeout=None):
        opener.requests.append(request)
        return FakeStream(lines)
    opener.requests = []
    return opener


def opener_error(exc):
    def opener(request, timeout=None):
        opener.calls += 1
        raise exc
    opener.calls = 0
    return opener


def sse(*chunks):
    lines = [('data: ' + json.dumps({'choices': [{'delta': {'content': c}}]})
              + '\n\n').encode('utf-8') for c in chunks]
    lines.append(b'data: [DONE]\n\n')
    return lines


class ApiMode:
    """Context manager: flip module config into API mode, restore after."""

    def __init__(self, url='http://127.0.0.1:9999/v1', key='test-key',
                 model='mock-model'):
        self.values = (url, key, model)

    def __enter__(self):
        self.old = (ai.AI_API_URL, ai.AI_API_KEY, ai.AI_API_MODEL)
        ai.AI_API_URL, ai.AI_API_KEY, ai.AI_API_MODEL = self.values
        return self

    def __exit__(self, *exc):
        ai.AI_API_URL, ai.AI_API_KEY, ai.AI_API_MODEL = self.old
        return False


class ModeTests(unittest.TestCase):
    def test_api_mode_flag_model_and_endpoint(self):
        with ApiMode():
            self.assertTrue(ai.api_mode())
            self.assertEqual(ai.effective_model(), 'mock-model')
            self.assertEqual(ai.endpoint_label(), 'http://127.0.0.1:9999/v1')
        self.assertFalse(ai.api_mode())
        self.assertEqual(ai.effective_model(), ai.DEFAULT_MODEL)
        self.assertEqual(ai.endpoint_label(), ai.DEFAULT_ENDPOINT)

    def test_available_models_reports_configured_api_model(self):
        with ApiMode():
            self.assertEqual(ai.available_models(), ['mock-model'])
        with ApiMode(model=''):
            self.assertEqual(ai.available_models(), [])

    def test_warm_is_noop_in_api_mode(self):
        with ApiMode():
            self.assertFalse(ai.warm())


class ApiAskTests(unittest.TestCase):
    def test_successful_api_reply_and_request_shape(self):
        opener = opener_returning({'choices': [{'message': {'content': 'api note'}}]})
        with ApiMode():
            result = ai.ask(ai.build_payload(), opener=opener)
        self.assertTrue(result['ok'])
        self.assertEqual(result['text'], 'api note')
        self.assertEqual(result['model'], 'mock-model')
        request = opener.requests[0]
        self.assertTrue(request.full_url.endswith('/v1/chat/completions'))
        self.assertEqual(request.headers.get('Authorization'), 'Bearer test-key')
        body = json.loads(request.data.decode('utf-8'))
        self.assertEqual(body['model'], 'mock-model')
        self.assertFalse(body['stream'])

    def test_api_http_error_names_status(self):
        http = urllib.error.HTTPError('http://x', 401, 'Unauthorized', None,
                                      io.BytesIO(b'{"error": "bad key"}'))
        with ApiMode():
            result = ai.ask(ai.build_payload(), opener=opener_error(http))
        self.assertFalse(result['ok'])
        self.assertIn('HTTP 401', result['error'])

    def test_api_unreachable_is_reported(self):
        with ApiMode():
            result = ai.ask(ai.build_payload(), opener=opener_error(OSError('refused')))
        self.assertFalse(result['ok'])
        self.assertIn('unreachable', result['error'])

    def test_api_empty_message_is_error(self):
        opener = opener_returning({'choices': [{'message': {'content': ' '}}]})
        with ApiMode():
            result = ai.ask(ai.build_payload(), opener=opener)
        self.assertFalse(result['ok'])
        self.assertIn('empty', result['error'])


class ApiStreamTests(unittest.TestCase):
    def test_api_stream_parses_sse(self):
        opener = opener_lines(sse('Hel', 'lo'))
        with ApiMode():
            events = list(ai.stream_ask([{'role': 'user', 'content': 'hi'}],
                                        opener=opener))
        self.assertEqual(events, [{'type': 'chunk', 'text': 'Hel'},
                                  {'type': 'chunk', 'text': 'lo'},
                                  {'type': 'done'}])
        request = opener.requests[0]
        self.assertTrue(request.full_url.endswith('/v1/chat/completions'))

    def test_api_stream_without_key_sends_no_auth(self):
        opener = opener_lines(sse('x'))
        with ApiMode(key=''):
            list(ai.stream_ask([{'role': 'user', 'content': 'hi'}], opener=opener))
        self.assertIsNone(opener.requests[0].headers.get('Authorization'))

    def test_api_stream_empty_reply_is_error(self):
        opener = opener_lines([b'data: [DONE]\n\n'])
        with ApiMode():
            events = list(ai.stream_ask([], opener=opener))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'error')
        self.assertIn('no answer text', events[0]['error'])

    def test_api_stream_midstream_failure_keeps_partial_text(self):
        class Exploding(FakeStream):
            def __iter__(self):
                def gen():
                    yield self._lines[0]
                    raise OSError('reset')
                return gen()
        raw = sse('A')

        def opener(request, timeout=None):
            return Exploding(raw)
        with ApiMode():
            events = list(ai.stream_ask([], opener=opener))
        self.assertEqual(events[0], {'type': 'chunk', 'text': 'A'})
        self.assertEqual(events[1]['type'], 'error')
        self.assertIn('stream failed', events[1]['error'])


class LocalDispatchTests(unittest.TestCase):
    def test_local_ask_still_hits_ollama_chat(self):
        opener = opener_returning({'message': {'content': 'local note'}})
        result = ai.ask(ai.build_payload(), opener=opener)
        self.assertTrue(result['ok'])
        self.assertTrue(opener.requests[0].full_url.endswith('/api/chat'))

    def test_local_stream_still_parses_ndjson(self):
        lines = [(json.dumps({'message': {'content': 'hi'}, 'done': True})
                  + '\n').encode('utf-8')]
        opener = opener_lines(lines)
        events = list(ai.stream_ask([{'role': 'user', 'content': 'x'}], opener=opener))
        self.assertEqual(events, [{'type': 'chunk', 'text': 'hi'}, {'type': 'done'}])
        self.assertTrue(opener.requests[0].full_url.endswith('/api/chat'))


class SettingsFileTests(unittest.TestCase):
    def test_settings_file_sets_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ai_settings.env'
            path.write_text('# comment\nAI_API_URL=https://example.com/v1\n'
                            'not a line\nAI_API_KEY="sekret"\n', encoding='utf-8')
            saved = {k: os.environ.pop(k, None) for k in ('AI_API_URL', 'AI_API_KEY')}
            try:
                ai._load_ai_settings(path)
                self.assertEqual(os.environ['AI_API_URL'], 'https://example.com/v1')
                self.assertEqual(os.environ['AI_API_KEY'], 'sekret')
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_real_environment_wins_over_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ai_settings.env'
            path.write_text('AI_API_URL=https://file.example/v1\n', encoding='utf-8')
            saved = os.environ.get('AI_API_URL')
            os.environ['AI_API_URL'] = 'https://env.example/v1'
            try:
                ai._load_ai_settings(path)
                self.assertEqual(os.environ['AI_API_URL'], 'https://env.example/v1')
            finally:
                if saved is None:
                    os.environ.pop('AI_API_URL', None)
                else:
                    os.environ['AI_API_URL'] = saved


if __name__ == '__main__':
    unittest.main()
