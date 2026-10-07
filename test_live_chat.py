"""Live-chat monitor tests: prompts, stream events, warm-up (no network, no GUI)."""
import json
import sys
import unittest
import urllib.error
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import ai_doctor as ai  # noqa: E402


class FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return b''.join(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ExplodingStream(FakeStream):
    """Yields the first line, then the connection dies mid-stream."""

    def __iter__(self):
        def gen():
            yield self._lines[0]
            raise OSError('connection reset')
        return gen()


def opener_stream(lines, cls=FakeStream):
    def opener(request, timeout=None):
        opener.requests.append(request)
        return cls(lines)
    opener.requests = []
    return opener


def ndjson(*events):
    return [(json.dumps(event) + '\n').encode('utf-8') for event in events]


def sample_payload():
    report = {'valid': True, 'ready': True, 'reason': 'ok', 'dominant': 'Alpha',
              'dominant_pct': 41.2, 'peak_hz': 10.1, 'samples': 1250,
              'bands': {'Alpha': {'power_uv2': 11.2, 'relative_pct': 41.2}},
              'channels': {'O1': {'dominant': 'Alpha', 'valid': True, 'flags': []}}}
    transport = {'samples': 9000, 'rate_hz': 250.0, 'gap_events': 0, 'battery': 90,
                 'mode': 'signal'}
    device = {'name': 'BrainBit', 'serial': 'TESTONLY', 'fs': 250, 'battery': 90,
              'device_label': 'SIMULATED BrainBit Classic', 'simulated': True}
    return ai.build_payload(report=report, stable=None, transport=transport,
                            device=device, contact=None)


class LiveMessageTests(unittest.TestCase):
    def test_system_prompt_keeps_the_hard_rules(self):
        messages = ai.build_live_messages(sample_payload())
        self.assertEqual(messages[0]['role'], 'system')
        system = messages[0]['content']
        for rule in ('live monitor', 'Never diagnose', 'Never infer emotions', 'Never invent',
                     'simulated: true', '60-120 words'):
            self.assertIn(rule, system)

    def test_update_message_carries_numbered_payload(self):
        messages = ai.build_live_messages(sample_payload(), update_number=3)
        self.assertEqual([m['role'] for m in messages], ['system', 'user'])
        last = messages[-1]['content']
        self.assertIn('Live update #3', last)
        self.assertIn('Measurements JSON', last)
        self.assertIn('"dominant":"Alpha"', last)

    def test_question_message_carries_question(self):
        messages = ai.build_live_messages(sample_payload(), question='Why is alpha so big?')
        self.assertIn('Operator question: Why is alpha so big?', messages[-1]['content'])
        self.assertIn('under 120 words', messages[-1]['content'])

    def test_history_is_cleaned_capped_and_bounded(self):
        history = [{'role': 'assistant', 'content': 'previous update'},
                   {'role': 'tool', 'content': 'dropped'},
                   {'role': 'user', 'content': 'x' * 5000}]
        messages = ai.build_live_messages(sample_payload(), history=history)
        mid = messages[1:-1]
        self.assertEqual([m['role'] for m in mid], ['assistant', 'user'])
        self.assertEqual(len(mid[1]['content']), 700)          # clipped
        many = [{'role': 'user', 'content': f'turn {i}'} for i in range(20)]
        messages = ai.build_live_messages(sample_payload(), history=many)
        self.assertEqual(len(messages), 1 + 6 + 1)             # last six kept
        self.assertEqual(messages[1]['content'], 'turn 14')
        json.dumps(messages, allow_nan=False)                  # JSON-safe by contract


class StreamAskTests(unittest.TestCase):
    def test_streams_chunks_then_done(self):
        opener = opener_stream(ndjson(
            {'message': {'role': 'assistant', 'content': 'Data '}, 'done': False},
            {'message': {'role': 'assistant', 'content': 'valid.'}, 'done': False},
            {'message': {'content': ''}, 'done': True}))
        events = list(ai.stream_ask([{'role': 'user', 'content': 'go'}],
                                    model='m', opener=opener))
        self.assertEqual(events[0], {'type': 'chunk', 'text': 'Data '})
        self.assertEqual(events[-1], {'type': 'done'})
        body = json.loads(opener.requests[0].data.decode('utf-8'))
        self.assertTrue(body['stream'])
        self.assertFalse(body['think'])
        self.assertEqual(body['keep_alive'], '30m')
        self.assertEqual(opener.requests[0].full_url, ai.DEFAULT_ENDPOINT + '/api/chat')
        self.assertEqual(opener.requests[0].get_method(), 'POST')

    def test_thinking_chunks_are_not_shown(self):
        opener = opener_stream(ndjson(
            {'message': {'thinking': 'reasoning…'}, 'done': False},
            {'message': {'content': 'Answer.'}, 'done': False},
            {'message': {'content': ''}, 'done': True}))
        events = list(ai.stream_ask([], opener=opener))
        self.assertEqual([e['text'] for e in events if e['type'] == 'chunk'], ['Answer.'])

    def test_no_text_reports_error(self):
        opener = opener_stream(ndjson({'message': {'thinking': 'hmm'}, 'done': False},
                                      {'message': {'content': ''}, 'done': True}))
        events = list(ai.stream_ask([], opener=opener))
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('no answer text', events[-1]['error'])

    def test_malformed_lines_are_skipped(self):
        lines = [b'{not json\n'] + ndjson({'message': {'content': 'ok'}, 'done': False},
                                          {'message': {'content': ''}, 'done': True})
        events = list(ai.stream_ask([], opener=opener_stream(lines)))
        self.assertEqual([e['text'] for e in events if e['type'] == 'chunk'], ['ok'])
        self.assertEqual(events[-1]['type'], 'done')

    def test_http_error_never_raises(self):
        def opener(request, timeout=None):
            raise urllib.error.HTTPError('http://127.0.0.1:11434/api/chat', 404,
                                         'Not Found', {}, None)
        events = list(ai.stream_ask([], model='missing', opener=opener))
        self.assertEqual(len(events), 1)
        self.assertIn('404', events[0]['error'])
        self.assertIn('missing', events[0]['error'])

    def test_unreachable_never_raises(self):
        def opener(request, timeout=None):
            raise urllib.error.URLError('connection refused')
        events = list(ai.stream_ask([], opener=opener))
        self.assertIn('unreachable', events[0]['error'])

    def test_midstream_break_reports_error_after_chunks(self):
        opener = opener_stream(ndjson({'message': {'content': 'part'}, 'done': False}),
                               cls=ExplodingStream)
        events = list(ai.stream_ask([], opener=opener))
        self.assertEqual(events[0], {'type': 'chunk', 'text': 'part'})
        self.assertEqual(events[1]['type'], 'error')
        self.assertIn('stream failed', events[1]['error'])


class WarmTests(unittest.TestCase):
    def test_warm_posts_pregenerate_and_survives_failure(self):
        opener = opener_stream(ndjson({'response': '', 'done': True}))
        self.assertTrue(ai.warm(model='m', opener=opener))
        body = json.loads(opener.requests[0].data.decode('utf-8'))
        self.assertEqual(body['model'], 'm')
        self.assertTrue(opener.requests[0].full_url.endswith('/api/generate'))

        def broken(request, timeout=None):
            raise OSError('down')
        self.assertFalse(ai.warm(model='m', opener=broken))


if __name__ == '__main__':
    unittest.main()
