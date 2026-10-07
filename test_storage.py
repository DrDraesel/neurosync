"""Tests for storage.py. Temporary directories only; no hardware, no Qt, no network."""
import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import storage

HEADER = ['sample_index', 'packet', 'O1', 'O2', 'T3', 'T4']
NAME = 'Jane Doe'
PHONE = '+1 (555) 010-2030'
KEY = 'jane_doe_15550102030'


def sample_rows(count=6):
    """TEST ONLY rows: mixed ints and floats, no device data."""
    return [[index, 100 + index, 1.5, 2.25, -3.5, 0.125] for index in range(count)]


def sample_session():
    """TEST ONLY summary payload shaped like the app's session dict."""
    return {
        'started_at': '2026-09-28T12:00:00+00:00',
        'device_type': 'BrainBit',
        'device_label': 'Classic 250 Hz',
        'channels': ['O1', 'O2', 'T3', 'T4'],
        'duration_s': 5.0,
        'notes': 'Test-only window',
        'bands': {
            'Alpha': {'low_hz': 8.0, 'high_hz': 13.0, 'power_uv2': 12.5, 'relative_pct': 40.0},
            'Beta': {'low_hz': 13.0, 'high_hz': 30.0, 'power_uv2': 6.25, 'relative_pct': 20.0},
        },
        'quality_flags': {'O1': [], 'O2': ['flatline'], 'T3': [], 'T4': []},
    }


def read_summary(folder):
    return json.loads((Path(folder) / 'summary.json').read_text(encoding='utf-8'))


def craft_session(root, key, session_id, summary, name=None, phone=None):
    """Hand-written session folder used to test listing rules in isolation."""
    folder = Path(root) / key / session_id
    folder.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if name is not None:
        payload['subject_name'] = name
    if phone is not None:
        payload['subject_phone'] = phone
    payload.setdefault('session_id', session_id)
    (folder / 'summary.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return folder


class StorageTests(unittest.TestCase):
    def test_public_api_surface(self):
        self.assertEqual(storage.DEFAULT_ROOT,
                         Path(storage.__file__).resolve().parent / 'recordings' / 'subjects')
        self.assertEqual(storage.MAX_RAW_ROWS, 200_000)
        for name in ('slug', 'subject_key', 'subject_dir', 'session_dir', 'save_session',
                     'list_subjects', 'list_sessions', 'load_session'):
            self.assertTrue(callable(getattr(storage, name)), name)

    def test_slug_edge_cases(self):
        cases = {
            '': 'subject',
            '   ': 'subject',
            '!!!': 'subject',
            'José María!': 'jose_maria',
            '  Foo   Bar__baz  ': 'foo_bar_baz',
            'MiXeD 123': 'mixed_123',
            'a-b': 'a_b',
            'UPPER': 'upper',
            '2026': '2026',
            'a' * 50: 'a' * 40,
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(storage.slug(value), expected)
        self.assertEqual(storage.slug('a' * 50, max_len=10), 'a' * 10)
        self.assertEqual(storage.slug('abcdef', max_len=6), 'abcdef')
        self.assertEqual(storage.slug('ab_cd_ef', max_len=3), 'ab')
        self.assertEqual(len(storage.slug('x' * 100, max_len=12)), 12)

    def test_subject_key_uses_phone_digits_only(self):
        self.assertEqual(storage.subject_key(NAME, PHONE), KEY)
        self.assertEqual(storage.subject_key(NAME, PHONE), 'jane_doe_15550102030')
        self.assertEqual(storage.subject_key(NAME, ''), 'jane_doe_nophone')
        self.assertEqual(storage.subject_key(NAME, None), 'jane_doe_nophone')
        self.assertEqual(storage.subject_key(NAME, 'no digits here'), 'jane_doe_nophone')
        self.assertEqual(storage.subject_key(NAME, 5551234), 'jane_doe_5551234')
        self.assertEqual(storage.subject_key('   ', ''), 'subject_nophone')
        self.assertEqual(storage.subject_key('phone only', '+99-000'), 'phone_only_99000')

    def test_save_list_reload_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            session = sample_session()
            rows = sample_rows()
            folder = storage.save_session(root, NAME, PHONE, session, HEADER, rows)

            self.assertTrue(folder.is_dir())
            self.assertEqual(folder.parent, root / KEY)
            self.assertEqual(folder.parent.name, storage.slug(NAME) + '_15550102030')

            with open(folder / 'raw.csv', newline='', encoding='utf-8') as handle:
                parsed = list(csv.reader(handle))
            self.assertEqual(parsed[0], HEADER)
            self.assertEqual(len(parsed), len(rows) + 1)
            self.assertEqual(parsed[1], ['0', '100', '1.5', '2.25', '-3.5', '0.125'])

            summary = read_summary(folder)
            self.assertEqual(summary['bands'], session['bands'])
            self.assertEqual(summary['device_type'], 'BrainBit')
            self.assertIn('local-only', summary['storage_note'].lower())
            self.assertEqual(summary['session_id'], folder.name)
            self.assertEqual(summary['subject_name'], NAME)
            self.assertEqual(summary['subject_phone'], PHONE)
            self.assertEqual(summary['subject_key'], KEY)
            self.assertEqual(summary['raw_csv'], 'raw.csv')
            self.assertEqual(summary['charts'], [])
            self.assertEqual(summary['storage_version'], 1)
            self.assertEqual(datetime.fromisoformat(summary['saved_at']).utcoffset().total_seconds(), 0)
            self.assertFalse((folder / 'summary.json.tmp').exists())
            text = (folder / 'summary.json').read_text(encoding='utf-8')
            self.assertTrue(text.splitlines()[1].startswith('  '))
            self.assertFalse(text.splitlines()[1].startswith('   '))
            self.assertEqual(session['device_type'], 'BrainBit')
            self.assertNotIn('session_id', session)

            subjects = storage.list_subjects(root)
            self.assertEqual(len(subjects), 1)
            subject = subjects[0]
            self.assertEqual(subject['name'], NAME)
            self.assertEqual(subject['phone'], PHONE)
            self.assertEqual(subject['key'], KEY)
            self.assertEqual(subject['path'], root / KEY)
            self.assertEqual(subject['sessions'], 1)
            self.assertEqual(subject['last_session_at'], summary['saved_at'])

            sessions = storage.list_sessions(root, NAME, PHONE)
            self.assertEqual(len(sessions), 1)
            item = sessions[0]
            self.assertEqual(set(item), {'session_id', 'path', 'started_at', 'device_type',
                                        'device_label', 'channels', 'duration_s', 'notes',
                                        'charts'})
            self.assertEqual(item['session_id'], folder.name)
            self.assertEqual(item['path'], folder)
            self.assertEqual(item['started_at'], session['started_at'])
            self.assertEqual(item['device_type'], 'BrainBit')
            self.assertEqual(item['device_label'], 'Classic 250 Hz')
            self.assertEqual(item['channels'], ['O1', 'O2', 'T3', 'T4'])
            self.assertEqual(item['duration_s'], 5.0)
            self.assertEqual(item['notes'], 'Test-only window')
            self.assertEqual(item['charts'], [])

            loaded = storage.load_session(folder)
            self.assertEqual(set(loaded), {'summary', 'raw_path', 'raw_header',
                                           'raw_rows', 'charts'})
            self.assertEqual(loaded['summary']['bands'], session['bands'])
            self.assertEqual(loaded['summary']['device_type'], 'BrainBit')
            self.assertEqual(loaded['raw_path'], folder / 'raw.csv')
            self.assertEqual(loaded['raw_header'], HEADER)
            self.assertEqual(len(loaded['raw_rows']), len(rows))
            self.assertEqual(loaded['raw_rows'][0], parsed[1])
            self.assertEqual(loaded['raw_rows'][-1][0], str(len(rows) - 1))
            self.assertEqual(loaded['charts'], [])

    def test_second_session_bumps_count_and_sessions_are_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            first = datetime(2026, 1, 2, 3, 4, 5, 678901)
            second = datetime(2026, 2, 3, 4, 5, 6, 789012)
            folder_one = storage.save_session(root, NAME, PHONE, sample_session(),
                                              HEADER, sample_rows(), timestamp=first)
            folder_two = storage.save_session(root, NAME, PHONE, sample_session(),
                                              HEADER, sample_rows(), timestamp=second)
            self.assertEqual(folder_one.name, '20260102_030405_678901')
            self.assertEqual(folder_two.name, '20260203_040506_789012')
            self.assertEqual(folder_one.parent, folder_two.parent)

            subjects = storage.list_subjects(root)
            self.assertEqual(len(subjects), 1)
            self.assertEqual(subjects[0]['sessions'], 2)
            stamps = [read_summary(folder_one)['saved_at'], read_summary(folder_two)['saved_at']]
            self.assertEqual(subjects[0]['last_session_at'], max(stamps))

            sessions = storage.list_sessions(root, NAME, PHONE)
            self.assertEqual([item['path'].name for item in sessions],
                             [folder_two.name, folder_one.name])
            self.assertEqual([item['session_id'] for item in sessions],
                             [folder_two.name, folder_one.name])

    def test_same_timestamp_gets_unique_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            stamp = datetime(2026, 3, 4, 5, 6, 7, 890123)
            folder_one = storage.save_session(root, NAME, PHONE, sample_session(),
                                              HEADER, sample_rows(), timestamp=stamp)
            folder_two = storage.save_session(root, NAME, PHONE, sample_session(),
                                              HEADER, sample_rows(), timestamp=stamp)
            self.assertEqual(folder_two.name, f'{folder_one.name}_2')
            next_folder = storage.session_dir(root, NAME, PHONE, stamp)
            self.assertEqual(next_folder.name, f'{folder_one.name}_3')
            self.assertFalse(next_folder.exists())
            self.assertEqual(len(storage.list_sessions(root, NAME, PHONE)), 2)
            self.assertEqual(storage.list_subjects(root)[0]['sessions'], 2)

    def test_two_subjects_are_separated_and_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            storage.save_session(root, NAME, PHONE, sample_session(), HEADER, sample_rows())
            storage.save_session(root, 'Bob Smith', '', sample_session(), HEADER, sample_rows())

            self.assertEqual(sorted(entry.name for entry in root.iterdir()),
                             ['bob_smith_nophone', KEY])
            subjects = storage.list_subjects(root)
            self.assertEqual(len(subjects), 2)
            self.assertEqual(sorted(subject['name'] for subject in subjects),
                             ['Bob Smith', NAME])
            for subject in subjects:
                self.assertEqual(subject['sessions'], 1)
                self.assertTrue(subject['last_session_at'])
            self.assertEqual(len(storage.list_sessions(root, NAME, PHONE)), 1)
            self.assertEqual(len(storage.list_sessions(root, 'Bob Smith', '')), 1)
            self.assertEqual(storage.list_sessions(root, 'Nobody Here', ''), [])

    def test_chart_name_and_payload_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            bad_names = ['../evil.png', 'x.jpg', 'a/b.png', 'a\\b.png', '', 'evil.png ',
                         'C:evil.png', '.']
            for chart_name in bad_names:
                with self.subTest(chart_name=chart_name):
                    with self.assertRaises(ValueError):
                        storage.save_session(root, NAME, PHONE, sample_session(), HEADER,
                                             sample_rows(), charts={chart_name: b'x'})
            with self.assertRaises(ValueError):
                storage.save_session(root, NAME, PHONE, sample_session(), HEADER,
                                     sample_rows(), charts={'ok.png': 'not bytes'})
            self.assertFalse(storage.subject_dir(root, NAME, PHONE).exists())
            self.assertFalse(root.exists())

    def test_chart_is_written_listed_and_absent_folder_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            payload = b'\x89PNG\r\n\x1a\nTEST-ONLY'
            folder = storage.save_session(
                root, NAME, PHONE, sample_session(), HEADER, sample_rows(),
                charts={'zeta.png': b'z', 'alpha.png': payload})
            chart_path = folder / 'charts' / 'alpha.png'
            self.assertTrue(chart_path.is_file())
            self.assertEqual(chart_path.read_bytes(), payload)
            self.assertEqual((folder / 'charts' / 'zeta.png').read_bytes(), b'z')
            self.assertEqual(read_summary(folder)['charts'], ['alpha.png', 'zeta.png'])
            loaded = storage.load_session(folder)
            self.assertEqual(loaded['charts'], [folder / 'charts' / 'alpha.png',
                                                folder / 'charts' / 'zeta.png'])
            listed = storage.list_sessions(root, NAME, PHONE)[0]
            self.assertEqual(listed['charts'], ['alpha.png', 'zeta.png'])

            plain = storage.save_session(root, NAME, PHONE, sample_session(), HEADER,
                                         sample_rows())
            self.assertFalse((plain / 'charts').exists())
            self.assertEqual(read_summary(plain)['charts'], [])

    def test_malformed_session_raises_and_leaves_nothing_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            malformed = [['not', 'a', 'dict'], {'note': object()}, {'bands': {1, 2}},
                         {'blob': b'bytes'}, {'power': float('nan')},
                         {'power': float('inf')}]
            for session in malformed:
                with self.subTest(session=repr(session)):
                    with self.assertRaises(ValueError):
                        storage.save_session(root, NAME, PHONE, session, HEADER, sample_rows())
            self.assertFalse(root.exists())
            self.assertFalse(storage.subject_dir(root, NAME, PHONE).exists())
            self.assertEqual(storage.list_subjects(root), [])
            self.assertEqual(storage.list_sessions(root, NAME, PHONE), [])

    def test_missing_root_and_missing_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nowhere'
            self.assertEqual(storage.list_subjects(root), [])
            self.assertEqual(storage.list_sessions(root, NAME, PHONE), [])
            with self.assertRaises(FileNotFoundError):
                storage.load_session(root)
            empty = Path(tmp) / 'empty'
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                storage.load_session(empty)

            summary_only = Path(tmp) / 'summary_only'
            summary_only.mkdir()
            (summary_only / 'summary.json').write_text(
                json.dumps({'session_id': 'x'}, indent=2), encoding='utf-8')
            loaded = storage.load_session(summary_only)
            self.assertIsNone(loaded['raw_path'])
            self.assertEqual(loaded['raw_header'], [])
            self.assertEqual(loaded['raw_rows'], [])
            self.assertEqual(loaded['charts'], [])
            self.assertNotIn('raw_truncated', loaded)

    def test_raw_rows_are_capped_with_truncation_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            folder = storage.save_session(root, NAME, PHONE, sample_session(), HEADER,
                                          sample_rows(10), timestamp=datetime(2026, 4, 5))
            with mock.patch.object(storage, 'MAX_RAW_ROWS', 3):
                loaded = storage.load_session(folder)
                self.assertEqual(len(loaded['raw_rows']), 3)
                self.assertIs(loaded['raw_truncated'], True)
                self.assertEqual(loaded['raw_header'], HEADER)
            exact = storage.save_session(root, NAME, PHONE, sample_session(), HEADER,
                                         sample_rows(3), timestamp=datetime(2026, 4, 6))
            with mock.patch.object(storage, 'MAX_RAW_ROWS', 3):
                loaded = storage.load_session(exact)
            self.assertEqual(len(loaded['raw_rows']), 3)
            self.assertNotIn('raw_truncated', loaded)

    def test_subject_folder_without_summary_is_still_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            folder = root / 'old_person_5551234'
            folder.mkdir(parents=True)
            (folder / 'sessionish').mkdir()
            (folder / 'notes.txt').write_text('test only', encoding='utf-8')
            subjects = storage.list_subjects(root)
            self.assertEqual(len(subjects), 1)
            self.assertEqual(subjects[0]['name'], 'old_person')
            self.assertEqual(subjects[0]['phone'], '5551234')
            self.assertEqual(subjects[0]['key'], 'old_person_5551234')
            self.assertEqual(subjects[0]['sessions'], 0)
            self.assertIsNone(subjects[0]['last_session_at'])
            self.assertEqual(storage.list_sessions(root, 'old_person', '5551234'), [])

    def test_subject_order_is_last_session_desc_then_name_with_none_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            craft_session(root, storage.subject_key('Zara', '111'), '20260101_000000_000000',
                          {'saved_at': '2026-05-01T00:00:00+00:00', 'duration_s': 5.0},
                          name='Zara', phone='111')
            craft_session(root, storage.subject_key('Anna', '222'), '20260101_000000_000000',
                          {'saved_at': '2026-06-01T00:00:00+00:00'},
                          name='Anna', phone='222')
            craft_session(root, storage.subject_key('Ann', '333'), '20260101_000000_000000',
                          {'saved_at': '2026-06-01T00:00:00+00:00'},
                          name='Ann', phone='333')
            (root / storage.subject_key('nobody', '555')).mkdir(parents=True)

            subjects = storage.list_subjects(root)
            self.assertEqual([subject['name'] for subject in subjects],
                             ['Ann', 'Anna', 'Zara', 'nobody'])
            self.assertEqual([subject['phone'] for subject in subjects],
                             ['333', '222', '111', '555'])
            self.assertEqual([subject['sessions'] for subject in subjects], [1, 1, 1, 0])
            self.assertIsNone(subjects[-1]['last_session_at'])
            self.assertEqual(subjects[0]['last_session_at'], '2026-06-01T00:00:00+00:00')
            self.assertEqual(subjects[1]['path'].name, storage.subject_key('Anna', '222'))

            listed = storage.list_sessions(root, 'Anna', '222')
            self.assertEqual(len(listed), 1)
            self.assertEqual(listed[0]['started_at'], None)
            self.assertEqual(listed[0]['device_type'], None)
            self.assertEqual(listed[0]['channels'], [])
            self.assertEqual(listed[0]['charts'], [])
            self.assertEqual(listed[0]['session_id'], '20260101_000000_000000')
            self.assertIsNone(listed[0]['duration_s'])
            zara = storage.list_sessions(root, 'Zara', '111')
            self.assertEqual(zara[0]['duration_s'], 5.0)
            self.assertEqual(zara[0]['path'].parent.name, storage.subject_key('Zara', '111'))

    def test_user_input_cannot_escape_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'subjects'
            escaped = [('../../etc', '..'), ('..', '../..'), ('a/b/c', '5 5'),
                       ('C:\\Windows', '123'), ('....//x', '99')]
            for name, phone in escaped:
                with self.subTest(name=name, phone=phone):
                    key = storage.subject_key(name, phone)
                    self.assertNotIn('/', key)
                    self.assertNotIn('\\', key)
                    self.assertNotIn('..', key)
                    subject = storage.subject_dir(root, name, phone)
                    session = storage.session_dir(root, name, phone)
                    for path in (subject, session):
                        self.assertTrue(storage._inside(root, path))
                        relative = path.resolve().relative_to(root.resolve())
                        self.assertEqual(relative.parts[0], key)
                    self.assertEqual(session.parent, subject)
                    self.assertEqual(subject.name, key)
            self.assertEqual(storage.subject_key('../../etc', '..'), 'etc_nophone')
            self.assertEqual(storage.subject_key('a/b/c', '5 5'), 'a_b_c_55')

            with self.assertRaises(ValueError):
                storage._checked(root, Path(tmp) / 'outside')
            with self.assertRaises(ValueError):
                storage._checked(root, root / '..' / 'escape')
            self.assertFalse(storage._inside(root, Path(tmp)))
            self.assertTrue(storage._inside(root, root / 'deep' / 'child'))


if __name__ == '__main__':
    unittest.main()
