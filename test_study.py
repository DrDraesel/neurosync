"""Pure-logic tests for study.py (study recording bookkeeping, no Qt).

Covers: subject gate validation, elapsed formatting, the 60-minute cap and
auto-stop trigger, span facts, and that study rows keep accumulating past the
60 s deque capacity (the plain list has no maxlen).
"""
import unittest

import study


class ElapsedFormattingTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(study.format_elapsed(0), '00:00')

    def test_documented_example(self):
        self.assertEqual(study.format_elapsed(754.9), '12:34')

    def test_floor_not_round(self):
        # 12:34 until the 35th second actually completes.
        self.assertEqual(study.format_elapsed(754.9), '12:34')
        self.assertEqual(study.format_elapsed(755.0), '12:35')

    def test_59_59(self):
        self.assertEqual(study.format_elapsed(3599.2), '59:59')

    def test_60_minutes(self):
        self.assertEqual(study.format_elapsed(3600), '60:00')

    def test_negative_never(self):
        self.assertEqual(study.format_elapsed(-5), '00:00')

    def test_over_an_hour(self):
        self.assertEqual(study.format_elapsed(3661), '61:01')


class StatusLabelTests(unittest.TestCase):
    def test_running_label(self):
        self.assertEqual(study.status_label(754), 'STUDY 12:34 · recording')

    def test_fresh_label(self):
        self.assertEqual(study.status_label(0), 'STUDY 00:00 · recording')


class SubjectGateTests(unittest.TestCase):
    def test_none_is_not_ready(self):
        self.assertFalse(study.subject_ready(None))

    def test_non_dict_is_not_ready(self):
        self.assertFalse(study.subject_ready('Maria'))

    def test_empty_name_is_not_ready(self):
        self.assertFalse(study.subject_ready({'name': ''}))
        self.assertFalse(study.subject_ready({'name': '   '}))

    def test_name_is_enough_phone_optional(self):
        self.assertTrue(study.subject_ready({'name': 'Maria'}))
        self.assertTrue(study.subject_ready({'name': 'Maria', 'phone': '+1 555'}))

    def test_can_start_needs_subject_and_streaming(self):
        subject = {'name': 'Maria'}
        self.assertTrue(study.can_start(subject, True))
        self.assertFalse(study.can_start(subject, False))   # not streaming
        self.assertFalse(study.can_start(None, True))       # no subject
        self.assertFalse(study.can_start({'name': ''}, True))


class CapTests(unittest.TestCase):
    def test_max_rows_at_250_hz(self):
        self.assertEqual(study.max_rows(250), 900_000)

    def test_max_rows_defensive(self):
        self.assertEqual(study.max_rows('x'), 0)
        self.assertEqual(study.max_rows(-5), 0)
        self.assertEqual(study.max_rows(None), 0)

    def test_elapsed_cap(self):
        self.assertTrue(study.reached_cap([], 250, 1000.0, 4600.0))   # 3600 s elapsed
        self.assertFalse(study.reached_cap([], 250, 1000.0, 4599.0))

    def test_row_count_cap(self):
        self.assertFalse(study.reached_cap([0] * 899_999, 250, 0.0, 1.0))
        self.assertTrue(study.reached_cap([0] * 900_000, 250, 0.0, 1.0))

    def test_elapsed_never_negative(self):
        self.assertEqual(study.elapsed_seconds(None, 5.0), 0.0)
        self.assertEqual(study.elapsed_seconds(10.0, 5.0), 0.0)

    def test_span_facts_are_honest(self):
        facts = study.span_facts([1, 2, 3], 250, 1000.0, 1054.25)
        self.assertEqual(facts['recorded_span_s'], 54.25)
        self.assertEqual(facts['raw_rows'], 3)
        self.assertFalse(facts['capped'])


class AccumulateBeyondBufferTests(unittest.TestCase):
    def test_study_rows_keep_accumulating_past_60_seconds_of_samples(self):
        # 61 s at 250 Hz = 15250 rows. The live analysis deque caps at 15000
        # (BUFFER_SECONDS * fs); a study plain list must keep every row.
        rows = []
        for index in range(15250):
            rows.append([float(index % 4), 0.0, 0.0, 0.0])
        self.assertGreater(len(rows), 15000)
        facts = study.span_facts(rows, 250, 100.0, 161.0)
        self.assertEqual(facts['raw_rows'], 15250)          # nothing dropped
        self.assertEqual(facts['recorded_span_s'], 61.0)
        self.assertFalse(facts['capped'])                   # 61 s is NOT 60 minutes


class CapMessageTests(unittest.TestCase):
    def test_exact_cap_message(self):
        # The auto-stop status note the UI shows; keep it verbatim.
        self.assertEqual(study.CAP_MESSAGE,
                         'Study reached the 60-minute cap and was saved')


if __name__ == '__main__':
    unittest.main()
