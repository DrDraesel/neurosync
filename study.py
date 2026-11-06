"""Study recording bookkeeping for NeuroSync — pure logic (no Qt, no storage).

A "study" is the full recording span: every data row between "Start study
(record)" and "Stop study", kept in a plain list (``self.study_rows`` in
``main.py``) instead of the rolling 60 s analysis deque. Rows are per-sample
channel lists; 250 Hz x 4 channels is 60,000 rows per minute. The hard cap is
``STUDY_MAX_SECONDS`` (60 minutes): when the cap is reached the study
auto-stops and the rows are saved exactly as recorded, with an explicit cap
note — the recording is never silently truncated and never loses the last
minute the way the deque does.

Honesty contract (do not weaken): the saved span is always stated as the
ACTUAL elapsed duration (monotonic clock) AND the actual row count, so a
transport gap shows up as a difference between the two instead of being
hidden. This module performs no analysis and touches no files; it only tracks
the span facts the UI and storage need.
"""
import time

STUDY_MAX_SECONDS = 3600.0          # 60-minute hard cap: auto-stop and save
CAP_MESSAGE = 'Study reached the 60-minute cap and was saved'


def subject_ready(subject) -> bool:
    """True when a subject with a non-empty name is set (phone stays optional)."""
    if not isinstance(subject, dict):
        return False
    name = subject.get('name')
    return bool(name) and bool(str(name).strip())


def can_start(subject, streaming) -> bool:
    """Start gate: a subject is set AND the device stream is live."""
    return bool(streaming) and subject_ready(subject)


def max_rows(fs) -> int:
    """Row count that equals the 60-minute cap at this sample rate."""
    try:
        rate = int(fs)
    except (TypeError, ValueError, OverflowError):
        return 0
    return int(rate * STUDY_MAX_SECONDS) if rate > 0 else 0


def elapsed_seconds(started_monotonic, now) -> float:
    """Actual elapsed study time in seconds (never negative)."""
    if started_monotonic is None:
        return 0.0
    return max(0.0, float(now) - float(started_monotonic))


def format_elapsed(seconds) -> str:
    """Visible 'MM:SS' elapsed time, floored: 754.9 -> '12:34'."""
    total = int(max(0.0, float(seconds)))
    minutes, secs = divmod(total, 60)
    return f'{minutes:02d}:{secs:02d}'


def status_label(seconds) -> str:
    """Running-study status, e.g. 'STUDY 12:34 · recording'."""
    return f'STUDY {format_elapsed(seconds)} · recording'


def reached_cap(rows, fs, started_monotonic, now) -> bool:
    """True once the study hit the 60-minute cap (by row count or elapsed time)."""
    return (len(rows) >= max_rows(fs)
            or elapsed_seconds(started_monotonic, now) >= STUDY_MAX_SECONDS)


def span_facts(rows, fs, started_monotonic, now) -> dict:
    """Honest recorded-span facts for the saved summary.

    ``recorded_span_s`` is the actual elapsed wall time (monotonic) and
    ``raw_rows`` the actual row count; they may differ slightly when the
    transport paused. ``capped`` is True when the study reached the cap.
    """
    return {'recorded_span_s': round(elapsed_seconds(started_monotonic, now), 3),
            'raw_rows': len(rows),
            'capped': reached_cap(rows, fs, started_monotonic, now)}
