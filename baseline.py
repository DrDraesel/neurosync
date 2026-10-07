"""Per-subject baseline ("predominant-wave profile"): capture, storage, delta reporting.

A baseline is ONE quality-gated analysis of one subject, stored as
``recordings/subjects/<slug(name)>_<phone>/baseline.json`` and used afterwards as
that subject's own descriptive reference. It records the mean absolute band
powers (uV^2), the mean relative shares (% of 1..45 Hz), the largest band
(``dominant``), the peak frequency of the mean density between 1 and 45 Hz, a
per-channel dominant/peak table, the measurement condition the user picked
('eyes closed', 'eyes open' or 'other'), device, channel source, quality
information and the app version.

Honesty rules (see METHODS.md). Everything here describes a measured EEG
pattern of one short recording. It is not a diagnosis, not a medical device and
NOT a measurement of mood, attention, emotion, personality or sleep. The
condition (eyes open/closed) changes what the numbers mean, which is why it is
required at capture time and always displayed with the profile; only records
taken in the same condition should be compared. A single optional interpretive
sentence is allowed and produced by ``interpretive_note`` in exactly this
pattern: pattern description, then "not an assessment of the person", then the
literature caveat for eyes-closed alpha dominance.

Storage is local only: nothing is uploaded, ``baseline.json`` is written with
the same atomic temp-file/fsync/replace path and ``allow_nan=False`` check as
``summary.json``, and every path is re-checked to stay inside the recordings
root (``storage.subject_dir``). Repeated captures REPLACE the previous baseline
of that subject; sessions are never touched.
"""
import math
import numbers
from datetime import datetime, timezone
from pathlib import Path

import storage
from eeg_analysis import BANDS

APP_VERSION = '1.1.0'
BASELINE_FILE = 'baseline.json'
BASELINE_VERSION = 1
KIND = 'predominant-wave profile'
CONDITIONS = ('eyes closed', 'eyes open', 'other')
SOURCES = ('stable', 'five_second')
SOURCE_LABELS = {
    'stable': 'stable multi-epoch estimate',
    'five_second': 'single five-second window',
}
REASONS = {
    'no_baseline': 'no baseline set for this subject',
    'no_current_estimate': 'no current estimate yet (wait for a passing five-second window)',
    'current_estimate_not_valid': 'the current estimate is not passing the quality gates',
    'no_comparable_band_numbers': 'no comparable band numbers in the current estimate',
}
BASELINE_NOTE = ('Local-only reference profile of this subject\'s own measurements: descriptive '
                 'engineering numbers, not a diagnosis.')
CONDITION_NOTE = ('Condition changes interpretation: compare only measurements taken in the same '
                  'condition (eyes closed / eyes open / other).')
DELTA_NOTE = ('Descriptive differences of measured band numbers against this subject\'s baseline. '
              'Electrode placement, contact and movement change these numbers too; not a '
              'diagnosis and not a mental-state measurement.')
LIMITS = ('Short windows and repeated electrode placements also change these numbers. Descriptive '
          'EEG pattern numbers only: not a diagnosis, and not a measurement of mood, attention, '
          'emotion or sleep.')
INTERPRETIVE_TAIL = 'not an assessment of the person'
LITERATURE_CLAUSE = ('in the literature, eyes-closed alpha dominance is commonly associated with '
                     'relaxed wakefulness.')


def _number(value):
    """Finite float of ``value``, else None (booleans and NaN are not numbers)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _round(value, places=3):
    number = _number(value)
    return None if number is None else round(number, places)


def _count(value):
    """Non-negative integer count of ``value``, else None."""
    number = _number(value)
    if number is None or number < 0:
        return None
    return int(number)


def _text(value):
    return value if isinstance(value, str) and value.strip() else None


def normalize_condition(value) -> str:
    """The captured condition, exactly one of ``CONDITIONS`` (surrounding spaces stripped).

    A baseline without a condition cannot be interpreted, so a missing or
    unknown value raises ValueError instead of being stored as blank.
    """
    allowed = ', '.join(CONDITIONS)
    if not isinstance(value, str):
        raise ValueError(f'condition is required and must be one of: {allowed}')
    text = value.strip()
    if text not in CONDITIONS:
        raise ValueError(f'condition is required and must be one of: {allowed}')
    return text


def pattern_description(record) -> str:
    """Short EEG pattern description from a baseline record ('' when unusable).

    Example: 'Alpha-dominant rhythm, peak 10.2 Hz, 43%'. Names a measured
    pattern only; it says nothing about the person.
    """
    profile = record.get('profile') if isinstance(record, dict) else None
    if not isinstance(profile, dict):
        return ''
    dominant = _text(profile.get('dominant'))
    if dominant is None:
        return ''
    parts = [f'{dominant}-dominant rhythm' if dominant != 'mixed'
             else 'mixed measured bands (top two within 5 percentage points)']
    peak = _number(profile.get('peak_hz'))
    if peak is not None:
        parts.append(f'peak {peak:.1f} Hz')
    share = _number(profile.get('dominant_pct'))
    if share is not None:
        parts.append(f'{share:.0f}%')
    return ', '.join(parts)


def profile_line(record) -> str:
    """One-line baseline profile, e.g.
    'Baseline: Alpha-dominant, peak 10.2 Hz, 43% (eyes closed) - set 2026-09-28, BrainBit Classic'.
    """
    if not isinstance(record, dict):
        return 'No baseline set'
    profile = record.get('profile') if isinstance(record.get('profile'), dict) else {}
    dominant = _text(profile.get('dominant')) or 'unknown'
    parts = [f'Baseline: {dominant}-dominant' if dominant != 'mixed'
             else 'Baseline: mixed measured bands (no single largest band)']
    peak = _number(profile.get('peak_hz'))
    if peak is not None:
        parts.append(f'peak {peak:.1f} Hz')
    share = _number(profile.get('dominant_pct'))
    if share is not None:
        parts.append(f'{share:.0f}%')
    condition = _text(record.get('condition')) or 'condition not recorded'
    head = ', '.join(parts) + f' ({condition})'
    set_at = _text(record.get('set_at')) or ''
    device = _text(record.get('device_label')) or _text(record.get('device_type'))
    tail = ' - set ' + (set_at[:10] or 'date not recorded')
    if device:
        tail += f', {device}'
    source = profile.get('source') if isinstance(profile, dict) else None
    if source in SOURCE_LABELS:
        tail += f' - from the {SOURCE_LABELS[source]}'
    return head + tail


def interpretive_note(record) -> str:
    """The one allowed optional interpretive sentence ('' when no usable record).

    Always of the form '<EEG pattern description> - EEG pattern description,
    not an assessment of the person' and, only for an eyes-closed alpha-dominant
    baseline, followed by the literature caveat about eyes-closed alpha
    dominance. No other interpretation is produced anywhere in this module.
    """
    description = pattern_description(record)
    if not description:
        return ''
    profile = record.get('profile') if isinstance(record.get('profile'), dict) else {}
    tail = f'EEG pattern description, {INTERPRETIVE_TAIL}'
    if record.get('condition') == 'eyes closed' and profile.get('dominant') == 'Alpha':
        return f'{description} - {tail}; {LITERATURE_CLAUSE}'
    return f'{description} - {tail}.'


def profile_from_report(report, source='stable') -> dict:
    """Descriptive profile extracted from one quality-gated analysis report.

    Only a PASSING report can become a baseline: ``report['valid']`` must be
    true and every band must carry finite absolute (uV^2) and relative (%) power,
    otherwise ValueError is raised (a baseline is a reference, not a placeholder).
    Per-channel entries keep only dominant/peak/validity, never raw samples.
    """
    if source not in SOURCES:
        raise ValueError(f'source must be one of: {", ".join(SOURCES)}')
    if not isinstance(report, dict):
        raise ValueError('a report dict from analyze_window/analyze_stable is required')
    if not report.get('valid'):
        reason = _text(report.get('reason')) or 'unknown'
        raise ValueError('only a quality-gated estimate can become a baseline '
                         f'(current status: {reason})')
    bands = report.get('bands')
    if not isinstance(bands, dict):
        raise ValueError('the estimate carries no band powers')
    profile_bands = {}
    for name in BANDS:
        band = bands.get(name) if isinstance(bands.get(name), dict) else {}
        relative = _number(band.get('relative_pct'))
        power = _number(band.get('power_uv2'))
        if relative is None or power is None:
            raise ValueError(f'the estimate has no usable {name} band numbers')
        profile_bands[name] = {'power_uv2': power, 'relative_pct': relative}
    channels = report.get('channels') if isinstance(report.get('channels'), dict) else {}
    table = {}
    for name, entry in channels.items():
        entry = entry if isinstance(entry, dict) else {}
        table[str(name)] = {
            'dominant': _text(entry.get('dominant')),
            'peak_hz': _round(entry.get('peak_hz'), 3),
            'valid': bool(entry.get('valid')),
            'flags': [str(flag) for flag in (entry.get('flags') or [])][:8],
        }
    dominant = _text(report.get('dominant'))
    return {
        'source': source,
        'source_label': SOURCE_LABELS[source],
        'bands': profile_bands,
        'dominant': dominant,
        'dominant_pct': _round(report.get('dominant_pct'), 3),
        'peak_hz': _round(report.get('peak_hz'), 3),
        'channels': table,
        'quality': {
            'valid': True,
            'reason': _text(report.get('reason')) or '',
            'samples': _count(report.get('samples')),
            'duration_s': _round(report.get('duration_s'), 3),
            'epochs_total': _count(report.get('epochs_total')),
            'epochs_used': _count(report.get('epochs_used')),
            'seconds_used': _round(report.get('seconds_used'), 3),
        },
    }


def validate_baseline(payload) -> dict:
    """Validated, normalised copy of a baseline record; ValueError when unusable.

    Used both before writing and after reading, so a hand-edited or truncated
    ``baseline.json`` is reported as 'not a usable baseline' instead of being
    silently treated as a measurement.
    """
    if not isinstance(payload, dict):
        raise ValueError('baseline must be a JSON object')
    condition = normalize_condition(payload.get('condition'))
    set_at = _text(payload.get('set_at'))
    if set_at is None:
        raise ValueError('baseline must carry a set_at timestamp')
    profile = payload.get('profile')
    if not isinstance(profile, dict):
        raise ValueError('baseline must carry a profile object')
    bands = profile.get('bands')
    if not isinstance(bands, dict) or not bands:
        raise ValueError('baseline profile must carry band numbers')
    clean_bands = {}
    for name, band in bands.items():
        band = band if isinstance(band, dict) else {}
        relative = _number(band.get('relative_pct'))
        power = _number(band.get('power_uv2'))
        if relative is None or power is None:
            raise ValueError(f'baseline {name} band has no usable numbers')
        clean_bands[str(name)] = {'power_uv2': power, 'relative_pct': relative}
    channels = profile.get('channels') if isinstance(profile.get('channels'), dict) else {}
    clean_channels = {}
    for name, entry in channels.items():
        entry = entry if isinstance(entry, dict) else {}
        clean_channels[str(name)] = {
            'dominant': _text(entry.get('dominant')),
            'peak_hz': _round(entry.get('peak_hz'), 3),
            'valid': bool(entry.get('valid')),
            'flags': [str(flag) for flag in (entry.get('flags') or [])][:8],
        }
    source = profile.get('source') if profile.get('source') in SOURCES else None
    record = dict(payload)
    record.update({
        'baseline_version': int(_number(payload.get('baseline_version')) or BASELINE_VERSION),
        'kind': _text(payload.get('kind')) or KIND,
        'set_at': set_at,
        'condition': condition,
        'subject_name': '' if payload.get('subject_name') is None else str(payload.get('subject_name')),
        'subject_phone': '' if payload.get('subject_phone') is None else str(payload.get('subject_phone')),
        'subject_key': _text(payload.get('subject_key')) or '',
        'device_type': _text(payload.get('device_type')),
        'device_label': _text(payload.get('device_label')),
        'channel_source': _text(payload.get('channel_source')) or 'unknown',
        'channels': [str(name) for name in (payload.get('channels') or [])] or list(clean_channels),
        'fs_hz': _round(payload.get('fs_hz'), 3),
        'session_id': _text(payload.get('session_id')),
        'captured_from': payload.get('captured_from') if payload.get('captured_from') in
                         ('live', 'saved_session') else 'live',
        'app_version': _text(payload.get('app_version')) or 'unknown',
        'baseline_note': BASELINE_NOTE,
        'condition_note': CONDITION_NOTE,
        'limits': LIMITS,
    })
    record['profile'] = {
        'source': source,
        'source_label': SOURCE_LABELS[source] if source else None,
        'bands': clean_bands,
        'dominant': _text(profile.get('dominant')),
        'dominant_pct': _round(profile.get('dominant_pct'), 3),
        'peak_hz': _round(profile.get('peak_hz'), 3),
        'channels': clean_channels,
        'quality': dict(profile.get('quality')) if isinstance(profile.get('quality'), dict) else {},
    }
    return record


def build_baseline(report, condition, *, subject_name='', subject_phone='', device=None,
                   channel_source=None, channels=None, fs_hz=None, source='stable',
                   session_id=None, captured_from='live', captured_at=None,
                   app_version=APP_VERSION) -> dict:
    """Validated baseline record built from one passing estimate + the picked condition.

    ``condition`` is required and stored exactly as one of CONDITIONS. The
    estimate must be passing (see ``profile_from_report``). ``source`` records
    which estimate was used ('stable' preferred, 'five_second' fallback) and is
    displayed with the profile so a weaker reference is never hidden.
    """
    profile = profile_from_report(report, source)
    condition = normalize_condition(condition)
    device = device if isinstance(device, dict) else {}
    when = captured_at or datetime.now(timezone.utc)
    if not isinstance(when, datetime):
        raise ValueError('captured_at must be a datetime')
    record = {
        'baseline_version': BASELINE_VERSION,
        'kind': KIND,
        'set_at': when.astimezone(timezone.utc).isoformat(),
        'condition': condition,
        'subject_name': '' if subject_name is None else str(subject_name),
        'subject_phone': '' if subject_phone is None else str(subject_phone),
        'subject_key': storage.subject_key(subject_name or '', subject_phone or ''),
        'device_type': _text(device.get('device_key')) or _text(device.get('device_type')),
        'device_label': _text(device.get('device_label')),
        'channel_source': (_text(channel_source) or _text(device.get('channel_source'))
                           or 'unknown'),
        'channels': [str(name) for name in (channels if channels is not None
                                            else profile.get('channels') or [])],
        'fs_hz': _round(fs_hz if fs_hz is not None else device.get('fs'), 3),
        'session_id': _text(session_id),
        'captured_from': captured_from if captured_from in ('live', 'saved_session') else 'live',
        'profile': profile,
        'app_version': str(app_version),
        'baseline_note': BASELINE_NOTE,
        'condition_note': CONDITION_NOTE,
        'limits': LIMITS,
    }
    return validate_baseline(record)


def save_baseline(root, name, phone, record) -> Path:
    """Write the subject's baseline.json atomically and return its path.

    The record is validated first, the subject folder is created when needed,
    and the file lives in exactly one place:
    ``<root>/<slug(name)>_<phone>/baseline.json`` (``storage.subject_dir``
    re-checks that the path stays inside ``root``). An earlier baseline of the
    same subject is replaced; session folders are never touched.
    """
    validated = validate_baseline(record)
    folder = storage.subject_dir(root, name, phone)
    folder.mkdir(parents=True, exist_ok=True)
    return storage.write_json_atomic(folder / BASELINE_FILE, validated)


def load_baseline(root, name, phone) -> dict:
    """Read one subject's baseline.json.

    Returns ``{'present', 'record', 'error', 'path'}``: ``record`` is the
    validated baseline dict or None; ``error`` is '' for 'no file', and a short
    reason when a file exists but cannot be read or validated. Never raises for
    a missing/broken file, and never invents numbers.
    """
    folder = storage.subject_dir(root, name, phone)
    path = folder / BASELINE_FILE
    result = {'present': False, 'record': None, 'error': '', 'path': path}
    if not path.is_file():
        return result
    result['present'] = True
    try:
        payload = storage.read_json_object(path)
    except OSError as exc:
        result['error'] = f'{BASELINE_FILE} could not be read: {exc}'
        return result
    if payload is None:
        result['error'] = f'{BASELINE_FILE} is not a readable JSON object'
        return result
    try:
        result['record'] = validate_baseline(payload)
    except ValueError as exc:
        result['error'] = f'{BASELINE_FILE} is not a usable baseline: {exc}'
    return result


def best_estimate(summary):
    """Best stored estimate of one saved session: ``(report, source)``.

    Prefers a valid stable multi-epoch estimate, falls back to a valid
    five-second analysis, and returns ``(None, None)`` when neither is usable.
    """
    if not isinstance(summary, dict):
        return None, None
    stable = summary.get('stable_estimate')
    if isinstance(stable, dict) and stable.get('valid'):
        return stable, 'stable'
    report = summary.get('analysis')
    if isinstance(report, dict) and report.get('valid'):
        return report, 'five_second'
    return None, None


def reason_text(reason) -> str:
    """Short, honest explanation for a delta ``reason`` code."""
    return REASONS.get(reason if isinstance(reason, str) else '', 'not comparable')


def compare(record, report=None, source='stable', device=None) -> dict:
    """Descriptive deltas of one estimate against a baseline record.

    Returns a JSON-safe dict with ``available`` and ``reason`` plus, when both
    sides carry usable numbers: per-band relative-power deltas in percentage
    points (``current - baseline``), the peak-frequency delta in Hz
    (``current - baseline``, so positive = the current peak is higher), the
    dominant-band change (``dominant_from``/``dominant_to``/``dominant_changed``)
    and the source percentages each delta was computed from. Missing bands, None
    values or a non-passing estimate leave the affected delta None instead of
    raising or guessing. Nothing here assesses the person: these are differences
    of measured band numbers.
    """
    if source not in SOURCES:
        source = None
    delta = {
        'available': False, 'reason': 'no_baseline',
        'baseline_set_at': None, 'baseline_condition': None, 'baseline_source': None,
        'baseline_source_label': None, 'baseline_dominant': None,
        'baseline_dominant_pct': None, 'baseline_peak_hz': None,
        'baseline_channels': [], 'current_channels': [],
        'channels_changed': None, 'device_changed': None,
        'current_source': source,
        'current_source_label': SOURCE_LABELS.get(source) if source else None,
        'current_dominant': None, 'current_peak_hz': None,
        'dominant_from': None, 'dominant_to': None, 'dominant_changed': None,
        'peak_hz_delta': None,
        'relative_pct_delta': {name: None for name in BANDS},
        'baseline_relative_pct': {name: None for name in BANDS},
        'current_relative_pct': {name: None for name in BANDS},
        'largest_change': None,
        'note': DELTA_NOTE,
    }
    profile = record.get('profile') if isinstance(record, dict) else None
    if not isinstance(profile, dict):
        delta['reason'] = 'no_baseline'
        return delta
    delta.update({
        'baseline_set_at': _text(record.get('set_at')),
        'baseline_condition': _text(record.get('condition')),
        'baseline_source': profile.get('source') if profile.get('source') in SOURCES else None,
        'baseline_source_label': _text(profile.get('source_label')),
        'baseline_dominant': _text(profile.get('dominant')),
        'baseline_dominant_pct': _round(profile.get('dominant_pct'), 3),
        'baseline_peak_hz': _round(profile.get('peak_hz'), 3),
        'baseline_channels': [str(name) for name in (record.get('channels') or [])],
    })
    base_bands = profile.get('bands') if isinstance(profile.get('bands'), dict) else {}
    for name in BANDS:
        band = base_bands.get(name)
        delta['baseline_relative_pct'][name] = (_round(band.get('relative_pct'), 3)
                                                if isinstance(band, dict) else None)
    if not isinstance(report, dict):
        delta['reason'] = 'no_current_estimate'
        return delta
    delta['current_channels'] = [str(name) for name in (report.get('channels')
                                                       if isinstance(report.get('channels'), dict)
                                                       else {})]
    if delta['baseline_channels'] and delta['current_channels']:
        delta['channels_changed'] = delta['baseline_channels'] != delta['current_channels']
    current_device = device if isinstance(device, dict) else {}
    baseline_type = _text(record.get('device_type'))
    current_type = _text(current_device.get('device_key')) or _text(current_device.get('device_type'))
    baseline_label = _text(record.get('device_label'))
    current_label = _text(current_device.get('device_label'))
    if baseline_type and current_type:
        delta['device_changed'] = baseline_type != current_type
    elif baseline_label and current_label:
        delta['device_changed'] = baseline_label != current_label
    delta['current_dominant'] = _text(report.get('dominant'))
    delta['current_peak_hz'] = _round(report.get('peak_hz'), 3)
    report_bands = report.get('bands') if isinstance(report.get('bands'), dict) else {}
    for name in BANDS:
        band = report_bands.get(name)
        delta['current_relative_pct'][name] = (_round(band.get('relative_pct'), 3)
                                                if isinstance(band, dict) else None)
    if not report.get('valid'):
        delta['reason'] = 'current_estimate_not_valid'
        return delta
    delta['dominant_from'] = delta['baseline_dominant']
    delta['dominant_to'] = delta['current_dominant']
    if delta['dominant_from'] and delta['dominant_to']:
        delta['dominant_changed'] = delta['dominant_from'] != delta['dominant_to']
    if delta['baseline_peak_hz'] is not None and delta['current_peak_hz'] is not None:
        delta['peak_hz_delta'] = round(delta['current_peak_hz'] - delta['baseline_peak_hz'], 3)
    for name in BANDS:
        current = delta['current_relative_pct'][name]
        baseline = delta['baseline_relative_pct'][name]
        if current is not None and baseline is not None:
            delta['relative_pct_delta'][name] = round(current - baseline, 3)
    changes = {name: value for name, value in delta['relative_pct_delta'].items()
               if value is not None}
    if changes:
        name = max(changes, key=lambda band: (abs(changes[band]), band))
        delta['largest_change'] = {'band': name, 'delta_pp': changes[name]}
        delta['available'] = True
        delta['reason'] = 'ok'
    else:
        delta['reason'] = 'no_comparable_band_numbers'
    return delta


def vs_baseline_for_summary(record, report, source='stable', device=None):
    """Value for the ``vs_baseline`` key of a saved session, or None without a baseline.

    Returns None when the subject has no baseline, so a session summary only
    carries baseline deltas when a baseline actually existed at save time.
    """
    if not isinstance(record, dict) or not isinstance(record.get('profile'), dict):
        return None
    return compare(record, report, source=source, device=device)


def one_line(delta) -> str:
    """Compact one-line vs-baseline summary for a session row or panel ('' when nothing).

    Example: 'vs baseline: largest band change Theta +4.1 pp · peak 10.2 -> 9.8 Hz
    (-0.4 Hz) · Alpha -> Theta'. Only measured numbers and band names appear.
    """
    if not isinstance(delta, dict):
        return ''
    if not delta.get('available'):
        return f'vs baseline: {reason_text(delta.get("reason"))}'
    parts = []
    change = delta.get('largest_change') or {}
    if change.get('band') and _number(change.get('delta_pp')) is not None:
        parts.append(f'largest band change {change["band"]} '
                     f'{_signed(change["delta_pp"], 1)} pp')
    baseline_peak, current_peak = delta.get('baseline_peak_hz'), delta.get('current_peak_hz')
    peak_delta = _number(delta.get('peak_hz_delta'))
    if _number(baseline_peak) is not None and _number(current_peak) is not None:
        text = f'peak {baseline_peak:.1f} -> {current_peak:.1f} Hz'
        if peak_delta is not None:
            text += f' ({_signed(peak_delta, 1)} Hz)'
        parts.append(text)
    if delta.get('dominant_from') and delta.get('dominant_to'):
        arrow = ' -> '
        parts.append((delta['dominant_from'] + arrow + delta['dominant_to'])
                     if delta.get('dominant_changed') else
                     f'dominant unchanged ({delta["dominant_from"]})')
    if not parts:
        return 'vs baseline: no comparable numbers'
    return 'vs baseline: ' + ' · '.join(parts)


def _signed(value, places=1) -> str:
    """'+3.2' / '-0.4' for a delta value."""
    number = _number(value)
    if number is None:
        return '—'
    return f'{number:+.{places}f}'


def short_dominant_change(delta) -> str:
    """'Alpha -> Theta' / 'Alpha -> Alpha' / '—' for the dominant-band indicator."""
    if not isinstance(delta, dict):
        return '—'
    start, end = delta.get('dominant_from'), delta.get('dominant_to')
    if not start or not end:
        return '—'
    return f'{start} -> {end}'


def band_table(delta):
    """Rows for the UI delta table: ``[(band, baseline_pct, current_pct, delta_pp), ...]``."""
    if not isinstance(delta, dict) or not delta.get('available'):
        return [(name, None, None, None) for name in BANDS]
    return [(name, _number((delta.get('baseline_relative_pct') or {}).get(name)),
             _number((delta.get('current_relative_pct') or {}).get(name)),
             _number((delta.get('relative_pct_delta') or {}).get(name)))
            for name in BANDS]
