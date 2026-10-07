"""Local-only per-subject recording store for NeuroSync (nothing is uploaded).

Layout: ``<root>/<slug(name)>_<phone digits>/<YYYYmmdd_HHMMSS_ffffff>/`` with
``raw.csv``, ``summary.json`` and, when charts were supplied, ``charts/*.png``.
``DEFAULT_ROOT`` is ``recordings/subjects`` next to this file. Folder tokens
contain only lowercase ASCII letters, digits and single underscores, so a name
or phone value cannot introduce a path separator, drive letter or ``..``
segment; every built path is re-checked to stay inside the given root and
ValueError is raised otherwise. The module performs no network access of any
kind and never overwrites or deletes another session folder.

``json_text``/``write_json_atomic``/``read_json_object`` are the shared JSON
record helpers: ``baseline.py`` uses them to store the per-subject
``baseline.json`` in the same subject folder, with the same atomic write and
``allow_nan=False`` guarantees as ``summary.json``.

Honesty limits. The store serialises exactly what the caller passes in: it does
not validate, filter, repair or reinterpret EEG values, does not screen
artifacts and does not verify clock accuracy. ``saved_at`` is the machine's UTC
wall clock, not a hardware timestamp, and a flagged window stays flagged after
saving. ``summary.json`` is checked with ``json.dumps(..., allow_nan=False)``,
written to a temporary file and moved into place with ``os.replace``, so a
crash cannot leave a truncated summary; ``raw.csv`` is streamed in one pass, so
a hard crash can leave it short and readers must treat it as recoverable input
rather than a journal. ``load_session`` never returns more than MAX_RAW_ROWS
rows and marks the rest with ``raw_truncated`` instead of letting a clipped
recording look complete.
"""
import csv
import json
import numbers
import os
import re
import shutil
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parent / 'recordings' / 'subjects'

MAX_RAW_ROWS = 200_000
STORAGE_VERSION = 1
STORAGE_NOTE = 'Local-only research data; never uploaded.'
SUMMARY_FILE = 'summary.json'
RAW_FILE = 'raw.csv'
CHARTS_FOLDER = 'charts'

_UNSAFE_RUN = re.compile(r'[^a-z0-9]+')


def slug(value: str, max_len: int = 40) -> str:
    """Return a lowercase ASCII-safe folder token for an arbitrary value.

    NFKD-normalises, drops non-ASCII characters, lowercases, replaces every run
    of remaining non-token characters with a single '_', strips leading and
    trailing '_', then truncates to ``max_len`` without leaving a trailing '_'.
    Returns 'subject' when nothing usable remains.
    """
    try:
        limit = int(max_len)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('max_len must be an integer') from exc
    text = '' if value is None else str(value)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    token = _UNSAFE_RUN.sub('_', text.lower()).strip('_')
    if limit < 1:
        return 'subject'
    return token[:limit].rstrip('_') or 'subject'


def _phone_digits(phone) -> str:
    """Only the ASCII digit characters of ``phone`` (empty when there are none)."""
    text = '' if phone is None else str(phone)
    return ''.join(character for character in text if '0' <= character <= '9')


def subject_key(name: str, phone: str) -> str:
    """Folder token for one subject: ``<slug(name)>_<phone digits or nophone>``."""
    return f'{slug(name)}_{_phone_digits(phone) or "nophone"}'


def _inside(root, path) -> bool:
    """True when ``path`` resolves to ``root`` or something below it."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


def _checked(root, path):
    """Return ``path`` when it stays inside ``root``; raise ValueError otherwise."""
    if not _inside(root, path):
        raise ValueError(f'refusing a path outside the recordings root: {path}')
    return path


def subject_dir(root: Path, name: str, phone: str) -> Path:
    """Subject folder for one name/phone pair. Never creates anything."""
    return _checked(root, Path(root) / subject_key(name, phone))


def session_dir(root: Path, name: str, phone: str, timestamp: datetime | None = None) -> Path:
    """Unused session folder path for a timestamp. Never creates anything.

    The folder name is ``timestamp.strftime('%Y%m%d_%H%M%S_%f')`` (now when no
    timestamp is given); when that name is already taken, '_2', '_3', ... are
    appended until the candidate is unique.
    """
    folder = subject_dir(root, name, phone)
    stamp = (timestamp or datetime.now()).strftime('%Y%m%d_%H%M%S_%f')
    candidate = _checked(root, folder / stamp)
    suffix = 2
    while candidate.exists():
        candidate = _checked(root, folder / f'{stamp}_{suffix}')
        suffix += 1
    return candidate


def _validate_chart_name(chart_name):
    """Charts are bare ``*.png`` file names; anything else is a caller bug."""
    if not isinstance(chart_name, str) or not chart_name.strip():
        raise ValueError('chart names must be non-empty strings')
    if '/' in chart_name or '\\' in chart_name:
        raise ValueError(f'chart name must be a bare file name, got {chart_name!r}')
    if chart_name != Path(chart_name).name or chart_name in ('.', '..'):
        raise ValueError(f'chart name must be a bare file name, got {chart_name!r}')
    if Path(chart_name).suffix.lower() != '.png':
        raise ValueError(f'chart name must end in .png, got {chart_name!r}')
    return chart_name


def _chart_files(charts):
    """Validate a charts mapping into a name-sorted list of (name, bytes)."""
    if charts is None:
        return []
    if not isinstance(charts, dict):
        raise ValueError('charts must be a dict of file name -> PNG bytes')
    files = []
    for chart_name, data in charts.items():
        _validate_chart_name(chart_name)
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f'chart {chart_name!r} must be bytes, got {type(data).__name__}')
        files.append((chart_name, bytes(data)))
    files.sort(key=lambda item: item[0])
    return files


def _extended(session, folder, name, phone, chart_names):
    """Copy of the caller's summary plus the storage bookkeeping fields."""
    summary = dict(session)
    summary.update({
        'session_id': folder.name,
        'subject_name': '' if name is None else str(name),
        'subject_phone': '' if phone is None else str(phone),
        'subject_key': subject_key(name, phone),
        'saved_at': datetime.now(timezone.utc).isoformat(),
        'raw_csv': RAW_FILE,
        'charts': sorted(chart_names),
        'storage_version': STORAGE_VERSION,
        'storage_note': STORAGE_NOTE,
    })
    return summary


def _cell(value):
    """Raw-cell formatting: real numbers keep full precision, rest passes through."""
    if isinstance(value, numbers.Real) and not isinstance(value, (numbers.Integral, bool)):
        return repr(float(value))
    return value


def _write_raw(path, header, rows):
    """Stream header plus rows; values go to csv.writer as-is."""
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([_cell(value) for value in row])


def json_text(payload) -> str:
    """Indented JSON text of ``payload``; a non-serialisable value raises ValueError.

    ``allow_nan=False`` keeps NaN/Infinity out of every stored file, so a reader
    can never mistake a non-number for a measurement.
    """
    try:
        return json.dumps(payload, indent=2, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'record is not JSON-serialisable: {exc}') from exc


def _atomic_text(path, text):
    """Write text to ``path`` atomically: temp sibling, fsync, then os.replace.

    The parent folder must already exist. A failure removes the temporary file
    again, so no partially written record is ever left in place.
    """
    target = Path(path)
    temporary = target.with_name(target.name + '.tmp')
    try:
        with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return target


def write_json_atomic(path, payload: dict):
    """Write one JSON object to ``path`` atomically and return the path.

    Used for ``summary.json`` and the per-subject ``baseline.json``: the record
    is validated (must be a dict, must be JSON-serialisable with allow_nan=False)
    before anything is written, the parent folder must exist, and a crash cannot
    leave a truncated file behind. This function does not create directories and
    performs no network access.
    """
    if not isinstance(payload, dict):
        raise ValueError('payload must be a dict of JSON fields')
    return _atomic_text(Path(path), json_text(payload))


def read_json_object(path):
    """Parsed JSON object from one local record file, or None when unusable.

    Returns None when the file is missing, unreadable, not valid JSON or not a
    JSON object; callers must treat that as 'no record' rather than as an empty
    measurement.
    """
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def save_session(root, name, phone, session: dict, raw_header: list[str],
                 raw_rows: Iterable[Sequence], charts: dict[str, bytes] | None = None,
                 timestamp: datetime | None = None) -> Path:
    """Write one recording session and return its folder.

    Creates ``session_dir(...)`` with parents and ``exist_ok=False``, then writes
    ``raw.csv`` (first row is ``raw_header`` exactly), optional ``charts/<name>.png``
    files and ``summary.json`` (the session dict extended with session_id,
    subject fields, saved_at, raw_csv, charts, storage_version and storage_note).
    Inputs are validated before the folder exists: a non-dict session, values that
    are not JSON-serialisable (json's TypeError/ValueError becomes ValueError), a
    non-.png or non-bare chart name, or non-bytes chart data all raise ValueError
    with nothing written. ``summary.json`` goes through the same atomic
    temp-file/fsync/replace path as ``baseline.json``. If a write fails part-way
    the new session folder is removed again.
    """
    if not isinstance(session, dict):
        raise ValueError('session must be a dict of summary fields')
    try:
        header = list(raw_header)
    except TypeError as exc:
        raise ValueError('raw_header must be a sequence of column names') from exc
    folder = session_dir(root, name, phone, timestamp)
    chart_files = _chart_files(charts)
    summary = _extended(session, folder, name, phone, [item[0] for item in chart_files])
    # Serialise before anything exists on disk: a non-JSON payload must not
    # create a session folder.
    payload = json_text(summary)
    folder.mkdir(parents=True, exist_ok=False)
    try:
        _write_raw(folder / RAW_FILE, header, raw_rows)
        if chart_files:
            chart_folder = folder / CHARTS_FOLDER
            chart_folder.mkdir()
            for chart_name, data in chart_files:
                (chart_folder / chart_name).write_bytes(data)
        _atomic_text(folder / SUMMARY_FILE, payload)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    return folder


def _text(value):
    """Return a non-empty string, else None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, numbers.Real):
        return str(value)
    return None


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else []


def _key_parts(key):
    """Best-effort name/phone from a folder name: split at the last '_'."""
    name, separator, phone = key.rpartition('_')
    if not separator:
        return (key or None), None
    return (name or None), (phone or None)


def _read_summary(path):
    """Parsed summary dict, or None when the file is missing/unreadable/not an object."""
    return read_json_object(path)


def _session_summaries(subject):
    """List of ``(session_folder, summary)`` for one subject, newest name first."""
    try:
        entries = list(subject.iterdir())
    except OSError:
        return []
    found = []
    for entry in sorted(entries, key=lambda path: path.name, reverse=True):
        if not entry.is_dir():
            continue
        summary = _read_summary(entry / SUMMARY_FILE)
        if summary is not None:
            found.append((entry, summary))
    return found


def list_subjects(root: Path) -> list[dict]:
    """One entry per subject folder, newest recording first then by name.

    Each entry: name/phone (from the newest persisted summary when it has them,
    otherwise parsed from the folder name key), key, path, sessions (folders that
    contain a summary.json) and last_session_at (ISO UTC string or None). Folders
    without any summary are still listed. A missing root returns [].
    """
    root = Path(root)
    if not root.is_dir():
        return []
    items = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        sessions = _session_summaries(folder)
        name = phone = None
        for _, summary in sessions:
            if name is None:
                name = _text(summary.get('subject_name'))
            if phone is None:
                phone = _text(summary.get('subject_phone'))
            if name is not None and phone is not None:
                break
        if name is None or phone is None:
            key_name, key_phone = _key_parts(folder.name)
            name = key_name if name is None else name
            phone = key_phone if phone is None else phone
        stamps = [stamp for stamp in (_text(summary.get('saved_at'))
                                      for _, summary in sessions) if stamp]
        items.append({
            'name': name, 'phone': phone, 'key': folder.name, 'path': folder,
            'sessions': len(sessions),
            'last_session_at': max(stamps) if stamps else None,
        })
    items.sort(key=lambda item: (item['name'] or '', item['key']))
    items.sort(key=lambda item: item['last_session_at'] or '', reverse=True)
    return items


def list_sessions(root, name, phone) -> list[dict]:
    """Session folders of one subject that contain a summary.json, newest first.

    Each entry: session_id, path, started_at, device_type, device_label,
    channels (list), duration_s, notes, charts (list), read from summary.json
    with .get defaults. A missing subject folder returns [].
    """
    subject = subject_dir(root, name, phone)
    if not subject.is_dir():
        return []
    return [{
        'session_id': _text(summary.get('session_id')),
        'path': folder,
        'started_at': _text(summary.get('started_at')),
        'device_type': summary.get('device_type'),
        'device_label': summary.get('device_label'),
        'channels': _as_list(summary.get('channels')),
        'duration_s': summary.get('duration_s'),
        'notes': summary.get('notes'),
        'charts': _as_list(summary.get('charts')),
    } for folder, summary in _session_summaries(subject)]


def _read_raw(path):
    """Header plus up to MAX_RAW_ROWS data rows and a truncation flag."""
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = []
        truncated = False
        for row in reader:
            if len(rows) >= MAX_RAW_ROWS:
                truncated = True
                break
            rows.append(row)
    return header, rows, truncated


def load_session(path: Path) -> dict:
    """Read one saved session folder back.

    Returns ``{'summary', 'raw_path', 'raw_header', 'raw_rows', 'charts'}`` where
    raw_path/header/rows are empty when raw.csv is absent, charts are the sorted
    existing files in charts/, and 'raw_truncated': True is added when more than
    MAX_RAW_ROWS data rows exist (at most MAX_RAW_ROWS are returned). Raises
    FileNotFoundError when summary.json is missing.
    """
    folder = Path(path)
    summary_path = folder / SUMMARY_FILE
    if not summary_path.is_file():
        raise FileNotFoundError(f'no {SUMMARY_FILE} in {folder}')
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if not isinstance(summary, dict):
        raise ValueError(f'{summary_path} must contain a JSON object')
    result = {'summary': summary, 'raw_path': None, 'raw_header': [],
              'raw_rows': [], 'charts': []}
    raw_path = folder / RAW_FILE
    if raw_path.is_file():
        header, rows, truncated = _read_raw(raw_path)
        result.update(raw_path=raw_path, raw_header=header, raw_rows=rows)
        if truncated:
            result['raw_truncated'] = True
    chart_folder = folder / CHARTS_FOLDER
    if chart_folder.is_dir():
        result['charts'] = sorted(
            (entry for entry in chart_folder.iterdir() if entry.is_file()),
            key=lambda entry: entry.name)
    return result
