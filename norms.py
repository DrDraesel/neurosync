"""Norm files: licensed-export import and local accumulation.

Implements section 1 of ``docs/QEEG-NORMS-SPEC.md``. A ``NormFile`` is a
mean/SD table per grid row, band and site (or homologous pair), plus the
metadata that makes it honest: where it came from, how many subjects, which
age band and condition, which device scope, and what was done to the numbers.
This module never invents a normative value and never relabels an import as
local.

Two sources, never mixed
------------------------
* ``import_csv`` reads a licensed export (NeuroGuide, Brain Science
  International, or a generic CSV). Only ``mean`` + ``SD`` can seed a
  reference; a z-only export is refused because a z-score without its mean and
  spread cannot z-score a new recording. Channels outside the app's 10-20
  schematic montage are dropped with a note, the file's SHA-256 is kept in
  ``provenance``, and ``source`` is preserved verbatim.
* ``build_local_norm`` recomputes every session with
  ``qeeg.compute_metrics`` from the stored raw CSV - never from cached numbers
  of unknown version - and aggregates ACROSS SUBJECTS. A subject with two
  sessions in the same condition contributes the mean of those sessions, not
  two rows. Fewer than ``min_n`` eligible subjects returns ``(None, reason)``;
  there is no "almost a norm".

Return convention
-----------------
``build_local_norm`` returns ``(NormFile, '')`` on success and
``(None, reason)`` when it refuses, so a caller can show the real reason
instead of a bare ``None``. ``import_csv`` raises ``ValueError`` for an export
it cannot honestly use.

Montage
-------
The channel filter uses ``topomap.ELECTRODE_POSITIONS``, the same 23-site
schematic layout the head map draws (standard 10-20 plus the app's Fp7/Fp8/Fpz
names). Using one montage everywhere beats inventing a second, narrower one in
this module.

Honesty limits
--------------
A local norm is a description of the recordings this lab actually made, in one
condition and age band, with one analysis version. It is not an age-matched
clinical normative database. z-scores against it are only as good as that
sample, and nothing here is a diagnosis.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import qeeg
from topomap import ELECTRODE_POSITIONS

SCHEMA = 'lab.norm/1'
SOURCES = ('local', 'neuroguide', 'bsi', 'generic')
CONDITIONS = ('eyes_closed', 'eyes_open', 'other')
MIN_LOCAL_SUBJECTS = 20

# The app's schematic 10-20 set: every site the head map can draw. Fp7/Fp8 are
# 10-10 extensions that this device family reports, so they are included rather
# than silently dropped from an imported export.
MONTAGE = tuple(ELECTRODE_POSITIONS)
_MONTAGE_SET = frozenset(MONTAGE)
_MONTAGE_ORDER = {name: index for index, name in enumerate(MONTAGE)}
_SITE_BY_KEY = {name.lower(): name for name in MONTAGE}

# Age bands used when only a numeric age is stored. A band string in the
# summary wins; these are the default buckets and any 'a-b' / 'x+' string is
# accepted for matching.
AGE_BANDS = ('0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+')

_ROWS = tuple(qeeg.ROWS)
_BANDS = tuple(qeeg.BANDS)

__all__ = [
    'SCHEMA', 'SOURCES', 'CONDITIONS', 'MIN_LOCAL_SUBJECTS', 'MONTAGE',
    'AGE_BANDS', 'NormFile', 'import_csv', 'build_local_norm', 'save', 'load',
    'to_dict', 'from_dict', 'scope_warning', 'footer_text', 'reference_line',
    'to_reference', 'normalize_condition', 'age_band_for', 'age_in_band',
    'qeeg_version',
]


# --------------------------------------------------------------------------
# NormFile
# --------------------------------------------------------------------------
@dataclass
class NormFile:
    """A mean/SD reference table with its provenance (schema ``lab.norm/1``).

    ``rows`` is ``{row: {band: {channel_or_pair: {'mean': float, 'sd': float}}}}``
    in ``qeeg.ROWS`` / ``qeeg.BANDS`` terms. ``provenance`` carries the file
    name and SHA-256 for an import, or the per-subject list for a local build.
    """

    schema: str = SCHEMA
    source: str = 'generic'
    label: str = ''
    n: int = 0
    age_band: str = ''
    condition: str = 'other'
    device_scope: str = 'any'
    created_at: str = ''
    rows: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# small validators / canonicalisers
# --------------------------------------------------------------------------
def _iso_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean_key(value):
    """Header/name key: lower case, non-alphanumerics collapsed to '_'."""
    if value is None:
        return ''
    text = str(value).strip().lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def _number(value):
    """Finite float or None; booleans are not numbers here."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_site(value):
    """Canonical 10-20 spelling for a site name, or None outside the montage."""
    if not isinstance(value, str):
        return None
    return _SITE_BY_KEY.get(value.strip().lower().replace(' ', ''))


def _canonical_cell_key(value):
    """A montage site (``'F3'``) or homologous pair (``'F3/F4'``), else None."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for separator in ('/', '-'):
        if separator in text:
            left_text, right_text = text.split(separator, 1)
            left = _canonical_site(left_text)
            right = _canonical_site(right_text)
            if left and right and left != right:
                return qeeg.pair_label(left, right)
            return None
    return _canonical_site(text)


def _canonical_band(value):
    key = _clean_key(value)
    if not key:
        return None
    aliases = {
        'delta': 'Delta', 'theta': 'Theta', 'alpha': 'Alpha',
        'beta': 'Beta', 'high_beta': 'HighBeta', 'highbeta': 'HighBeta',
        'high_beta_2': 'HighBeta', 'beta2': 'HighBeta', 'beta_2': 'HighBeta',
        'high': 'HighBeta',
    }
    if key in aliases:
        return aliases[key]
    for band in _BANDS:
        if key == band.lower() or key.startswith(band.lower() + '_'):
            return band
    return None


def _canonical_row(value):
    key = _clean_key(value)
    if not key:
        return None
    changed = True
    while changed:
        changed = False
        for suffix in ('_power', '_uv2', '_uv', '_squared', '_uv_squared'):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                changed = True
    aliases = {
        'absolute': 'absolute', 'abs': 'absolute', 'absolute_power': 'absolute',
        'relative': 'relative', 'rel': 'relative', 'relative_power': 'relative',
        'asymmetry': 'asymmetry', 'amplitude_asymmetry': 'asymmetry',
        'amp_asymmetry': 'asymmetry', 'asym': 'asymmetry',
        'coherence': 'coherence', 'coh': 'coherence',
        'phase': 'phase', 'phase_lag': 'phase', 'phaselag': 'phase',
    }
    return aliases.get(key)


_CONDITION_ALIASES = {
    'eyes_closed': 'eyes_closed', 'eyesclosed': 'eyes_closed',
    'ec': 'eyes_closed', 'closed': 'eyes_closed',
    'eyes_open': 'eyes_open', 'eyesopen': 'eyes_open',
    'eo': 'eyes_open', 'open': 'eyes_open',
    'other': 'other',
}


def normalize_condition(value):
    """Canonical ``eyes_closed|eyes_open|other``; ValueError otherwise.

    Accepts the app's display spellings ('eyes closed', 'Eyes Open') and the
    short EC/EO forms so a stored summary and an import agree on one value.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError('condition is required and must be one of: '
                         + ', '.join(CONDITIONS))
    key = _clean_key(value)
    key = key.replace('_', '')
    canonical = _CONDITION_ALIASES.get(key) or _CONDITION_ALIASES.get(_clean_key(value))
    if canonical is None:
        raise ValueError(f'unknown condition {value!r}; expected one of: '
                         + ', '.join(CONDITIONS))
    return canonical


def _condition_or_none(value):
    try:
        return normalize_condition(value)
    except ValueError:
        return None


def _required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} is required')
    return value.strip()


def _validated_source(source):
    if not isinstance(source, str) or not source.strip():
        raise ValueError('source is required (local, neuroguide, bsi or generic)')
    text = source.strip().lower()
    if text not in SOURCES:
        raise ValueError(f'source must be one of: {", ".join(SOURCES)}')
    return text


def _validated_scope(scope):
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError('device_scope is required (e.g. 19ch, 21ch, 4ch, any)')
    text = scope.strip().lower()
    if text == 'any':
        return text
    if not re.fullmatch(r'\d+ch', text) or int(text[:-2]) < 1:
        raise ValueError("device_scope must look like '19ch', '21ch', '4ch' or 'any'")
    return text


def _scope_count(scope):
    """Channel count a scope promises, or None for 'any'."""
    text = str(scope or '').strip().lower()
    if text == 'any' or not text:
        return None
    match = re.fullmatch(r'(\d+)ch', text)
    return int(match.group(1)) if match else None


def _scope_for_count(count):
    return f'{int(count)}ch'


def _digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


_QEEG_VERSION = None


def qeeg_version():
    """Version stamp of the numerics used: ``qeeg.py@<sha256 prefix>``.

    A hash of the actual source file is honest about what changed; a
    hand-maintained version string would drift. If the file cannot be read the
    stamp says so instead of pretending.
    """
    global _QEEG_VERSION
    if _QEEG_VERSION is not None:
        return _QEEG_VERSION
    try:
        data = Path(qeeg.__file__).read_bytes()
        _QEEG_VERSION = 'qeeg.py@' + _digest_bytes(data)[:16]
    except (OSError, TypeError):
        _QEEG_VERSION = 'qeeg.py@unknown'
    return _QEEG_VERSION


# --------------------------------------------------------------------------
# age bands
# --------------------------------------------------------------------------
def _band_text(value):
    return (str(value).strip().lower()
            .replace('\u2013', '-').replace('\u2014', '-')).replace(' ', '')


def age_in_band(age, band):
    """True when ``age`` falls inside an ``'a-b'`` or ``'x+'`` band string."""
    number = _number(age)
    if number is None:
        return False
    text = _band_text(band)
    match = re.fullmatch(r'(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)', text)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return low <= number <= high
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\+', text)
    if match:
        return number >= float(match.group(1))
    return False


def age_band_for(age, bands=AGE_BANDS):
    """Default age band for a numeric age, or None outside every band."""
    number = _number(age)
    if number is None:
        return None
    for band in bands:
        if age_in_band(number, band):
            return band
    return None


# --------------------------------------------------------------------------
# row normalisation
# --------------------------------------------------------------------------
def _clean_rows(source):
    """Validate ``{row: {band: {cell: {'mean','sd'}}}}`` into plain floats.

    Empty rows/bands are dropped; a non-finite or negative entry is a caller
    or file bug and raises ValueError rather than being smoothed over.
    """
    if not isinstance(source, Mapping):
        raise ValueError("norm file is missing its 'rows' table")
    rows = OrderedDict()
    for row, bands in source.items():
        if not isinstance(row, str) or not row.strip():
            continue
        if not isinstance(bands, Mapping):
            continue
        clean_bands = OrderedDict()
        for band, cells in bands.items():
            if not isinstance(band, str) or not band.strip():
                continue
            if not isinstance(cells, Mapping):
                continue
            clean_cells = OrderedDict()
            for name, cell in cells.items():
                if not isinstance(name, str) or not name.strip():
                    continue
                if not isinstance(cell, Mapping):
                    raise ValueError(f'norm cell {row}/{band}/{name} must be a mean/SD mapping')
                mean = _number(cell.get('mean'))
                sd = _number(cell.get('sd'))
                if mean is None or sd is None or sd < 0:
                    raise ValueError(f'norm cell {row}/{band}/{name} needs a finite '
                                     f'mean and a non-negative SD')
                clean_cells[name] = {'mean': mean, 'sd': sd}
            if clean_cells:
                clean_bands[band] = clean_cells
        if clean_bands:
            rows[row] = clean_bands
    if not rows:
        raise ValueError('norm file contains no mean/SD rows')
    return rows


def to_dict(norm):
    """Plain JSON-shaped dict for a NormFile (validated)."""
    if isinstance(norm, NormFile):
        raw = {
            'schema': norm.schema, 'source': norm.source, 'label': norm.label,
            'n': norm.n, 'age_band': norm.age_band, 'condition': norm.condition,
            'device_scope': norm.device_scope, 'created_at': norm.created_at,
            'rows': norm.rows, 'provenance': norm.provenance,
        }
    elif isinstance(norm, Mapping):
        raw = dict(norm)
    else:
        raise ValueError('norm must be a NormFile or a mapping')
    return _validated(raw)


def _validated(raw):
    """Normalise and validate a norm mapping; ValueError on anything dishonest."""
    schema = raw.get('schema', SCHEMA)
    if schema != SCHEMA:
        raise ValueError(f'norm schema must be {SCHEMA!r}')

    source = raw.get('source')
    if not source:
        raise ValueError("norm file is missing 'source'")
    source = _validated_source(source)

    n = raw.get('n')
    if n is None:
        raise ValueError("norm file is missing 'n'")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
        raise ValueError("norm file 'n' must be a non-negative integer")
    n = int(n)

    age_band = raw.get('age_band')
    if not isinstance(age_band, str) or not age_band.strip():
        raise ValueError('norm file is missing a recorded age band')
    age_band = age_band.strip()

    condition = raw.get('condition')
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError('norm file is missing a recorded condition')
    condition = normalize_condition(condition)

    device_scope = _validated_scope(raw.get('device_scope') or 'any')

    label = raw.get('label')
    label = '' if label is None else str(label)

    created = raw.get('created_at')
    if not isinstance(created, str) or not created.strip():
        created = _iso_now()

    provenance = raw.get('provenance')
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, Mapping):
        raise ValueError('norm provenance must be a mapping')
    provenance = dict(provenance)

    if source == 'local':
        subjects = provenance.get('subjects')
        if not isinstance(subjects, (list, tuple)) or not subjects:
            raise ValueError('a local norm must carry its provenance.subjects list '
                             '(one entry per subject)')
        if int(n) != len(subjects):
            raise ValueError(f'local norm n={n} does not match '
                             f'{len(subjects)} provenance.subjects entries')

    rows = _clean_rows(raw.get('rows'))
    return {
        'schema': SCHEMA, 'source': source, 'label': label, 'n': n,
        'age_band': age_band, 'condition': condition,
        'device_scope': device_scope, 'created_at': created,
        'rows': rows, 'provenance': provenance,
    }


def from_dict(raw):
    """NormFile from a JSON-shaped mapping, validated."""
    return NormFile(**_validated(raw))


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------
def save(norm, path):
    """Write a NormFile as JSON atomically and return the path.

    A local file without ``provenance.subjects`` (or whose ``n`` disagrees
    with that list) is refused before anything is written.
    """
    clean = to_dict(norm)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    try:
        text = json.dumps(clean, indent=2, allow_nan=False) + '\n'
        temporary.write_text(text, encoding='utf-8')
        os.replace(temporary, target)
    except (OSError, ValueError):
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    return str(target)


def load(path):
    """Read and validate a NormFile JSON file.

    Refuses (ValueError) a wrong schema, a missing ``n``/``age_band``/
    ``source``, and a file whose ``source == 'local'`` but which lacks its
    per-subject provenance - an accumulated norm must be able to say who is in
    it.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding='utf-8')
    except OSError as exc:
        raise ValueError(f'could not read norm file: {exc}') from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'norm file is not valid JSON: {exc}') from exc
    if not isinstance(raw, Mapping):
        raise ValueError('norm file must contain a JSON object')
    return from_dict(raw)


# --------------------------------------------------------------------------
# device scope
# --------------------------------------------------------------------------
def _device_site_count(device_channels):
    """Count of montage sites in a device channel list / summary / int."""
    if isinstance(device_channels, bool):
        raise ValueError('device_channels must be a channel sequence or a count')
    if isinstance(device_channels, (int, np.integer)):
        return int(device_channels)
    if isinstance(device_channels, Mapping):
        names = (device_channels.get('scalp_channels')
                 or device_channels.get('channels') or ())
    elif isinstance(device_channels, str):
        raise ValueError('device_channels must be a sequence of channel names, '
                         'not one string')
    else:
        names = device_channels
    try:
        return len([site for site in (_canonical_site(name) for name in names)
                    if site])
    except TypeError as exc:
        raise ValueError('device_channels must be a sequence of channel names') from exc


def scope_warning(norm, device_channels):
    """Mismatch sentence for a norm used on a device of another scope, else ''.

    A norm may still be applied (the caller decides), but a 21-channel norm on
    a 4-channel recording must never be silently presented as comparable.
    'any' scopes match everything; an unknown device count cannot be judged and
    returns ''.
    """
    scope = norm.device_scope if isinstance(norm, NormFile) else norm.get('device_scope')
    scope = _validated_scope(scope or 'any')
    expected = _scope_count(scope)
    count = _device_site_count(device_channels)
    if expected is None or count == 0 or count == expected:
        return ''
    return f'device scope mismatch ({scope} norm on {_scope_for_count(count)} recording)'


def reference_line(norm):
    """The grid footer line: ``reference: <source> · n=<n> · age <band> · <cond>``."""
    def field(name):
        if isinstance(norm, NormFile):
            return getattr(norm, name)
        if isinstance(norm, Mapping):
            return norm.get(name)
        raise ValueError('norm must be a NormFile or a mapping')
    return (f'reference: {field("source")} · n={field("n")} · '
            f'age {field("age_band")} · {field("condition")}')


def footer_text(norm, device_channels=None):
    """Footer line plus the mismatch suffix when the device scope differs."""
    line = reference_line(norm)
    if device_channels is not None:
        warning = scope_warning(norm, device_channels)
        if warning:
            line += f' · {warning}'
    return line


def to_reference(norm):
    """Convert a NormFile into a qeeg Reference mapping (kind ``norm``).

    The grid reads one mean/SD per (row, band, channel); a NormFile stores the
    same numbers with a per-cell dict, so the flattening happens here. A norm
    with fewer than 20 subjects is refused because qeeg will not score against
    it and calling it a norm would be a lie.
    """
    clean = to_dict(norm)
    if clean['n'] < MIN_LOCAL_SUBJECTS:
        raise ValueError(f'a norm reference needs at least {MIN_LOCAL_SUBJECTS} '
                         f'subjects; this file has n={clean["n"]}')
    mean, sd = OrderedDict(), OrderedDict()
    for row, bands in clean['rows'].items():
        mean[row], sd[row] = OrderedDict(), OrderedDict()
        for band, cells in bands.items():
            mean[row][band] = OrderedDict((name, cell['mean'])
                                          for name, cell in cells.items())
            sd[row][band] = OrderedDict((name, cell['sd'])
                                        for name, cell in cells.items())
    return {
        'schema': qeeg.REFERENCE_SCHEMA,
        'kind': 'norm',
        'label': clean['label'],
        'n': clean['n'],
        'age_band': clean['age_band'],
        'bands': list(OrderedDict.fromkeys(
            band for bands in clean['rows'].values() for band in bands)),
        'rows': list(clean['rows']),
        'mean': mean,
        'sd': sd,
        'created_at': clean['created_at'],
    }


# --------------------------------------------------------------------------
# licensed-export import
# --------------------------------------------------------------------------
_CHANNEL_ALIASES = ('channel', 'electrode', 'site', 'chan', 'electrode_name')
_BAND_ALIASES = ('band', 'frequency_band', 'freq_band', 'band_name')
_ROW_ALIASES = ('row', 'measure', 'metric', 'row_name', 'type')
_Z_ALIASES = ('z', 'z_score', 'zscore', 'z_value')
_MEAN_ALIASES = ('mean', 'mean_uv2', 'value', 'value_mean', 'band_mean')
_SD_ALIASES = ('sd', 'std', 'stdev', 'stddev', 'standard_deviation',
               'sd_uv2', 'value_sd')
_N_ALIASES = ('n', 'n_subjects', 'subjects', 'subject_count', 'sample_size', 'count')
_RAW_SUFFIXES = ('_uv2', '_uv', '_microvolts', '_microvolt', '_uv_squared')


def _find(keys, aliases):
    for alias in aliases:
        if alias in keys:
            return alias
    return None


def _first_clean_keys(fieldnames):
    lookup = OrderedDict()
    for name in fieldnames:
        if name is None:
            continue
        key = _clean_key(name)
        if key and key not in lookup:
            lookup[key] = name
    return lookup


def _wide_families(lookup):
    """``(family, mean_header, sd_header)`` for every ``*_mean``/``*_sd`` pair."""
    families = []
    keys = list(lookup)
    for key in keys:
        if not key.endswith('_mean'):
            continue
        prefix = key[:-5]
        if prefix and prefix + '_sd' in lookup:
            families.append((prefix, lookup[key], lookup[prefix + '_sd']))
    return families


def _fmt_names(names, limit=12):
    names = sorted(names)
    shown = ', '.join(names[:limit])
    if len(names) > limit:
        shown += f', … (+{len(names) - limit} more)'
    return shown


NOTE_LIMIT = 50


def _capped(notes, limit=NOTE_LIMIT):
    """Keep a provenance note list readable on a large recording tree."""
    notes = list(notes)
    if len(notes) <= limit:
        return notes
    return notes[:limit] + [f'… {len(notes) - limit} more note(s) omitted']


def import_csv(path, source='generic', age_band=None, condition=None,
               device_scope='any'):
    """Import a licensed CSV export as a NormFile (docs/QEEG-NORMS-SPEC.md §1a).

    Three shapes are detected by header:

    * **long**: ``channel,band,row,mean,sd`` (preferred);
    * **wide mean/SD**: ``channel,band,absolute_mean,absolute_sd,
      relative_mean,relative_sd,…`` - each ``*_mean``/``*_sd`` pair becomes a
      grid row;
    * **z-only**: ``channel,band,row,z`` - refused, because a z-score alone
      cannot seed a reference. The message is the spec's exact wording.

    Channels outside the montage are dropped with a note, as are unknown bands
    or rows and cells without a finite mean and non-negative SD. The file's
    SHA-256 and the detected shape go into ``provenance``.
    """
    source = _validated_source(source)
    age_band = _required_text(age_band, 'age_band')
    condition = normalize_condition(condition)
    device_scope = _validated_scope(device_scope)

    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ValueError(f'could not read norm export: {exc}') from exc
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError as exc:
        raise ValueError(f'norm export is not UTF-8 text: {exc}') from exc

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError('norm export is empty (no header row)')
    lookup = _first_clean_keys(fieldnames)
    keys = set(lookup)

    channel_col = _find(keys, _CHANNEL_ALIASES)
    band_col = _find(keys, _BAND_ALIASES)
    row_col = _find(keys, _ROW_ALIASES)
    z_col = _find(keys, _Z_ALIASES)
    mean_col = _find(keys, _MEAN_ALIASES)
    sd_col = _find(keys, _SD_ALIASES)
    n_col = _find(keys, _N_ALIASES)
    families = _wide_families(lookup)

    has_plain_mean_sd = bool(mean_col and sd_col)
    has_family_mean_sd = bool(families)

    if z_col and not (has_plain_mean_sd or has_family_mean_sd):
        raise ValueError('z-only exports cannot seed a reference — export mean and SD')
    if not (has_plain_mean_sd or has_family_mean_sd):
        if mean_col and not sd_col:
            raise ValueError('exports must carry both mean and SD; this file has '
                             'mean but no SD column')
        raise ValueError('unrecognised norm export header; expected '
                         'channel,band,row,mean,sd or channel,band,<row>_mean,<row>_sd')
    if not channel_col:
        raise ValueError("norm export has no channel column")
    if not band_col:
        raise ValueError("norm export has no band column")

    if has_plain_mean_sd and row_col:
        shape = 'long'
    elif has_plain_mean_sd:
        shape = 'wide-mean-sd'
    elif families:
        shape = 'wide-mean-sd'
    else:  # pragma: no cover - guarded above
        shape = 'unknown'

    notes = []
    dropped = set()
    skipped = 0
    duplicates = 0
    unknown_rows = set()
    unknown_bands = set()
    n_values = []

    rows = OrderedDict()
    for record in reader:
        channel_raw = record.get(lookup[channel_col])
        channel = _canonical_cell_key(channel_raw)
        if channel is None:
            if channel_raw is not None and str(channel_raw).strip():
                dropped.add(str(channel_raw).strip())
            continue
        band = _canonical_band(record.get(lookup[band_col]))
        if band is None:
            band_raw = record.get(lookup[band_col])
            if band_raw is not None and str(band_raw).strip():
                unknown_bands.add(str(band_raw).strip())
            continue

        if has_plain_mean_sd:
            row_name = _canonical_row(record.get(lookup[row_col])) if row_col else 'absolute'
            if row_name is None:
                raw_row = record.get(lookup[row_col]) if row_col else ''
                if raw_row:
                    unknown_rows.add(str(raw_row).strip())
                continue
            sources = [(row_name, lookup[mean_col], lookup[sd_col])]
        else:
            sources = []
            for family, mean_name, sd_name in families:
                row_name = _canonical_row(family)
                if row_name is None and row_col:
                    row_name = _canonical_row(record.get(lookup[row_col]))
                if row_name is None:
                    unknown_rows.add(family)
                    continue
                sources.append((row_name, mean_name, sd_name))

        for row_name, mean_name, sd_name in sources:
            mean = _number(record.get(mean_name))
            sd = _number(record.get(sd_name))
            if mean is None or sd is None or sd < 0:
                skipped += 1
                continue
            band_cells = rows.setdefault(row_name, OrderedDict()).setdefault(
                band, OrderedDict())
            if channel in band_cells:
                duplicates += 1
                continue
            band_cells[channel] = {'mean': mean, 'sd': sd}

        if n_col:
            count = _number(record.get(lookup[n_col]))
            if count is not None and count >= 0 and float(count).is_integer():
                n_values.append(int(count))

    rows = _drop_empty(rows)
    if not rows:
        raise ValueError('norm export contains no usable rows after filtering')

    if dropped:
        notes.append(f'dropped {len(dropped)} channel/pair name(s) outside the '
                     f'10-20 montage: {_fmt_names(dropped)}')
    if unknown_bands:
        notes.append(f'dropped {len(unknown_bands)} unknown band(s): '
                     f'{_fmt_names(unknown_bands)}')
    if unknown_rows:
        notes.append(f'dropped {len(unknown_rows)} row(s) with no grid row: '
                     f'{_fmt_names(unknown_rows)}')
    if skipped:
        notes.append(f'skipped {skipped} cell(s) without a finite mean and '
                     f'non-negative SD')
    if duplicates:
        notes.append(f'ignored {duplicates} duplicate cell(s) (first value kept)')

    n = 0
    if n_values:
        n = min(n_values)
        if len(set(n_values)) > 1:
            notes.append(f'export states varying subject counts; the smallest '
                         f'({n}) is used')
    else:
        notes.append('the export states no subject count; n=0 means "not stated"')

    provenance = {
        'file': file_path.name,
        'sha256': _digest_bytes(data),
        'bytes': len(data),
        'exporter': source,
        'shape': shape,
        'imported_at': _iso_now(),
        'notes': notes,
        'dropped_channels': sorted(dropped),
        'n_source': 'column' if n_values else 'not stated',
    }
    return NormFile(
        schema=SCHEMA, source=source, label=file_path.stem, n=n,
        age_band=age_band, condition=condition, device_scope=device_scope,
        created_at=_iso_now(), rows=rows, provenance=provenance,
    )


def _drop_empty(rows):
    out = OrderedDict()
    for row, bands in rows.items():
        kept = OrderedDict((band, cells) for band, cells in bands.items() if cells)
        if kept:
            out[row] = kept
    return out


# --------------------------------------------------------------------------
# local norm accumulation
# --------------------------------------------------------------------------
def _read_summary(path):
    try:
        with open(path, encoding='utf-8') as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _analysis_of(summary):
    analysis = summary.get('analysis')
    return analysis if isinstance(analysis, Mapping) else {}


def _session_valid(summary):
    """True when the summary carries a passing validity flag.

    The app writes the flag inside ``analysis`` / ``stable_estimate`` today and
    older notes use a top-level ``valid``; all three are read, top level first.
    """
    if isinstance(summary.get('valid'), bool):
        return summary['valid']
    for key in ('analysis', 'stable_estimate'):
        block = summary.get(key)
        if isinstance(block, Mapping) and isinstance(block.get('valid'), bool):
            return block['valid']
    return False


def _summary_condition(summary):
    for value in (summary.get('condition'), summary.get('measurement_condition'),
                  _analysis_of(summary).get('condition')):
        canonical = _condition_or_none(value)
        if canonical:
            return canonical
    return None


def _summary_age(summary):
    for value in (summary.get('age'), summary.get('subject_age')):
        number = _number(value)
        if number is not None and number >= 0:
            return number
    return None


def _summary_age_band(summary):
    for value in (summary.get('age_band'), summary.get('subject_age_band')):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _summary_channels(summary):
    for key in ('scalp_channels', 'channels'):
        values = summary.get(key)
        if isinstance(values, (list, tuple)) and values:
            sites = []
            for name in values:
                site = _canonical_site(name)
                if site and site not in sites:
                    sites.append(site)
            if sites:
                return sites
    return []


def _resolve_raw_path(session_folder, summary):
    raw_name = summary.get('raw_csv')
    if isinstance(raw_name, str) and raw_name.strip():
        candidate = session_folder / raw_name.strip()
        if candidate.is_file():
            return candidate
    exact = session_folder / 'raw.csv'
    if exact.is_file():
        return exact
    try:
        candidates = sorted(session_folder.glob('raw*.csv'))
    except OSError:
        return None
    return candidates[0] if candidates else None


def _raw_channel_name(column):
    key = _clean_key(column)
    if not key:
        return None
    if key.startswith('raw_'):
        key = key[4:]
    changed = True
    while changed:
        changed = False
        for suffix in _RAW_SUFFIXES:
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                changed = True
    if key in ('sample_index', 'relative_time_s', 'packet_number', 'index',
               'time_s', 'sample', 'packet', 'timestamp'):
        return None
    return _canonical_site(key)


def _read_raw(path):
    """``(samples, channels)`` from a raw CSV; ValueError when unreadable."""
    with open(path, encoding='utf-8', newline='') as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError('raw CSV is empty') from exc
        indices, channels = [], []
        for index, column in enumerate(header):
            site = _raw_channel_name(column)
            if site is None or site in channels:
                continue
            indices.append(index)
            channels.append(site)
        if len(channels) < qeeg.MIN_CHANNELS:
            raise ValueError(f'raw CSV has {len(channels)} scalp channel(s); '
                             f'{qeeg.MIN_CHANNELS} are needed')
        samples = []
        for line, row in enumerate(reader, start=2):
            try:
                values = [float(row[index]) for index in indices]
            except (IndexError, TypeError, ValueError) as exc:
                raise ValueError(f'raw CSV line {line} is not numeric') from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f'raw CSV line {line} holds a non-finite sample')
            samples.append(values)
    if len(samples) < qeeg.MIN_SEGMENT_SAMPLES:
        raise ValueError('raw CSV holds too few samples for an analysis window')
    return np.asarray(samples, dtype=float), tuple(channels)


def _grid_values(metrics):
    """Per-row/band/site values of a Metrics, via the public grid() path."""
    table = qeeg.grid(metrics)
    out = {}
    for row, bands in table['rows'].items():
        out[row] = {}
        for band, cells in bands.items():
            values = {}
            for site, cell in cells.items():
                value = _number(cell.get('value'))
                if value is not None:
                    values[site] = value
            out[row][band] = values
    return out


def _mean_sd(values):
    count = len(values)
    mean = sum(values) / count
    if count < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance)


def _fs_of(summary):
    for key in ('fs_hz', 'fs', 'sampling_hz', 'sample_rate_hz'):
        number = _number(summary.get(key))
        if number is not None and number > 0:
            return number
    return None


def _session_values(session_folder, summary, notes):
    """Recomputed grid values for one session, or None with a note."""
    raw_path = _resolve_raw_path(session_folder, summary)
    if raw_path is None:
        notes.append(f'{session_folder.name}: no raw CSV')
        return None
    fs = _fs_of(summary)
    if fs is None:
        notes.append(f'{session_folder.name}: no sampling rate recorded')
        return None
    try:
        samples, channels = _read_raw(raw_path)
    except (OSError, ValueError) as exc:
        notes.append(f'{session_folder.name}: {exc}')
        return None
    try:
        metrics = qeeg.compute_metrics(samples, fs, channels)
    except (ValueError, FloatingPointError) as exc:
        notes.append(f'{session_folder.name}: metrics failed ({exc})')
        return None
    if len(metrics.get('usable_channels') or []) < qeeg.MIN_CHANNELS:
        notes.append(f'{session_folder.name}: fewer than {qeeg.MIN_CHANNELS} '
                     f'usable channels')
        return None
    return _grid_values(metrics)


def _iter_session_folders(subject_folder):
    try:
        entries = sorted(subject_folder.iterdir(), key=lambda path: path.name,
                         reverse=True)
    except OSError:
        return []
    return [entry for entry in entries if entry.is_dir()]


def build_local_norm(root, min_n=MIN_LOCAL_SUBJECTS, condition=None,
                     age_band=None, device_scope='any'):
    """Accumulate a local norm from this lab's saved sessions.

    Returns ``(NormFile, '')`` on success, ``(None, reason)`` when the sample
    is too small or the selection is inconsistent. The scan walks
    ``root/<subject>/<session>/`` (``root`` may also be the parent that
    contains a ``subjects`` folder):

    * a session counts only when its summary is valid, is not simulated, has
      the requested condition, has an age inside ``age_band``, and its scalp
      channel count matches ``device_scope``;
    * every session is recomputed from its raw CSV with
      ``qeeg.compute_metrics`` - cached band numbers are never trusted - and
      the SHA-256 of ``qeeg.py`` is stamped as ``provenance['qeeg_version']``;
    * each subject contributes ONE row: the mean of that subject's matching
      sessions per (row, band, site); across subjects the table holds the mean
      and the sample SD (ddof=1) of those subject values. A site present for
      fewer than 2 subjects has no spread and is omitted.

    ``condition`` and ``age_band`` are required: a norm without them would be
    compared against the wrong recordings. ``device_scope='any'`` is accepted
    only when every included session has the same channel count - mixing a
    4-channel and a 21-channel device in one table is refused outright.
    """
    if isinstance(min_n, bool) or not isinstance(min_n, (int, np.integer)) or int(min_n) < 2:
        raise ValueError('min_n must be an integer >= 2')
    min_n = int(min_n)
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError('condition is required for a local norm')
    condition = normalize_condition(condition)
    if not isinstance(age_band, str) or not age_band.strip():
        raise ValueError('age_band is required for a local norm')
    age_band = age_band.strip()
    device_scope = _validated_scope(device_scope)
    expected_channels = _scope_count(device_scope)

    root = Path(root)
    if root.name != 'subjects' and (root / 'subjects').is_dir():
        root = root / 'subjects'
    if not root.is_dir():
        return None, f'recording root not found: {root}'

    excluded = OrderedDict()
    notes = []
    subject_entries = []
    subject_values = []
    device_counts = set()

    def exclude(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    try:
        subject_folders = sorted(entry for entry in root.iterdir() if entry.is_dir())
    except OSError as exc:
        return None, f'could not read recording root: {exc}'

    for subject_folder in subject_folders:
        session_ids = []
        per_session_values = []
        subject_age = None
        subject_band = None
        subject_channels = []
        for session_folder in _iter_session_folders(subject_folder):
            summary = _read_summary(session_folder / 'summary.json')
            if summary is None:
                continue
            if summary.get('simulated'):
                exclude('simulated session')
                continue
            if not _session_valid(summary):
                exclude('summary not valid')
                continue
            session_condition = _summary_condition(summary)
            if session_condition is None:
                exclude('condition not recorded')
                continue
            if session_condition != condition:
                exclude('condition mismatch')
                continue

            age = _summary_age(summary)
            stored_band = _summary_age_band(summary)
            if age is None and stored_band is None:
                exclude('age not recorded')
                continue
            if age is not None:
                if not age_in_band(age, age_band):
                    exclude('age band mismatch')
                    continue
                band = stored_band or age_band_for(age) or age_band
                if stored_band and _band_text(stored_band) != _band_text(age_band):
                    notes.append(f'{session_folder.name}: stored age_band '
                                 f'{stored_band!r} disagrees with stored age '
                                 f'{age:g}; age was used')
            else:
                if _band_text(stored_band) != _band_text(age_band):
                    exclude('age band mismatch')
                    continue
                band = stored_band

            channels = _summary_channels(summary)
            if not channels:
                exclude('channels not recorded / not on the montage')
                continue
            if expected_channels is not None and len(channels) != expected_channels:
                exclude('device scope mismatch')
                continue

            values = _session_values(session_folder, summary, notes)
            if values is None:
                exclude('recompute failed')
                continue

            device_counts.add(len(channels))
            session_ids.append(str(summary.get('session_id') or session_folder.name))
            per_session_values.append(values)
            subject_age = age if subject_age is None else subject_age
            subject_band = band if subject_band is None else subject_band
            if not subject_channels:
                subject_channels = channels

        if not per_session_values:
            continue

        # One row per subject: mean of that subject's sessions, per cell.
        collected = {}
        for values in per_session_values:
            for row, bands in values.items():
                for band, sites in bands.items():
                    for site, value in sites.items():
                        collected.setdefault(row, {}).setdefault(band, {}) \
                            .setdefault(site, []).append(value)
        subject_table = {
            row: {band: {site: _mean_sd(cell)[0] for site, cell in sites.items()}
                  for band, sites in bands.items()}
            for row, bands in collected.items()
        }
        subject_values.append(subject_table)
        subject_entries.append({
            'subject': subject_folder.name,
            'sessions': session_ids,
            'age': subject_age,
            'age_band': subject_band,
            'condition': condition,
            'device_scope': _scope_for_count(len(subject_channels)),
            'channels': list(subject_channels),
        })

    if len(device_counts) > 1:
        return None, (f'mixed device channel counts in the selected sessions '
                      f'({", ".join(_scope_for_count(c) for c in sorted(device_counts))}); '
                      f'choose a device_scope')

    n = len(subject_values)
    if n < min_n:
        detail = ', '.join(f'{reason}: {count}' for reason, count in excluded.items())
        reason = (f'local norm needs at least {min_n} subjects; found {n} '
                  f'eligible subject(s)')
        if detail:
            reason += f' (excluded — {detail})'
        return None, reason

    # Across subjects: mean and sample SD per (row, band, site).
    series = {}
    for table in subject_values:
        for row, bands in table.items():
            for band, sites in bands.items():
                for site, value in sites.items():
                    series.setdefault(row, {}).setdefault(band, {}) \
                        .setdefault(site, []).append(value)

    rows = OrderedDict()
    site_counts = OrderedDict()
    for row in _ROWS:
        if row not in series:
            continue
        rows[row] = OrderedDict()
        for band in _BANDS:
            if band not in series[row]:
                continue
            cells = OrderedDict()
            for site in sorted(series[row][band],
                               key=lambda name: (_MONTAGE_ORDER.get(
                                   name.split('/')[0], 999), name)):
                values = series[row][band][site]
                if len(values) < 2:
                    continue
                mean, sd = _mean_sd(values)
                cells[site] = {'mean': mean, 'sd': sd}
                site_counts[site] = len(values)
            if cells:
                rows[row][band] = cells

    if not rows:
        return None, ('no site had values from at least 2 subjects; nothing '
                      'can be accumulated')

    provenance = {
        'root': str(root),
        'built_at': _iso_now(),
        'qeeg_version': qeeg_version(),
        'method': ('recomputed each session from its raw CSV with '
                   'qeeg.compute_metrics; one row per subject (mean of that '
                   "subject's sessions); mean and sample SD (ddof=1) across "
                   'subjects; sites with fewer than 2 subjects omitted'),
        'min_n': min_n,
        'subjects': subject_entries,
        'excluded': dict(excluded),
        'site_subjects': dict(site_counts),
        'notes': _capped(notes),
    }
    norm = NormFile(
        schema=SCHEMA, source='local',
        label=f'local norm · {condition} · {age_band}', n=n,
        age_band=age_band, condition=condition, device_scope=device_scope,
        created_at=_iso_now(), rows=rows, provenance=provenance,
    )
    return norm, ''
