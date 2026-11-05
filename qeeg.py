"""Quantitative-EEG grid numerics: 5 bands x 5 rows, plus z-scoring references.

This module is PURE NUMERICS. It does no acquisition, opens no device, draws
nothing, and never touches the 1 s UI timer. Input samples are microvolts,
shape ``(n_samples, n_channels)``; the caller passes scalp channels only
(``devices.scalp_channels``), so auxiliary channels can never leak into a head
map.

Bands are the five classic 1-45 Hz bands; rows are the five rows of the
Z-Scored FFT Summary sheet: absolute power, relative power, amplitude
asymmetry, coherence and phase lag.

Welch settings (both power and cross-spectra)
---------------------------------------------
Hann window, ``nperseg = min(1024, n_samples)`` with 50 % overlap, constant
detrend, density scaling. Absolute power is the integral of the PSD over the
band (uV^2); relative power is that integral divided by the integral over
1-45 Hz times 100. Peak frequency is the argmax of the band's PSD bins.

``n_epochs`` is the number of Welch segments averaged (50 % overlap), not a
count of clinical epochs; ``window_s`` is n_samples / fs.

Coherence
---------
Per pair per band, magnitude-squared coherence
``|Pxy|^2 / (Pxx * Pyy)`` averaged over the band's frequency bins, where
``Pxy`` is the Welch cross-spectrum. The per-electrode value is the MEAN OF
THAT ELECTRODE'S PAIR COHERENCES (all pairs that contain it), i.e. a global
coherence for the site within this recording. Pair sets are ranked by
schematic electrode distance and truncated to ``pair_max`` so the O(n^2)
network stays bounded on 21 electrodes; every electrode keeps its nearest
neighbours and the cap only drops the longest-distance pairs. That ranking is
a compute bound, not an anatomical claim.

Phase
-----
Pair phase is the angle of the summed cross-spectrum over the band's bins
(degrees) - a magnitude-weighted circular mean, because a plain arithmetic
mean of angles is undefined for near-zero bins and wraps at +-180 deg. The
cross-spectrum is oriented so it is ``phase(first) - phase(second)``; when the
two sites are on opposite hemispheres the left-hemisphere site is always
first, so POSITIVE MEANS THE LEFT SITE LEADS THE RIGHT (the right signal
arrives late). For same-side and midline pairs the caller's channel order is
kept and the sign has no anatomical meaning. The per-electrode value is the
arithmetic mean of the signed pair phases of the pairs containing it (as the
contract specifies); a pair's signed value is mirrored onto both of its
sites, exactly as the asymmetry row is, so both sites show the same
left-minus-right number. This mean can cancel near +-180 deg, so read it
together with the pair table. ``phase_abs`` carries the mean absolute lag for
display. Phase is only meaningful where coherence is high; low-coherence
pairs have a numerically defined but meaningless angle.

Amplitude asymmetry
-------------------
Homologous 10-20 pairs only (Fp1/Fp2, Fp7/Fp8, F3/F4, F7/F8, C3/C4, T3/T4,
T5/T6, P3/P4, O1/O2). Value = ``(L - R) / ((L + R) / 2) * 100`` on absolute
band power; sign is left minus right. T4/T6 is NOT a homologous pair (both
sites are right hemisphere) and is deliberately absent. A pair is skipped
unless both channels are present and its mean power is > 0; nothing is
invented for missing electrodes. In ``grid`` the pair's signed percentage is
mirrored onto both sites, because the difference belongs to the pair.

Honesty rule
------------
There is NO normative clinical database here. Every grid carries a reference
block whose kind is one of ``session`` (mean/SD across channels of this
recording), ``baseline`` (a saved baseline of the same subject/condition),
``norm`` (a local norm file with >= 20 subjects and an age band) or ``none``
(raw values, no colour). z-scores exist only against the selected reference;
``zscore`` returns None when the SD is missing or non-positive, and ``grid``
returns None - never 0 and never a colour - wherever a value cannot be
computed. Nothing in this module is a diagnosis.

Guards
------
Fewer than 3 channels, less than 2 s of data, or fewer than 3 usable
(non-finite/flat) channels returns an empty Metrics with an explanatory
``notes`` list; it never raises for insufficient data. A malformed array
shape or an invalid ``fs`` is a caller bug and does raise ValueError.
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import signal

BANDS = OrderedDict([
    ('Delta', (1, 4)), ('Theta', (4, 8)), ('Alpha', (8, 13)),
    ('Beta', (13, 30)), ('HighBeta', (30, 45)),
])
ROWS = ('absolute', 'relative', 'asymmetry', 'coherence', 'phase')

# Homologous 10-20 sites, frontal to posterior. T4/T6 is not homologous (both
# right hemisphere) and is excluded on purpose.
HOMOLOGOUS_PAIRS = (
    ('Fp1', 'Fp2'), ('Fp7', 'Fp8'), ('F3', 'F4'), ('F7', 'F8'),
    ('C3', 'C4'), ('T3', 'T4'), ('T5', 'T6'), ('P3', 'P4'), ('O1', 'O2'),
)

REFERENCE_SCHEMA = 'lab.norm/1'
REFERENCE_KINDS = ('session', 'baseline', 'norm', 'none')
GRID_LIMITS = (-3.0, 3.0)
DEFAULT_PAIR_MAX = 200
MIN_CHANNELS = 3
MIN_WINDOW_S = 2.0
MIN_SEGMENT_SAMPLES = 8

# Blue below, green exactly at zero, yellow then red above. Interpolated in
# sRGB; the limits clamp every value outside the ramp.
_BLUE = (47, 95, 208)
_GREEN = (26, 168, 79)
_YELLOW = (232, 197, 26)
_RED = (200, 52, 28)
_NO_DATA_COLOR = '#b3bcc6'

_TOTAL_BAND = (1.0, 45.0)

try:  # numpy < 2 keeps the old spelling
    _trapezoid = np.trapezoid
except AttributeError:  # pragma: no cover - only on old numpy
    _trapezoid = np.trapz

__all__ = [
    'BANDS', 'ROWS', 'HOMOLOGOUS_PAIRS', 'REFERENCE_SCHEMA', 'GRID_LIMITS',
    'compute_metrics', 'grid', 'zscore', 'reference_from_session',
    'reference_from_baseline', 'save_reference', 'load_reference',
    'color_for_z', 'ratios', 'pair_label',
]


# --------------------------------------------------------------------------
# small validators
# --------------------------------------------------------------------------
def _channel_tuple(channels):
    """Validate a channel-name sequence (duplicates would corrupt the maps)."""
    if isinstance(channels, (str, bytes)):
        raise ValueError('channels must be a sequence of channel names, not one string')
    try:
        names = tuple(channels)
    except TypeError as exc:
        raise ValueError('channels must be a sequence of channel names') from exc
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError('channels must be a non-empty sequence of non-empty names')
    if len(set(names)) != len(names):
        raise ValueError('channels must not repeat a name')
    return names


def _band_table(bands):
    """Validate a band mapping name -> (low, high) and return an OrderedDict."""
    if bands is None:
        return OrderedDict(BANDS)
    if not isinstance(bands, Mapping) or not bands:
        raise ValueError('bands must be a non-empty mapping of name -> (low, high)')
    table = OrderedDict()
    for name, edges in bands.items():
        if not isinstance(name, str) or not name:
            raise ValueError('band names must be non-empty strings')
        try:
            low, high = edges
            low, high = float(low), float(high)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'band {name!r} must be a (low, high) pair') from exc
        if not (math.isfinite(low) and math.isfinite(high)) or low < 0 or high <= low:
            raise ValueError(f'band {name!r} must satisfy 0 <= low < high')
        table[name] = (low, high)
    return table


def _pair_cap(pair_max):
    if pair_max is None:
        return None
    if isinstance(pair_max, bool) or not isinstance(pair_max, (int, np.integer)):
        raise ValueError('pair_max must be a positive integer or None')
    if int(pair_max) < 1:
        raise ValueError('pair_max must be a positive integer or None')
    return int(pair_max)


def _float_or_none(value):
    """Finite float or None; booleans and non-numeric input become None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pair_label(left, right):
    """Stable dict key for a pair: ``'F3/F4'``."""
    return f'{left}/{right}'


def _mean(values):
    return float(np.mean(values)) if values else None


# --------------------------------------------------------------------------
# spectral building blocks
# --------------------------------------------------------------------------
def _band_integral(freqs, psd, low, high):
    """Integral of a PSD over [low, high], with exact band edges.

    The edge values are linearly interpolated from the frequency grid so a
    band boundary that falls between bins is not silently moved. Frequencies
    above Nyquist are clipped: a band that runs past the sampling limit only
    integrates the part that was actually measured.
    """
    if freqs.size < 2 or not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        return 0.0
    top = float(freqs[-1])
    if low >= top:
        return 0.0
    high = min(high, top)
    inside = (freqs > low) & (freqs < high)
    grid = np.concatenate(([low], freqs[inside], [high]))
    values = np.interp(grid, freqs, psd)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return max(float(_trapezoid(values, grid)), 0.0)


def _band_peak(freqs, psd, low, high):
    """Frequency of the largest PSD bin inside the band, or None."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return None
    inside = psd[mask]
    if not np.any(inside > 0):
        return None
    # Index the band's own frequency grid: argmax of the masked PSD is an
    # index into `inside`, not into the full freqs array.
    return float(freqs[mask][int(np.argmax(inside))])


def _welch_options(n_samples, fs):
    nperseg = int(min(1024, n_samples))
    return dict(fs=float(fs), window='hann', nperseg=nperseg,
                noverlap=nperseg // 2, detrend='constant',
                scaling='density', return_onesided=True)


def _segment_count(n_samples, nperseg):
    step = max(nperseg - nperseg // 2, 1)
    return int(1 + max(n_samples - nperseg, 0) // step)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _empty_metrics(channels, fs, n_samples, bands, notes, n_epochs=0):
    """A Metrics with the full documented shape but no values."""
    empty_band = {'absolute': {}, 'relative': {}, 'peak_hz': {}}
    return {
        'bands': {band: dict(empty_band) for band in bands},
        'asymmetry': {band: {} for band in bands},
        'coherence': {band: {} for band in bands},
        'phase': {band: {} for band in bands},
        'phase_abs': {band: {} for band in bands},
        'pairs': {band: {} for band in bands},
        'n_samples': int(n_samples),
        'n_epochs': int(n_epochs),
        'channels': list(channels),
        'fs': float(fs),
        'window_s': float(n_samples / fs) if fs > 0 else 0.0,
        'usable_channels': [],
        'notes': list(notes),
    }


def compute_metrics(samples, fs, channels, bands=BANDS, pair_max=DEFAULT_PAIR_MAX):
    """Band powers, asymmetry, coherence and phase for one analysis window.

    ``samples`` is ``(n_samples, n_channels)`` in microvolts; ``channels``
    names the columns (scalp channels only - the caller slices). Insufficient
    data returns an empty Metrics with ``usable_channels == []`` and a
    ``notes`` list instead of raising. Channels carrying a non-finite sample
    or a flat (zero-variance) signal are excluded from every calculation and
    named in ``notes``; they are not interpolated, guessed or zero-filled.

    The returned dict is JSON-shaped: band powers are plain floats keyed by
    channel, pairs are keyed ``'Left/Right'``, and every per-electrode
    coherence/phase value is an average over the pairs that contain that
    electrode (see the module docstring for the exact definitions).
    """
    channels = _channel_tuple(channels)
    bands = _band_table(bands)
    pair_max = _pair_cap(pair_max)
    try:
        fs = float(fs)
    except (TypeError, ValueError) as exc:
        raise ValueError('fs must be a positive number') from exc
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError('fs must be a positive number')

    try:
        data = np.asarray(samples, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('samples must be a numeric array') from exc
    if data.ndim == 1 and len(channels) == 1:
        data = data.reshape(-1, 1)
    if data.size == 0 and data.ndim == 1:
        # An empty buffer is insufficient data, not a malformed shape: let the
        # window guard below return notes instead of raising.
        data = data.reshape(0, len(channels))
    if data.ndim != 2:
        raise ValueError('samples must have shape (n_samples, n_channels)')
    if data.shape[1] != len(channels):
        raise ValueError(
            f'samples has {data.shape[1]} column(s) but {len(channels)} channel(s)')
    n_samples = int(data.shape[0])
    window_s = n_samples / fs

    if len(channels) < MIN_CHANNELS:
        return _empty_metrics(
            channels, fs, n_samples, bands,
            [f'{MIN_CHANNELS} channels are required for a qEEG grid; '
             f'{len(channels)} supplied - no heads are drawn and 2 points are '
             f'never interpolated.'], n_epochs=0)
    if window_s < MIN_WINDOW_S:
        return _empty_metrics(
            channels, fs, n_samples, bands,
            [f'at least {MIN_WINDOW_S:g} s of data are required; this window is '
             f'{window_s:.2f} s - no grid is computed.'], n_epochs=0)
    if n_samples < MIN_SEGMENT_SAMPLES:
        return _empty_metrics(
            channels, fs, n_samples, bands,
            [f'at least {MIN_SEGMENT_SAMPLES} samples are required for a Welch '
             f'segment; {n_samples} supplied.'], n_epochs=0)

    usable, dropped = [], []
    for index, name in enumerate(channels):
        column = data[:, index]
        if not np.all(np.isfinite(column)) or float(np.std(column)) <= 0.0:
            dropped.append(name)
        else:
            usable.append(name)

    notes = []
    if dropped:
        notes.append('excluded channel(s) ' + ', '.join(dropped) +
                     ': non-finite samples or a flat (zero-variance) signal.')
    if len(usable) < MIN_CHANNELS:
        notes.append(f'{MIN_CHANNELS} usable channels are required for a qEEG '
                     f'grid; {len(usable)} clean channel(s) supplied - no heads '
                     f'are drawn and 2 points are never interpolated.')
        return _empty_metrics(channels, fs, n_samples, bands, notes, n_epochs=0)

    options = _welch_options(n_samples, fs)
    segments = _segment_count(n_samples, options['nperseg'])
    column_index = {name: i for i, name in enumerate(usable)}
    matrix = data[:, [column_index[name] for name in usable]]

    freqs, psd = signal.welch(matrix, axis=0, **options)
    psd = np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)

    absolute = {band: {} for band in bands}
    relative = {band: {} for band in bands}
    peak = {band: {} for band in bands}
    totals = {}
    for j, name in enumerate(usable):
        totals[name] = _band_integral(freqs, psd[:, j], *_TOTAL_BAND)
        for band, (low, high) in bands.items():
            power = _band_integral(freqs, psd[:, j], low, high)
            absolute[band][name] = float(power)
            total = totals[name]
            relative[band][name] = (float(100.0 * power / total)
                                    if total > 0 else None)
            peak[band][name] = _band_peak(freqs, psd[:, j], low, high)

    asymmetry = {band: {} for band in bands}
    for left, right in HOMOLOGOUS_PAIRS:
        if left not in column_index or right not in column_index:
            continue
        for band in bands:
            left_power = absolute[band][left]
            right_power = absolute[band][right]
            half = (left_power + right_power) / 2.0
            if half > 0:
                asymmetry[band][pair_label(left, right)] = float(
                    (left_power - right_power) / half * 100.0)

    pairs = _ordered_pairs(usable, pair_max)
    pair_rows = {band: {} for band in bands}
    coherence_acc = {band: {name: [] for name in usable} for band in bands}
    phase_acc = {band: {name: [] for name in usable} for band in bands}
    for left, right in pairs:
        j_left = column_index[left]
        j_right = column_index[right]
        # csd(right, left) has angle phase(left) - phase(right): positive when
        # the left/first site leads the right/second site.
        _, cross = signal.csd(matrix[:, j_right], matrix[:, j_left], **options)
        power_product = psd[:, j_left] * psd[:, j_right]
        gamma = np.zeros_like(power_product)
        valid = power_product > 0
        gamma[valid] = (np.abs(cross[valid]) ** 2) / power_product[valid]
        gamma = np.clip(np.nan_to_num(gamma, nan=0.0, posinf=0.0), 0.0, 1.0)
        for band, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high) & valid
            if not mask.any():
                continue
            value = float(np.mean(gamma[mask]))
            phase_deg = float(np.degrees(np.angle(np.sum(cross[mask])))) if mask.any() else None
            if phase_deg is None:
                continue
            pair_rows[band][pair_label(left, right)] = {
                'coherence': value, 'phase_deg': phase_deg}
            coherence_acc[band][left].append(value)
            coherence_acc[band][right].append(value)
            phase_acc[band][left].append(phase_deg)
            phase_acc[band][right].append(phase_deg)

    coherence = {band: {name: _mean(values)
                        for name, values in coherence_acc[band].items() if values}
                 for band in bands}
    phase = {band: {name: _mean(values)
                    for name, values in phase_acc[band].items() if values}
             for band in bands}
    phase_abs = {band: {name: float(np.mean(np.abs(values)))
                        for name, values in phase_acc[band].items() if values}
                 for band in bands}

    return {
        'bands': {band: {'absolute': absolute[band],
                         'relative': relative[band],
                         'peak_hz': peak[band]} for band in bands},
        'asymmetry': asymmetry,
        'coherence': coherence,
        'phase': phase,
        'phase_abs': phase_abs,
        'pairs': pair_rows,
        'n_samples': n_samples,
        'n_epochs': segments,
        'channels': list(channels),
        'fs': float(fs),
        'window_s': float(window_s),
        'usable_channels': list(usable),
        'notes': notes,
    }


def _ordered_pairs(usable, pair_max):
    """All pairs, nearest schematic neighbours first, truncated to pair_max.

    Left-hemisphere sites are put first on cross-hemisphere pairs so the phase
    sign always reads left minus right; otherwise the caller's order is kept.
    """
    from topomap import ELECTRODE_POSITIONS

    positions = {}
    for name in usable:
        try:
            positions[name] = ELECTRODE_POSITIONS.get(name)
        except TypeError:
            positions[name] = None

    ranked = []
    for i, left in enumerate(usable):
        x_left = positions.get(left)
        for j in range(i + 1, len(usable)):
            right = usable[j]
            x_right = positions.get(right)
            if x_left is None or x_right is None:
                distance = float('inf')
            else:
                distance = math.hypot(x_left[0] - x_right[0],
                                      x_left[1] - x_right[1])
            ranked.append((distance, i, j))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    if pair_max is not None:
        ranked = ranked[:pair_max]

    pairs = []
    for _, i, j in ranked:
        left, right = usable[i], usable[j]
        a, b = positions.get(left), positions.get(right)
        if a is not None and b is not None:
            if a[0] < -1e-9 and b[0] > 1e-9:
                pairs.append((left, right))
                continue
            if b[0] < -1e-9 and a[0] > 1e-9:
                pairs.append((right, left))
                continue
        pairs.append((left, right))
    return pairs


# --------------------------------------------------------------------------
# references and z-scores
# --------------------------------------------------------------------------
def zscore(value, mean, sd):
    """(value - mean) / sd, or None when sd is missing/non-positive.

    A standard deviation of zero means the reference has no spread, so a
    z-score would be a division by zero dressed up as a number. Returning
    None keeps the grid honest: the cell is shown raw, never coloured.
    """
    value = _float_or_none(value)
    mean = _float_or_none(mean)
    sd = _float_or_none(sd)
    if value is None or mean is None or sd is None or sd <= 0:
        return None
    return float((value - mean) / sd)


def _row_values(metrics):
    """Every grid row's per-channel value, before any z-scoring."""
    channels = list(metrics.get('usable_channels') or [])
    bands = list(metrics.get('bands') or {})
    out = {row: {band: {} for band in bands} for row in ROWS}

    for band in bands:
        band_data = metrics['bands'][band]
        for name in channels:
            out['absolute'][band][name] = _float_or_none(
                band_data['absolute'].get(name))
            out['relative'][band][name] = _float_or_none(
                band_data['relative'].get(name))

    for band in bands:
        # Coherence and phase are already per-electrode means over pairs.
        for name in channels:
            out['coherence'][band][name] = _float_or_none(
                metrics.get('coherence', {}).get(band, {}).get(name))
            out['phase'][band][name] = _float_or_none(
                metrics.get('phase', {}).get(band, {}).get(name))

    for band in bands:
        per_channel = {name: [] for name in channels}
        for pair, pct in (metrics.get('asymmetry', {}).get(band) or {}).items():
            if not isinstance(pair, str) or '/' not in pair:
                continue
            left, right = pair.split('/', 1)
            value = _float_or_none(pct)
            if value is None:
                continue
            for name in (left, right):
                if name in per_channel:
                    per_channel[name].append(value)
        for name, values in per_channel.items():
            # The difference belongs to the pair, so it is mirrored onto both
            # sites; a site with no homologous partner stays None.
            out['asymmetry'][band][name] = _mean(values)
    return out


def _canonical_label(kind, label, n, age_band):
    if kind == 'session':
        return 'self-referenced vs this session\'s channel mean'
    if kind == 'baseline':
        return f'vs this subject\'s saved baseline ({label})' if label \
            else 'vs this subject\'s saved baseline'
    if kind == 'norm':
        text = f'vs local norm n={n}, age {age_band}'
        if label and label not in text:
            text += f' — {label}'
        return text
    return 'raw values — no reference selected'


def _stats_by_row(metrics, rows):
    """Mean/SD across channels for every (row, band), per channel key.

    The same scalar is stored under every channel key: a session reference is
    one distribution per (row, band), and every channel is z-scored against
    it. Values that do not exist for a channel (e.g. a midline asymmetry) are
    excluded from that distribution, not counted as zero.
    """
    channels = list(metrics.get('usable_channels') or [])
    bands = list(metrics.get('bands') or {})
    values = _row_values(metrics)
    mean, sd = {}, {}
    for row in rows:
        if row not in values:
            continue
        mean[row], sd[row] = {}, {}
        for band in bands:
            column = [values[row][band][name] for name in channels
                      if values[row][band].get(name) is not None]
            if len(column) >= 2:
                centre = float(np.mean(column))
                spread = float(np.std(column, ddof=1))
                # A spread at round-off level relative to the mean is not a
                # distribution: storing it would let the display produce an
                # arbitrary huge z for what is really a constant row.
                if spread <= 1e-12 * max(1.0, abs(centre)):
                    spread = 0.0
            elif len(column) == 1:
                centre, spread = float(column[0]), 0.0
            else:
                continue
            mean[row][band] = {name: centre for name in channels}
            sd[row][band] = {name: spread for name in channels}
    return mean, sd


def _make_reference(metrics, kind, label, n, age_band=None):
    if kind not in REFERENCE_KINDS or kind == 'none':
        raise ValueError('kind must be session, baseline or norm')
    mean, sd = _stats_by_row(metrics, ROWS)
    if not mean:
        raise ValueError('metrics contain no values to reference')
    return {
        'schema': REFERENCE_SCHEMA,
        'kind': kind,
        'label': str(label or ''),
        'n': int(n),
        'age_band': None if age_band is None else str(age_band),
        'bands': list(metrics.get('bands') or {}),
        'rows': list(ROWS),
        'mean': mean,
        'sd': sd,
        'created_at': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def reference_from_session(metrics):
    """Reference that z-scores each channel against this session's own mean.

    There is no external database behind this: the mean and SD of each
    (row, band) are taken across the usable channels of this recording.
    ``n`` is the number of channels averaged, not the number of subjects.
    """
    channels = list(metrics.get('usable_channels') or [])
    if len(channels) < MIN_CHANNELS:
        raise ValueError('a session reference needs at least 3 usable channels')
    reference = _make_reference(metrics, 'session',
                                _canonical_label('session', '', len(channels), None),
                                len(channels))
    return reference


def reference_from_baseline(metrics, label, n):
    """Reference built from a saved baseline recording of this subject.

    ``label`` identifies the baseline (typically its date and condition);
    ``n`` is the number of baseline observations the caller used. Neither is
    invented here.
    """
    channels = list(metrics.get('usable_channels') or [])
    if len(channels) < MIN_CHANNELS:
        raise ValueError('a baseline reference needs at least 3 usable channels')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 1:
        raise ValueError('n must be a positive integer (baseline observations)')
    reference = _make_reference(metrics, 'baseline', str(label or ''), int(n))
    return reference


def _nested_numbers(source):
    if not isinstance(source, Mapping):
        return {}
    out = {}
    for outer, rows in source.items():
        if not isinstance(outer, str) or not isinstance(rows, Mapping):
            continue
        out[outer] = {}
        for inner, values in rows.items():
            if not isinstance(inner, str) or not isinstance(values, Mapping):
                continue
            clean = {}
            for name, value in values.items():
                number = _float_or_none(value)
                if isinstance(name, str) and number is not None:
                    clean[name] = number
            out[outer][inner] = clean
    return out


def _validated_reference(reference, require_schema=False):
    """Normalise and validate a Reference; ValueError on anything dishonest."""
    if not isinstance(reference, Mapping):
        raise ValueError('reference must be a mapping')
    schema = reference.get('schema', REFERENCE_SCHEMA if not require_schema else None)
    if schema != REFERENCE_SCHEMA:
        raise ValueError(f'reference schema must be {REFERENCE_SCHEMA!r}')
    kind = reference.get('kind')
    if kind not in REFERENCE_KINDS:
        raise ValueError('reference kind must be one of ' + ', '.join(REFERENCE_KINDS))
    if 'n' not in reference or reference.get('n') is None:
        raise ValueError("reference is missing 'n'")
    n = reference.get('n')
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 0:
        raise ValueError("reference 'n' must be a non-negative integer")
    n = int(n)

    label = reference.get('label') or ''
    if not isinstance(label, str):
        raise ValueError('reference label must be text')
    age_band = reference.get('age_band')
    if age_band is not None and not isinstance(age_band, str):
        raise ValueError('reference age_band must be text or None')

    mean = _nested_numbers(reference.get('mean'))
    sd = _nested_numbers(reference.get('sd'))
    rows = reference.get('rows')
    bands = reference.get('bands')
    rows = [str(r) for r in rows] if isinstance(rows, (list, tuple)) else list(mean)
    bands = [str(b) for b in bands] if isinstance(bands, (list, tuple)) else []

    if kind == 'norm':
        if n < 20:
            raise ValueError('a norm reference needs at least 20 subjects')
        if not age_band or not str(age_band).strip():
            raise ValueError('a norm reference needs a recorded age band')
    if kind != 'none' and not mean:
        raise ValueError(f'a {kind} reference needs per-channel mean values')

    created = reference.get('created_at')
    if not isinstance(created, str) or not created:
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return {
        'schema': REFERENCE_SCHEMA,
        'kind': kind,
        'label': label,
        'n': n,
        'age_band': None if age_band is None else str(age_band),
        'bands': bands,
        'rows': rows,
        'mean': mean,
        'sd': sd,
        'created_at': created,
    }


def save_reference(reference, path):
    """Write a Reference as JSON (atomic replace) and return the path.

    The explicit schema, kind, n, age band, bands, rows, mean and sd are all
    written, so a file is self-describing and a reader cannot mistake it for
    a normative database. A norm reference with fewer than 20 subjects or no
    age band is refused.
    """
    clean = _validated_reference(reference)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_text(json.dumps(clean, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, target)
    return str(target)


def load_reference(path):
    """Read a Reference JSON file, validating schema, kind, n and honesty rules.

    Refuses (ValueError) anything that is not a lab.norm/1 reference or that
    lacks ``kind``/``n``; a norm file below 20 subjects or without an age band
    is refused rather than relabelled.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding='utf-8')
    except OSError as exc:
        raise ValueError(f'could not read reference file: {exc}') from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'reference file is not valid JSON: {exc}') from exc
    if not isinstance(raw, Mapping):
        raise ValueError('reference file must contain a JSON object')
    if 'kind' not in raw:
        raise ValueError("reference file has no 'kind'")
    if 'n' not in raw or raw.get('n') is None:
        raise ValueError("reference file has no 'n'")
    return _validated_reference(raw, require_schema=True)


# --------------------------------------------------------------------------
# grid and colours
# --------------------------------------------------------------------------
def _reference_block(reference):
    if reference is None:
        return {'kind': 'none', 'label': _canonical_label('none', '', 0, None), 'n': 0}
    if not isinstance(reference, Mapping):
        raise ValueError('reference must be a mapping or None')
    kind = reference.get('kind', 'none')
    if kind not in REFERENCE_KINDS:
        raise ValueError('reference kind must be one of ' + ', '.join(REFERENCE_KINDS))
    n = reference.get('n', 0)
    n = int(n) if isinstance(n, (int, np.integer)) and not isinstance(n, bool) else 0
    label = reference.get('label') or ''
    return {'kind': kind,
            'label': _canonical_label(kind, label, n, reference.get('age_band')),
            'n': n}


def grid(metrics, reference=None):
    """The 5 x 5 grid: value and z for every row/band/channel.

    ``reference`` may be None (raw values: every z is None and nothing is
    coloured), a Reference from ``reference_from_session`` /
    ``reference_from_baseline`` / ``load_reference``, or a mapping with
    ``kind='none'``. A value that cannot be computed is None - never 0; a
    cell without a reference mean/SD has z None. No head is produced for
    fewer than 3 usable channels, and two points are never interpolated.

    The contract's Grid sketch lists ``"rows"`` twice (the cell table and a
    row-label list). The cell table wins here because it is the shape the UI
    reads; the label order is returned as ``"rows_order"`` and the
    authoritative constant is ``qeeg.ROWS``.
    """
    bands = list(metrics.get('bands') or {})
    channels = list(metrics.get('usable_channels') or [])
    values = _row_values(metrics)
    block = _reference_block(reference)

    mean = reference.get('mean') if isinstance(reference, Mapping) else None
    sd = reference.get('sd') if isinstance(reference, Mapping) else None
    scoring = block['kind'] != 'none' and isinstance(mean, Mapping) \
        and isinstance(sd, Mapping)

    rows_out = {row: {band: {} for band in bands} for row in ROWS}
    missing_z = 0
    total_cells = 0
    for row in ROWS:
        for band in bands:
            for name in channels:
                value = values[row][band].get(name)
                z = None
                if scoring and value is not None:
                    mean_row = mean.get(row) if isinstance(mean.get(row), Mapping) else {}
                    sd_row = sd.get(row) if isinstance(sd.get(row), Mapping) else {}
                    mean_band = mean_row.get(band) if isinstance(mean_row.get(band), Mapping) else {}
                    sd_band = sd_row.get(band) if isinstance(sd_row.get(band), Mapping) else {}
                    z = zscore(value, mean_band.get(name), sd_band.get(name))
                if value is not None:
                    total_cells += 1
                    if scoring and z is None:
                        missing_z += 1
                rows_out[row][band][name] = {'value': value, 'z': z}

    notes = list(metrics.get('notes') or [])
    if not channels:
        notes.append(f'{MIN_CHANNELS} usable channels are required for a '
                     f'topographic grid; none are available - no heads are drawn.')
    if block['kind'] == 'none':
        notes.append('Raw values only: no reference selected, so no z colour is '
                     'shown. This app has no normative clinical database.')
    else:
        notes.append(f"Reference: {block['label']} (kind={block['kind']}, "
                     f"n={block['n']}). Not a normative clinical database; "
                     f"comparisons are against this selected reference only.")
        if block['kind'] == 'session':
            notes.append("n is the number of channels averaged in this session, "
                         "not a number of subjects.")
        elif block['kind'] == 'baseline':
            notes.append("n is the number of baseline observations supplied by "
                         "the caller, not a normative sample.")
        elif block['kind'] == 'norm':
            notes.append("n is the number of subjects in the local norm file.")
        if not scoring:
            notes.append('The selected reference carries no mean/SD table, so '
                         'no z-scores can be computed.')
        elif missing_z:
            notes.append(f'{missing_z} of {total_cells} value(s) have no z-score '
                         f'because the reference has no mean/SD for that channel.')

    return {
        'rows': rows_out,
        'reference': block,
        'limits': {'z': [GRID_LIMITS[0], GRID_LIMITS[1]]},
        'notes': notes,
        'channels': channels,
        'bands': bands,
        'rows_order': list(ROWS),
    }


def _hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*(int(round(channel)) for channel in rgb))


def _mix(start, end, fraction):
    fraction = min(max(fraction, 0.0), 1.0)
    return _hex(tuple(a + (b - a) * fraction for a, b in zip(start, end)))


def color_for_z(z, limits=GRID_LIMITS):
    """Blue -> green -> yellow -> red for a z-score, clamped to the limits.

    Green is returned exactly at z == 0. Negative z ramps blue -> green;
    positive z ramps green -> yellow (half way) -> red, so the warm end is
    visibly above the reference rather than a darker green. Values outside
    the limits take the end colour (no extrapolation), and a
    missing/non-finite z returns the neutral no-data grey.
    """
    value = _float_or_none(z)
    if value is None:
        return _NO_DATA_COLOR
    try:
        low, high = (float(limits[0]), float(limits[1]))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ValueError('limits must be a (low, high) pair') from exc
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
        raise ValueError('limits must be a (low, high) pair with low < high')
    if value <= low:
        return _hex(_BLUE)
    if value >= high:
        return _hex(_RED)
    if value == 0.0 or (value < 0 and low >= 0):
        return _hex(_GREEN)
    if value < 0:
        return _mix(_GREEN, _BLUE, value / low)
    fraction = value / high
    if fraction <= 0.5:
        return _mix(_GREEN, _YELLOW, fraction * 2.0)
    return _mix(_YELLOW, _RED, (fraction - 0.5) * 2.0)


# --------------------------------------------------------------------------
# ratios
# --------------------------------------------------------------------------
def ratios(metrics):
    """Theta/Beta, Alpha/Theta and slow/fast ratios (per channel, then mean).

    Theta/Beta = Theta absolute power / Beta absolute power, computed per
    channel and averaged across usable channels (not the ratio of the means).
    Alpha/Theta is the same shape. Slow/fast = (Delta+Theta)/(Alpha+Beta+
    HighBeta). A channel whose denominator is 0 is skipped; a missing band
    yields None rather than an invented number.
    """
    channels = list(metrics.get('usable_channels') or [])
    bands = metrics.get('bands') or {}
    if not channels or not all(name in bands for name in
                               ('Delta', 'Theta', 'Alpha', 'Beta', 'HighBeta')):
        return {'theta_beta': None, 'alpha_theta': None, 'slow_fast': None}

    theta_beta, alpha_theta, slow_fast = [], [], []
    for name in channels:
        absolute = {band: _float_or_none(bands[band]['absolute'].get(name))
                    for band in ('Delta', 'Theta', 'Alpha', 'Beta', 'HighBeta')}
        if None in absolute.values():
            continue
        if absolute['Beta'] > 0:
            theta_beta.append(absolute['Theta'] / absolute['Beta'])
        if absolute['Theta'] > 0:
            alpha_theta.append(absolute['Alpha'] / absolute['Theta'])
        denominator = absolute['Alpha'] + absolute['Beta'] + absolute['HighBeta']
        if denominator > 0:
            slow_fast.append((absolute['Delta'] + absolute['Theta']) / denominator)

    return {
        'theta_beta': _mean(theta_beta),
        'alpha_theta': _mean(alpha_theta),
        'slow_fast': _mean(slow_fast),
    }
