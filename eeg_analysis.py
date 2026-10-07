"""Pure numeric, descriptive BrainBit EEG analysis (input units: microvolts).

No acquisition, generated signals, diagnosis, or inferred mental state. Signal
quality gates are conservative engineering HEURISTICS, not medical validation:
raw std <0.2 uV; peak-to-peak >500 uV; adjacent step >100 uV; absolute raw
value >10000 uV; raw mains power / raw 1..100 Hz power >0.25. Inspect raw
signals too: passing these checks does not establish artifact-free EEG.

Contact must be finite, positive, <=1 MOhm, and measured <=120 seconds ago.
The 1 MOhm guidance and need to scrutinize raw signals come from BrainBit:
https://sdk.brainbit.com/device-recommendation
The freshness limit and remaining numerical thresholds are app heuristics.

Welch density uses Hann, 500-sample segments, 250 overlap, constant detrend:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
Integrals interpolate exact band edges. PSD units are uV^2/Hz, integrated
powers uV^2. Relative powers are normalized over the five 1..45 Hz bands.
Aggregate power is the arithmetic mean of channel absolute powers, NOT the
mean of relative percentages. All four channels must pass for an aggregate.

Optional Q=30 notch is applied with zero-phase filtfilt before PSD; no
bandpass display curve is ever used to estimate band power. Display traces
use independent zero-phase Butterworth SOS filters and retain uV amplitudes.
Both windowed filtering operations have edge transients; display traces are
not continuous-stream filter states and must not be used for timing claims.

analyze_stable() adds a robust multi-epoch estimate for longer buffers (e.g.
30-60 s): the window is split into fixed epochs; per-channel epochs that fail
the same engineering artifact heuristics (or carry contact flags or a known
stream discontinuity) are excluded; band powers are the mean over the clean
epochs and each band carries the standard error of that mean. More clean data
reduces random variance, and excluding artifact epochs limits the bias that a
single movement or blink would otherwise spread over one long window. These
remain descriptive engineering estimates, not clinical measurements; epoch
rejection is a heuristic and can discard real activity.
"""
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
from scipy import signal

CHANNELS = ('O1', 'O2', 'T3', 'T4')
BANDS = OrderedDict([
    ('Delta', (1., 4.)), ('Theta', (4., 8.)), ('Alpha', (8., 13.)),
    ('Beta', (13., 30.)), ('Gamma', (30., 45.)),
])
COLORS = {
    'Delta': '#B69CFF', 'Theta': '#64B5F6', 'Alpha': '#69DB9C',
    'Beta': '#FFB454', 'Gamma': '#F28BCB',
}
MIN_TRACE_SAMPLES = 64


def _channel_set(channels):
    """Validate a channel-name sequence used for a device's channel table.

    The default remains the four verified BrainBit Classic channels, so every
    existing call keeps its exact behaviour; a device with a different channel
    count (e.g. the 21+ channel DragonEEG) passes its own channel tuple and the
    arithmetic stays identical. Channel names are display/contact keys only and
    never enter the numeric path.
    """
    if channels is None:
        return CHANNELS
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


def _parameters(fs, notch_hz):
    try:
        fs = float(fs)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('fs must be finite and >=200 Hz') from exc
    if not np.isfinite(fs) or fs < 200:
        raise ValueError('fs must be finite and >=200 Hz for 1..100 Hz quality checks')
    count = fs * 5
    if not np.isfinite(count) or not count.is_integer():
        raise ValueError('fs must yield an integral sample count for exactly five seconds')
    if notch_hz is None or notch_hz == 0:
        notch_hz = None
    elif notch_hz not in (50, 60):
        raise ValueError('notch_hz must be None, 0, 50, or 60')
    return fs, int(count), notch_hz


def _samples(samples_uv, channels=CHANNELS):
    try:
        if samples_uv is None or np.iscomplexobj(samples_uv):
            return None
        samples = np.asarray(samples_uv, dtype=float)
    except (ValueError, TypeError, OverflowError):
        return None
    if samples.ndim != 2 or samples.shape[1] != len(channels):
        return None
    return samples


def _blank_result(count, fs, ready=False, channels=CHANNELS):
    return {
        'ready': bool(ready), 'valid': False, 'reason': '',
        'samples': int(count), 'duration_s': float(count / fs),
        'bands': {name: {'low_hz': low, 'high_hz': high,
                         'power_uv2': None, 'relative_pct': None}
                  for name, (low, high) in BANDS.items()},
        'channels': {name: {'valid': False, 'flags': [], 'dominant': None,
                            'peak_hz': None, 'powers_uv2': {}, 'relative_pct': {}}
                     for name in channels},
        'dominant': None, 'dominant_pct': None, 'peak_hz': None,
        'conclusion': 'No aggregate band conclusion: quality is unverified; descriptive only.',
    }


def _reject(result, reason):
    result['reason'] = reason
    for channel in result['channels'].values():
        channel['flags'].append(reason)
    return result


def _channel_value(values, index, scalar=False, name=None):
    if isinstance(values, Mapping):
        key = name if name is not None else CHANNELS[index]
        return values.get(key)
    if scalar and np.isscalar(values):
        return values
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return None


def _finite_number(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _contact_flags(contact_ohms, contact_age_s, index, name=None):
    raw = _channel_value(contact_ohms, index, name=name)
    resistance = _finite_number(raw)
    age = _finite_number(_channel_value(contact_age_s, index, scalar=True, name=name))
    flags = []
    if raw is None:
        flags.append('contact_unverified')
    elif resistance is None or resistance <= 0:
        flags.append('contact_invalid')
    elif resistance > 1_000_000:
        flags.append('contact_high')
    if age is None or age < 0:
        if 'contact_unverified' not in flags:
            flags.append('contact_unverified')
    elif age > 120:
        flags.append('contact_stale')
    return flags


def _psd(values, fs):
    return signal.welch(values, fs=fs, window='hann', nperseg=500,
                        noverlap=250, detrend='constant', scaling='density')


def _integral(frequencies, density, low, high):
    """Integrate a piecewise-linear PSD, including interpolated exact edges."""
    interior = (frequencies > low) & (frequencies < high)
    x = np.concatenate(([low], frequencies[interior], [high]))
    y = np.concatenate(([np.interp(low, frequencies, density)], density[interior],
                        [np.interp(high, frequencies, density)]))
    return float(np.trapezoid(y, x))


def _notched(values, fs, notch_hz):
    centered = values - np.mean(values)
    if notch_hz is None:
        return centered
    b, a = signal.iirnotch(notch_hz, Q=30, fs=fs)
    return signal.filtfilt(b, a, centered)


def _relative(powers):
    total = sum(powers.values())
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        return {}
    return {name: float(100 * power / total) for name, power in powers.items()}


def _dominant(relative):
    ranked = sorted(relative, key=relative.get, reverse=True)
    first, second = ranked[:2]
    mixed = relative[first] - relative[second] <= 5. + 1e-10
    return ('mixed' if mixed else first), float(relative[first])


def _peak_hz(frequencies, density):
    """Frequency of the largest density value between 1 and 45 Hz, else None.

    Same definition as the aggregate ``peak_hz``; per channel it describes that
    channel's own density, so a channel peak and the aggregate peak can differ.
    """
    inside = (frequencies >= 1) & (frequencies <= 45)
    if not inside.any():
        return None
    return float(frequencies[inside][np.argmax(density[inside])])


def analyze_window(samples_uv, fs=250, notch_hz=60, contact_ohms=None,
                   contact_age_s=None, continuous=True, channels=CHANNELS):
    """Analyze exactly the latest five seconds of real Nx4 microvolt samples.

    ``channels`` selects the device's channel table (default: the four verified
    BrainBit Classic channels); the numeric path is identical for any count, so
    a 21+ channel Dragon buffer is analysed with exactly the same statistics.
    Contacts must be keyed by those channel names.

    ``ready`` means enough samples, NOT adequate quality. ``samples`` reports
    samples retained (max 5*fs). All supplied input must be finite, including
    older rows. Missing/malformed data returns a suppressed JSON-safe result.
    Invalid configuration raises ValueError; fs >=200 and integer 5*fs are
    required to preserve the full 1..100 Hz artifact check.

    Contacts accept channel-name mappings or ordered sequences matching the
    channel table; ages accept either those formats or one shared age in
    seconds. Missing, invalid, high, or stale contact blocks aggregate output.
    ``continuous`` must be a true boolean supplied by acquisition after checking
    packet gaps. Finite per-channel powers can remain on flagged channels for
    inspection; only channels with no flags get a dominant label. Uncomputable
    per-channel powers/percentages are empty mappings (never NaN). Invalid
    aggregate band values, dominant, dominant_pct, and peak_hz are all None.
    Each channel that has usable band numbers also carries its own ``peak_hz``:
    the largest value of that channel's mean density between 1 and 45 Hz (same
    definition as the aggregate peak), None when no band power was computable.

    A top-two difference <=5 percentage points is labeled 'mixed'; its
    dominant_pct is the largest individual percentage, not their sum. The
    aggregate absolute power is the arithmetic mean over every analysed channel,
    so it describes a wider or narrower scalp region depending on the device.
    """
    fs, needed, notch_hz = _parameters(fs, notch_hz)
    names = _channel_set(channels)
    samples = _samples(samples_uv, names)
    if samples is None:
        return _reject(_blank_result(0, fs, channels=names), 'invalid_sample_shape')
    count = min(len(samples), needed)
    result = _blank_result(count, fs, count == needed, channels=names)
    if not np.all(np.isfinite(samples)):
        return _reject(result, 'nonfinite_input')
    if count < needed:
        return _reject(result, 'insufficient_samples')
    window = samples[-needed:]
    densities = []
    for index, name in enumerate(names):
        channel = result['channels'][name]
        flags = _contact_flags(contact_ohms, contact_age_s, index, name=name)
        channel['flags'] = flags
        if not isinstance(continuous, (bool, np.bool_)) or not continuous:
            flags.append('discontinuous')
        raw = window[:, index]
        # Skip arithmetic on saturated values to avoid overflow on hostile input.
        if np.max(np.abs(raw)) > 10000:
            flags.append('saturated_or_offset')
            continue
        if np.std(raw) < .2:
            flags.append('flatline')
        if np.ptp(raw) > 500:
            flags.append('peak_to_peak')
        if np.max(np.abs(np.diff(raw))) > 100:
            flags.append('abrupt_step')
        frequencies, raw_density = _psd(raw, fs)
        raw_power = _integral(frequencies, raw_density, 1., 100.)
        # Quality is checked independently of the chosen display notch.
        inspect_mains = (50., 60.)
        if raw_power > np.finfo(float).tiny and any(
            _integral(frequencies, raw_density, hz - 2, hz + 2) / raw_power > .25
            for hz in inspect_mains
        ):
            flags.append('mains_contamination')
        frequencies, density = _psd(_notched(raw, fs, notch_hz), fs)
        powers = {band: _integral(frequencies, density, low, high)
                  for band, (low, high) in BANDS.items()}
        relative = _relative(powers)
        channel['powers_uv2'] = powers
        channel['relative_pct'] = relative
        if not relative:
            flags.append('no_band_power')
        channel['valid'] = not flags
        if channel['valid']:
            channel['dominant'] = _dominant(relative)[0]
        # Per-channel peak frequency uses the channel's own mean density and the
        # same 1..45 Hz definition as the aggregate peak; None when the channel
        # has no usable band numbers at all.
        channel['peak_hz'] = _peak_hz(frequencies, density) if relative else None
        densities.append(density)
    if not all(channel['valid'] for channel in result['channels'].values()):
        result['reason'] = '; '.join(
            f"{name}: {', '.join(channel['flags'])}"
            for name, channel in result['channels'].items() if channel['flags'])
        return result
    powers = {band: float(np.mean([channel['powers_uv2'][band]
                                  for channel in result['channels'].values()]))
              for band in BANDS}
    relative = _relative(powers)
    dominant, pct = _dominant(relative)
    for band in BANDS:
        result['bands'][band]['power_uv2'] = powers[band]
        result['bands'][band]['relative_pct'] = relative[band]
    mean_density = np.mean(densities, axis=0)
    peak = _peak_hz(frequencies, mean_density)
    if dominant == 'mixed':
        conclusion = ('Latest 5s: mixed measured bands (top two within 5 percentage '
                      'points); descriptive only.')
    else:
        conclusion = (f'Largest measured band in latest 5s: {dominant} '
                      f'({pct:.1f}% of 1-45 Hz power); descriptive only.')
    result.update(valid=True, reason='ok', dominant=dominant,
                  dominant_pct=pct, peak_hz=peak, conclusion=conclusion)
    return result


def _epoch_parameters(fs, epoch_seconds, min_epochs):
    try:
        epoch_seconds = float(epoch_seconds)
        min_epochs = int(min_epochs)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('epoch_seconds and min_epochs must be numbers') from exc
    per = fs * epoch_seconds
    if not np.isfinite(per) or not float(per).is_integer() or per < 250:
        raise ValueError('epoch_seconds must give an integral sample count >= 250 at this rate')
    if not 2 <= min_epochs <= 20000:
        raise ValueError('min_epochs must be an integer between 2 and 20000')
    return epoch_seconds, int(per), min_epochs


def _epoch_flags(raw, edge=None):
    """Same engineering artifact heuristics as the five-second analysis.

    ``edge`` may carry one preceding sample so a step exactly at an epoch
    boundary still registers as an abrupt change in the entering epoch.
    """
    if np.max(np.abs(raw)) > 10000:
        return ['saturated_or_offset']
    flags = []
    if np.std(raw) < .2:
        flags.append('flatline')
    if np.ptp(raw) > 500:
        flags.append('peak_to_peak')
    if np.max(np.abs(np.diff(raw if edge is None else edge))) > 100:
        flags.append('abrupt_step')
    return flags


def _epoch_density(raw, fs, per):
    nperseg = min(500, per)
    return signal.welch(raw, fs=fs, window='hann', nperseg=nperseg,
                        noverlap=nperseg // 2, detrend='constant',
                        scaling='density')


def _blank_stable(epochs_total, per, fs, epoch_seconds, channels=CHANNELS):
    return {
        'valid': False, 'reason': '',
        'epochs_total': int(epochs_total), 'epochs_used': 0,
        'epoch_seconds': float(epoch_seconds),
        'seconds_total': float(epochs_total * per / fs), 'seconds_used': 0.,
        'bands': {name: {'low_hz': low, 'high_hz': high, 'power_uv2': None,
                         'sem_uv2': None, 'relative_pct': None, 'used_epochs': 0}
                  for name, (low, high) in BANDS.items()},
        'channels': {name: {'valid': False, 'flags': [], 'epochs_used': 0,
                            'excluded': {}, 'powers_uv2': {}, 'sem_uv2': {},
                            'relative_pct': {}, 'dominant': None, 'peak_hz': None}
                     for name in channels},
        'dominant': None, 'dominant_pct': None, 'peak_hz': None,
        'conclusion': 'No stable estimate available; descriptive only.',
    }


def analyze_stable(samples_uv, fs=250, notch_hz=60, contact_ohms=None,
                   contact_age_s=None, continuous=True, epoch_seconds=2.,
                   min_epochs=6, channels=CHANNELS):
    """Robust multi-epoch estimate over a longer real-microvolt buffer.

    Splits the newest whole epochs of ``epoch_seconds`` and, per channel,
    excludes epochs failing the artifact heuristics, epochs of a flagged
    channel, and everything when ``continuous`` is False. Each epoch's abrupt-
    change check includes one preceding sample, so a step exactly at an epoch
    boundary rejects the entering epoch instead of escaping detection. Powers
    are means over clean epochs; ``sem_uv2`` is the standard error of those
    clean epochs (0 when only one epoch is clean). Aggregate = mean of channel
    means and requires every analysed channel valid, matching the five-second
    analysis; aggregate SEM combines channel SEMs as sqrt(sum(sem^2))/4 and
    ``epochs_used`` is the minimum across channels. ``excluded`` counts the
    reject reasons per channel. Input is never mutated; all outputs are
    JSON-safe; malformed input returns a suppressed result; invalid
    configuration raises ValueError. Descriptive research estimate: not a
    clinical result, and epoch rejection is a heuristic.

    ``channels`` selects the device's channel table (default: the four verified
    BrainBit Classic channels). With a 21+ channel device the aggregate covers
    the whole reported electrode set and therefore needs every one of those
    channels to be clean; treat a missing aggregate there as the conservative
    gate it is, not as device failure. Each channel with usable band numbers
    also carries its own ``peak_hz`` (largest value of that channel's mean
    density across its clean epochs, 1..45 Hz), useful for the per-subject
    baseline table in ``baseline.py``; it is None when nothing was computable.
    """
    fs, _, notch_hz = _parameters(fs, notch_hz)
    epoch_seconds, per, min_epochs = _epoch_parameters(fs, epoch_seconds, min_epochs)
    names = _channel_set(channels)
    samples = _samples(samples_uv, names)
    if samples is None:
        return _reject(_blank_stable(0, per, fs, epoch_seconds, names), 'invalid_sample_shape')
    if not np.all(np.isfinite(samples)):
        return _reject(_blank_stable(len(samples) // per, per, fs, epoch_seconds, names),
                       'nonfinite_input')
    epochs_total = len(samples) // per
    result = _blank_stable(epochs_total, per, fs, epoch_seconds, names)
    if epochs_total < min_epochs:
        return _reject(result, 'insufficient_samples')
    window = samples[-epochs_total * per:].reshape(-1, len(names))
    continuous_ok = isinstance(continuous, (bool, np.bool_)) and bool(continuous)
    densities = []
    for index, name in enumerate(names):
        channel = result['channels'][name]
        flags = _contact_flags(contact_ohms, contact_age_s, index, name=name)
        if not continuous_ok:
            flags.append('discontinuous')
        powers_by_epoch = []
        excluded = {}
        channel_densities = []
        for epoch in range(epochs_total):
            start = epoch * per
            raw = window[start:start + per, index]
            edge = window[start - 1:start + per, index] if start else raw
            reasons = _epoch_flags(raw, edge)
            if not reasons:
                frequencies, raw_density = _epoch_density(raw, fs, per)
                raw_power = _integral(frequencies, raw_density, 1., 100.)
                if raw_power > np.finfo(float).tiny and any(
                    _integral(frequencies, raw_density, hz - 2, hz + 2) / raw_power > .25
                    for hz in (50., 60.)
                ):
                    reasons = ['mains_contamination']
            if reasons:
                for reason in reasons:
                    excluded[reason] = excluded.get(reason, 0) + 1
                continue
            frequencies, density = _epoch_density(_notched(raw, fs, notch_hz), fs, per)
            densities.append(density)
            channel_densities.append(density)
            powers_by_epoch.append({band: _integral(frequencies, density, low, high)
                                    for band, (low, high) in BANDS.items()})
        used = len(powers_by_epoch)
        channel['epochs_used'] = used
        channel['excluded'] = excluded
        if used:
            for band in BANDS:
                values = np.array([powers[band] for powers in powers_by_epoch])
                channel['powers_uv2'][band] = float(values.mean())
                channel['sem_uv2'][band] = float(values.std(ddof=1) / np.sqrt(used)) if used > 1 else 0.
            channel['relative_pct'] = _relative(channel['powers_uv2'])
        if used < min_epochs:
            flags.append('insufficient_clean_epochs')
        if not channel['relative_pct']:
            flags.append('no_band_power')
        channel['flags'] = flags
        channel['valid'] = not flags
        if channel['valid']:
            channel['dominant'] = _dominant(channel['relative_pct'])[0]
        # Per-channel peak frequency over that channel's own clean epochs, using
        # the same 1..45 Hz definition as the aggregate peak.
        if channel_densities and channel['relative_pct']:
            channel['peak_hz'] = _peak_hz(frequencies,
                                          np.mean(np.vstack(channel_densities), axis=0))
    channels = result['channels'].values()
    if not all(channel['valid'] for channel in channels):
        result['reason'] = '; '.join(
            f"{name}: {', '.join(channel['flags'])}"
            for name, channel in result['channels'].items() if channel['flags'])
        return result
    channel_list = list(result['channels'].values())
    used = min(channel['epochs_used'] for channel in channel_list)
    powers, sem = {}, {}
    for band in BANDS:
        means = [channel['powers_uv2'][band] for channel in channel_list]
        sems = [channel['sem_uv2'][band] for channel in channel_list]
        powers[band] = float(np.mean(means))
        sem[band] = float(np.sqrt(np.sum(np.square(sems))) / len(means))
    relative = _relative(powers)
    dominant, pct = _dominant(relative)
    for band in BANDS:
        result['bands'][band]['power_uv2'] = powers[band]
        result['bands'][band]['sem_uv2'] = sem[band]
        result['bands'][band]['relative_pct'] = relative[band]
        result['bands'][band]['used_epochs'] = used
    mean_density = np.mean(np.vstack(densities), axis=0)
    peak = _peak_hz(frequencies, mean_density)
    epoch_label = f'{epoch_seconds:g} s epochs'
    if dominant == 'mixed':
        conclusion = (f'Stable estimate over {used} clean {epoch_label} '
                      f'({used * epoch_seconds:g} s): mixed measured bands (top two within 5 '
                      'percentage points); descriptive only.')
    else:
        conclusion = (f'Stable estimate over {used} clean {epoch_label} '
                      f'({used * epoch_seconds:g} s): largest band {dominant} '
                      f'({pct:.1f}% of 1-45 Hz power); descriptive only.')
    result.update(valid=True, reason='ok', dominant=dominant, dominant_pct=pct,
                  peak_hz=peak, epochs_used=used,
                  seconds_used=used * epoch_seconds, conclusion=conclusion)
    return result


def band_traces(samples_uv, fs=250, channel=0, notch_hz=60, channels=CHANNELS):
    """Return independent display-only band traces in uV, preserving N samples.

    Mean removal, optional zero-phase Q30 notch, then independent order-four
    Butterworth SOS zero-phase bandpasses. Amplitudes are NOT normalized.
    Window edges are affected by padding/transients (especially Delta); this
    function neither certifies signal quality nor provides causal latency.
    Fewer than 64 rows, malformed shape, or any nonfinite input gives five
    empty lists. No input mutation. Only channel indices of the supplied
    channel table (default: the four BrainBit Classic channels) are accepted.
    """
    fs, _, notch_hz = _parameters(fs, notch_hz)
    names = _channel_set(channels)
    if not isinstance(channel, (int, np.integer)) or not 0 <= channel < len(names):
        raise ValueError(f'channel must be an integer index 0..{len(names) - 1}')
    empty = {name: [] for name in BANDS}
    samples = _samples(samples_uv, names)
    if samples is None or len(samples) < MIN_TRACE_SAMPLES or not np.all(np.isfinite(samples)):
        return empty
    values = _notched(samples[:, channel], fs, notch_hz)
    return {name: signal.sosfiltfilt(
        signal.butter(4, (low, high), btype='bandpass', fs=fs, output='sos'),
        values).tolist() for name, (low, high) in BANDS.items()}


def mean_psd_curve(samples_uv, fs=250, notch_hz=60, channels=CHANNELS, fmax=100.):
    """Display-only mean Welch density across the analysed channels.

    Returns ``(frequencies, density)`` as plain lists with exactly the same
    Welch settings used for band power (Hann, 500-sample segments, 250 sample
    overlap, constant detrend, density scaling), averaged over the channels
    that contain finite data. This curve is for the spectrum chart only: band
    powers always come from the per-channel integrated path in analyze_window/
    analyze_stable, never from a plotted curve. Frequencies above ``fmax`` are
    dropped, and it returns ``([], [])`` for unusable input instead of NaN.
    Fewer than 500 samples cannot be decomposed into one segment and also
    returns empty lists.
    """
    fs, needed, notch_hz = _parameters(fs, notch_hz)
    names = _channel_set(channels)
    samples = _samples(samples_uv, names)
    if samples is None or len(samples) < 500 or not np.all(np.isfinite(samples)):
        return [], []
    try:
        fmax = float(fmax)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('fmax must be a finite positive number') from exc
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError('fmax must be a finite positive number')
    densities = []
    frequencies = None
    for index in range(len(names)):
        raw = samples[:, index]
        if np.max(np.abs(raw)) > 10000:
            continue
        frequencies, density = _psd(_notched(raw, fs, notch_hz), fs)
        densities.append(density)
    if not densities:
        return [], []
    keep = frequencies <= fmax
    return (frequencies[keep].tolist(),
            np.mean(np.vstack(densities), axis=0)[keep].tolist())
