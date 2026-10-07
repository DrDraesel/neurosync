"""Schematic 2D topographic head maps for NeuroSync (engineering display only).

Pure numeric module: numpy + scipy only, no Qt/UI imports, deterministic (no
randomness, no acquisition, no hardware access). Band values in, a schematic
band-power raster out for the app to show with pyqtgraph ImageItem.

HONESTY LIMITS. ``ELECTRODE_POSITIONS`` holds APPROXIMATE SCHEMATIC 2D layout
positions on a unit head (nose at +y, ears at x = +-1); they are NOT measured
sensor locations, NOT digitised electrode coordinates and NOT a head mesh.
``interpolate_topomap`` performs smooth ENGINEERING INTERPOLATION between
whatever electrodes are supplied: the result is a sensor-space picture for
eyeballing band-power distribution. It is NOT a clinical brain map, NOT a
source localisation, and it says nothing about the depth, origin or anatomical
location of the activity, nor about the cortex beneath the scalp. A four
electrode map (BrainBit Classic) is mostly extrapolation and must be read as
such; the band powers themselves keep the limits documented in eeg_analysis
(descriptive engineering estimates only).

Orientation contract shared by every gridded output: a ``(grid_n, grid_n)``
float array sampled on the square [-radius, radius] x [-radius, radius] with
row 0 at y = +radius (so the map renders nose-up when the UI draws rows
top-down) and column 0 at x = -radius. Samples strictly outside the head
circle (x*x + y*y > radius*radius) are NaN so the UI can mask or blank them.
"""
from collections.abc import Mapping, Sequence

import numpy as np
from scipy import interpolate

# Approximate schematic 10-20-style layout on a unit head: x to the right,
# y up, nose at +y = (0, 1), ears at (+-1, 0). Left-hemisphere sites have
# x < 0, right-hemisphere sites x > 0, midline sites x == 0 exactly. These are
# schematic drawing positions, NOT measured sensor locations.
ELECTRODE_POSITIONS = {
    'Fp1': (-0.28, 0.82), 'Fp2': (0.28, 0.82), 'Fpz': (0.0, 0.86),
    'Fp7': (-0.72, 0.60), 'Fp8': (0.72, 0.60),
    'F7': (-0.88, 0.30), 'F3': (-0.50, 0.55), 'Fz': (0.0, 0.55),
    'F4': (0.50, 0.55), 'F8': (0.88, 0.30),
    'T3': (-0.94, 0.0), 'C3': (-0.52, 0.0), 'Cz': (0.0, 0.0),
    'C4': (0.52, 0.0), 'T4': (0.94, 0.0),
    'T5': (-0.75, -0.52), 'P3': (-0.48, -0.55), 'Pz': (0.0, -0.55),
    'P4': (0.48, -0.55), 'T6': (0.75, -0.52),
    'O1': (-0.25, -0.80), 'Oz': (0.0, -0.86), 'O2': (0.25, -0.80),
}

# The channel tuple the DragonEEG head map is built from, in the order the
# device/documentation lists them. Fp1/Fp2, Fp7/Fp8, F7/F8, F3/F4, Fz, Fpz,
# T3/T4, C3/C4, Cz, T5/T6, P3/P4, Pz, O1/O2 and Oz all have schematic
# positions above.
DRAGON_CHANNELS = ('Fp1', 'Fp7', 'F7', 'F3', 'Fz', 'Fpz', 'Fp2', 'F8', 'F4',
                   'Fp8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
                   'T6', 'O1', 'Oz', 'O2')

# BrainBit Classic sites; identical coordinates to the same-named Dragon sites
# above (one shared position table, never a second guess at the layout).
BRAINBIT_CHANNELS = ('O1', 'O2', 'T3', 'T4')

# Multiquadric scale: roughly one electrode-spacing in the schematic layout,
# so the basis stays well conditioned for both 4 and 23 electrodes.
RBF_EPSILON = 0.5
MIN_GRID_N = 8
MIN_BAND_CHANNELS = 3


def _grid_parameters(grid_n, radius):
    """Validate grid_n/radius; returns (int grid_n, float radius)."""
    if isinstance(grid_n, (bool, np.bool_)) or not isinstance(grid_n, (int, np.integer)):
        raise ValueError(f'grid_n must be an integer >= {MIN_GRID_N}')
    if grid_n < MIN_GRID_N:
        raise ValueError(f'grid_n must be an integer >= {MIN_GRID_N}')
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('radius must be a finite positive number') from exc
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be a finite positive number')
    return int(grid_n), radius


def _grid(grid_n, radius):
    """Meshgrid of the sampled square; row 0 is y = +radius, column 0 x = -radius."""
    axis = np.linspace(-radius, radius, grid_n)
    return np.meshgrid(axis, axis[::-1].copy())


def _names(channels):
    """Channel names as a list; a bare string is rejected, not iterated."""
    if isinstance(channels, (str, bytes)):
        raise ValueError('channels must be a sequence of channel names, not one string')
    try:
        return list(channels)
    except TypeError as exc:
        raise ValueError('channels must be a sequence of channel names') from exc


def _value_array(values, count):
    """1-D finite float array of exactly ``count`` values, else ValueError."""
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError('values must be a sequence of finite numbers')
    try:
        data = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('values must be a sequence of finite numbers') from exc
    if data.ndim != 1 or data.shape[0] != count:
        raise ValueError(f'values must hold exactly {count} finite numbers, '
                         'one per channel')
    if not np.all(np.isfinite(data)):
        raise ValueError('values must all be finite; NaN or inf cannot be mapped')
    return data


def _finite_number(value):
    """Finite float for a usable scalar value, else None (bools are not values)."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def head_outline(n=181):
    """Unit circle of ``n`` points (cos/sin of linspace(0, 2*pi, n)).

    Only the head circle is returned; the nose and ears are drawn by the UI
    from this same schematic layout. First and last points coincide, so the
    polyline closes. The circle is a drawing convention for the schematic head
    outline, not a measured head contour.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError('n must be an integer >= 2')
    angle = np.linspace(0.0, 2.0 * np.pi, int(n))
    return np.cos(angle), np.sin(angle)


def electrode_xy(channels: Sequence[str]) -> np.ndarray:
    """(len(channels), 2) schematic positions in the given order.

    Raises ValueError listing every name with no known schematic position;
    unknown names are never placed on a guessed coordinate.
    """
    names = _names(channels)
    unknown, positions = [], []
    for name in names:
        try:
            position = ELECTRODE_POSITIONS.get(name)
        except TypeError:            # unhashable name
            position = None
        if position is None:
            unknown.append(str(name))
        else:
            positions.append(position)
    if unknown:
        unique = sorted(set(unknown))
        raise ValueError('unknown channel name(s) with no known schematic '
                         f'position: {", ".join(unique)}')
    return np.asarray(positions, dtype=float).reshape(len(names), 2)


def head_mask(grid_n=64, radius=1.0):
    """Boolean (grid_n, grid_n) mask, True inside the head circle.

    True where x*x + y*y <= radius*radius, in the same row-0-is-+y orientation
    as interpolate_topomap, so np.isnan(interpolate_topomap(...)) equals the
    complement of this mask.
    """
    grid_n, radius = _grid_parameters(grid_n, radius)
    grid_x, grid_y = _grid(grid_n, radius)
    return grid_x * grid_x + grid_y * grid_y <= radius * radius


def _rbf_field(points, values, grid_x, grid_y):
    """Multiquadric radial-basis field over the whole square (may raise)."""
    rbf = interpolate.Rbf(points[:, 0], points[:, 1], values,
                          kind='multiquadric', epsilon=RBF_EPSILON)
    flat = np.asarray(rbf(grid_x.ravel(), grid_y.ravel()), dtype=float)
    return flat.reshape(grid_x.shape)


def _nearest_field(points, values, grid_x, grid_y):
    """Nearest-neighbour field, fills the whole square (may raise)."""
    return np.asarray(interpolate.griddata(points, values, (grid_x, grid_y),
                                           method='nearest'), dtype=float)


def interpolate_topomap(values: Sequence[float], channels: Sequence[str],
                        grid_n: int = 64, radius: float = 1.0) -> np.ndarray:
    """Interpolate one band's per-channel values into a schematic head map.

    Returns a (grid_n, grid_n) float array on [-radius, radius] x [-radius,
    radius]; row 0 is y = +radius (nose up when drawn top-down) and column 0 is
    x = -radius. Samples with x*x + y*y > radius*radius are NaN, matching the
    complement of head_mask, and the map is always finite inside the head.

    A multiquadric radial-basis interpolant is used because it extrapolates
    smoothly outside the convex hull of the electrodes (a four-electrode
    BrainBit set covers only the back of the head, so nearest-neighbour-only
    interpolation would draw flat plateaus instead of a map). Constants are
    reproduced exactly by interpolating the deviation from the mean and adding
    the mean back: a raw multiquadric would otherwise overshoot a constant
    input by a few percent. If the radial-basis solve raises (degenerate or too
    few points) the fallback chain is scipy griddata closest-point, then a
    constant field equal to the mean of the supplied values.

    ValueError on non-finite values, on len(values) != len(channels), on an
    empty channel list, on unknown channel names, on grid_n < 8 and on
    radius <= 0. These are schematic engineering head maps, not clinical brain
    maps and not source localisation.
    """
    grid_n, radius = _grid_parameters(grid_n, radius)
    points = electrode_xy(channels)
    if len(points) < 1:
        raise ValueError('at least one channel with a known position is required')
    data = _value_array(values, len(points))
    grid_x, grid_y = _grid(grid_n, radius)
    inside = head_mask(grid_n, radius)
    mean = float(np.mean(data))
    try:
        candidate = _rbf_field(points, data - mean, grid_x, grid_y) + mean
    except Exception:                # degenerate/too few points: griddata next
        candidate = None
    if candidate is None or not np.isfinite(candidate[inside]).all():
        try:
            candidate = _nearest_field(points, data, grid_x, grid_y)
        except Exception:            # nothing scipy-safe left: honest constant
            candidate = None
        if candidate is None or not np.isfinite(candidate[inside]).all():
            candidate = np.full((grid_n, grid_n), mean, dtype=float)
    return np.where(inside, candidate, np.nan)


def topomap_band(channel_powers: Mapping[str, Mapping[str, float] | float],
                 band: str, channels: Sequence[str], grid_n: int = 64,
                 radius: float = 1.0) -> np.ndarray:
    """Map one band from per-channel band powers onto the schematic head.

    ``channel_powers`` maps a channel name either to a mapping of band name to
    value, in which case the value is looked up with EXACTLY the ``band`` key
    given (for example 'Alpha'; no case folding and no aliases), or to a plain
    scalar value used as-is. Channels whose value is missing or not finite are
    dropped from the interpolation. If fewer than 3 channels remain, an
    all-NaN (grid_n, grid_n) array is returned so the UI can say 'not enough
    good channels' instead of inventing a map from one or two sensors.
    Otherwise the result is interpolate_topomap's schematic map, with the same
    NaN-outside-the-head and row-0-is-+y conventions. Unknown channel names and
    invalid grid_n/radius raise ValueError, exactly as in interpolate_topomap.
    """
    grid_n, radius = _grid_parameters(grid_n, radius)
    if not isinstance(band, str) or not band:
        raise ValueError('band must be a non-empty band name string')
    names = _names(channels)
    usable_names, usable_values = [], []
    for name in names:
        try:
            entry = channel_powers.get(name) if isinstance(channel_powers, Mapping) else None
        except TypeError:
            entry = None
        value = entry.get(band) if isinstance(entry, Mapping) else entry
        number = _finite_number(value)
        if number is not None:
            usable_names.append(name)
            usable_values.append(number)
    if len(usable_names) < MIN_BAND_CHANNELS:
        return np.full((grid_n, grid_n), np.nan)
    return interpolate_topomap(usable_values, usable_names, grid_n=grid_n,
                               radius=radius)
