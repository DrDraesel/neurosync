"""Ten-region (plus midline) roll-ups of the qEEG grid.

The reading aid from ``docs/QEEG-NORMS-SPEC.md`` §2: instead of staring at 25
single heads, average the z-scores of the sites inside each multivariate
region, per band, and flag a region only when its mean |z| reaches 2 across at
least two sites. A single coloured head is weak evidence; a region that repeats
is the pattern worth discussing.

Regions overlap by design (Fp1 belongs to left medial AND left anterior). They
are reading groups, not a partition of the scalp, and they never add a site
that the grid did not measure: a region with fewer than two available sites is
omitted, never filled from its neighbours.

The spec calls these "the ten multivariate regions" and then lists eleven
labels - five lateral/parasagittal pairs plus the midline. The list is the
contract, so all eleven are implemented; the midline is reported exactly like
the others.

Only ``z`` values are aggregated. Raw values are not comparable across rows or
bands, so a grid without a reference (every z None) produces an empty table
rather than an invented one.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping

from qeeg import BANDS, ROWS

SCHEMA = 'lab.regions/1'
FLAG_Z = 2.0
MIN_SITES = 2

# The contract's region table, in reading order: lateral, medial, anterior,
# central, posterior, midline.
REGIONS = OrderedDict([
    ('left_lateral', ('F7', 'T3', 'T5')),
    ('right_lateral', ('F8', 'T4', 'T6')),
    ('left_medial', ('Fp1', 'F3', 'C3', 'P3', 'O1')),
    ('right_medial', ('Fp2', 'F4', 'C4', 'P4', 'O2')),
    ('left_anterior', ('Fp1', 'F7', 'F3')),
    ('right_anterior', ('Fp2', 'F8', 'F4')),
    ('left_central', ('T3', 'C3')),
    ('right_central', ('T4', 'C4')),
    ('left_posterior', ('T5', 'P3', 'O1')),
    ('right_posterior', ('T6', 'P4', 'O2')),
    ('midline', ('Fz', 'Cz', 'Pz')),
])

REGION_ORDER = tuple(REGIONS)
_REGION_INDEX = {name: index for index, name in enumerate(REGION_ORDER)}

__all__ = [
    'REGIONS', 'REGION_ORDER', 'FLAG_Z', 'MIN_SITES', 'region_table',
    'region_rows', 'regions_for_site', 'region_sites', 'band_order',
]


def region_sites(region):
    """Sites of one region; ValueError when the name is not a region."""
    try:
        return REGIONS[region]
    except KeyError as exc:
        raise ValueError(f'unknown region {region!r}; expected one of: '
                         + ', '.join(REGION_ORDER)) from exc


def regions_for_site(site):
    """Every region containing ``site``, in reading order (may be empty)."""
    if not isinstance(site, str):
        return []
    return [name for name, sites in REGIONS.items() if site in sites]


def _band_order(grid):
    """Bands present in the grid, in qEEG band order then any extras."""
    rows = grid.get('rows') if isinstance(grid, Mapping) else None
    found = set()
    if isinstance(rows, Mapping):
        for bands in rows.values():
            if isinstance(bands, Mapping):
                found.update(band for band in bands if isinstance(band, str))
    declared = grid.get('bands') if isinstance(grid, Mapping) else None
    if isinstance(declared, (list, tuple)):
        ordered = [band for band in declared if band in found]
        ordered += [band for band in BANDS if band in found and band not in ordered]
        ordered += sorted(found.difference(ordered))
        return ordered
    ordered = [band for band in BANDS if band in found]
    ordered += sorted(found.difference(ordered))
    return ordered


def band_order(grid):
    """Public band order helper for a table widget."""
    return _band_order(grid)


def _z(cell):
    """Finite z from a grid cell, else None. A raw value is never a z."""
    if not isinstance(cell, Mapping):
        return None
    value = cell.get('z')
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _region_table(grid, row: str | None = 'absolute'):
    """All bands/regions of one grid row (see ``region_table``)."""
    out = OrderedDict()
    if not isinstance(grid, Mapping):
        raise ValueError('grid must be the mapping returned by qeeg.grid')
    rows = grid.get('rows')
    if not isinstance(rows, Mapping):
        raise ValueError('grid is missing its rows table')

    if row is None:
        row_names = [name for name in ROWS if name in rows]
        row_names += [name for name in rows if name not in row_names]
    elif isinstance(row, str) and row:
        if row not in rows:
            return out
        row_names = [row]
    else:
        raise ValueError('row must be a grid row name or None for all rows')

    for band in _band_order(grid):
        per_region = OrderedDict()
        for region, sites in REGIONS.items():
            values = []
            available = []
            for site in sites:
                site_values = []
                for name in row_names:
                    bands = rows.get(name)
                    if not isinstance(bands, Mapping):
                        continue
                    band_cells = bands.get(band)
                    if not isinstance(band_cells, Mapping):
                        continue
                    z = _z(band_cells.get(site))
                    if z is not None:
                        site_values.append(z)
                if site_values:
                    values.append(sum(site_values) / len(site_values))
                    available.append(site)
            if len(values) < MIN_SITES:
                continue
            mean_z = sum(values) / len(values)
            per_region[region] = {
                'mean_z': mean_z,
                'n_sites': len(values),
                'flag': abs(mean_z) >= FLAG_Z,
                'sites': tuple(available),
            }
        if per_region:
            ordered = sorted(per_region.items(),
                             key=lambda item: (-abs(item[1]['mean_z']),
                                               _REGION_INDEX[item[0]]))
            out[band] = OrderedDict(ordered)
    return out


def region_table(grid, row: str | None = 'absolute'):
    """Per band, per region: ``mean_z``, ``n_sites``, ``flag`` and ``sites``.

    ``row`` selects one grid row (``'absolute'`` by default - the row the
    sheet's power reading starts from). ``row=None`` averages every row that
    has a z at a site, which is the "does the finding repeat across rows"
    summary; the rows are still z-scores, but mixing rows in one number is a
    reading aid, not a score.

    Regions with fewer than ``MIN_SITES`` measurable sites are absent. The
    flag is ``|mean_z| >= 2`` and only ever set on at least two sites. Bands
    are in qEEG order; regions inside a band are sorted by |mean_z| descending.
    """
    return _region_table(grid, row=row)


def region_rows(grid, row: str | None = 'absolute'):
    """Flat rows for a table widget, band-major and |mean_z| descending.

    Each item has exactly ``band``, ``region``, ``mean_z``, ``n_sites`` and
    ``flag``; the same data as ``region_table`` in display order.
    """
    table = _region_table(grid, row=row)
    items = []
    for band, regions in table.items():
        for region, entry in regions.items():
            items.append({
                'band': band,
                'region': region,
                'mean_z': entry['mean_z'],
                'n_sites': entry['n_sites'],
                'flag': entry['flag'],
            })
    return items
