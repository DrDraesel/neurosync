"""Unit tests for the schematic topographic map module (pure numpy/scipy).

No GUI, no hardware, no acquisition: synthetic deterministic values only. The
tests pin the documented API contract, the schematic-layout invariants and the
numerical guarantees the UI relies on (finite inside the head, NaN outside,
nose-uporientation, all-NaN when too few usable channels).
"""
import unittest

import numpy as np

import topomap as tm

# The exact channel list the DragonEEG map is built from, as specified.
DRAGON_LITERAL = ('Fp1', 'Fp7', 'F7', 'F3', 'Fz', 'Fpz', 'Fp2', 'F8', 'F4',
                  'Fp8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
                  'T6', 'O1', 'Oz', 'O2')
MIDLINE = ('Fz', 'Fpz', 'Cz', 'Pz', 'Oz')
MIRROR_PAIRS = (('Fp1', 'Fp2'), ('F7', 'F8'), ('T3', 'T4'), ('T5', 'T6'),
                ('O1', 'O2'), ('P3', 'P4'), ('C3', 'C4'), ('F3', 'F4'),
                ('Fp7', 'Fp8'))
GRID_N = 64
RADIUS = 1.0


class TopoMapTests(unittest.TestCase):
    def grid_axes(self, grid_n=GRID_N, radius=RADIUS):
        axis = np.linspace(-radius, radius, grid_n)
        return axis, axis[::-1].copy()

    def node_value(self, field, x, y, grid_n=GRID_N, radius=RADIUS):
        """Value at the grid node nearest (x, y) with row 0 = +radius."""
        axis_x, axis_y = self.grid_axes(grid_n, radius)
        row = int(np.argmin(np.abs(axis_y - y)))
        column = int(np.argmin(np.abs(axis_x - x)))
        return float(field[row, column])

    def smooth_field(self, names):
        """Deterministic smooth test field over the schematic positions."""
        return np.array([1.0 + 0.6 * tm.ELECTRODE_POSITIONS[name][1]
                         for name in names])

    # ---- layout invariants -------------------------------------------------
    def test_positions_are_inside_the_unit_head(self):
        for name, (x, y) in tm.ELECTRODE_POSITIONS.items():
            with self.subTest(name=name):
                self.assertIsInstance(name, str)
                self.assertTrue(np.isfinite(x) and np.isfinite(y))
                squared = x * x + y * y
                self.assertLess(squared, 1.0)
                self.assertLessEqual(squared, 0.95)

    def test_left_right_pairs_are_mirrored_in_x(self):
        for left, right in MIRROR_PAIRS:
            with self.subTest(pair=(left, right)):
                x_left, y_left = tm.ELECTRODE_POSITIONS[left]
                x_right, y_right = tm.ELECTRODE_POSITIONS[right]
                self.assertLess(x_left, 0.0)
                self.assertGreater(x_right, 0.0)
                self.assertAlmostEqual(x_left, -x_right, delta=1e-9)
                self.assertAlmostEqual(y_left, y_right, delta=1e-9)

    def test_midline_sites_have_zero_x(self):
        for name in MIDLINE:
            with self.subTest(name=name):
                self.assertEqual(tm.ELECTRODE_POSITIONS[name][0], 0.0)
        self.assertGreater(tm.ELECTRODE_POSITIONS['Fpz'][1], 0.0)
        self.assertLess(tm.ELECTRODE_POSITIONS['Oz'][1], 0.0)
        self.assertGreater(tm.ELECTRODE_POSITIONS['Fz'][1], 0.0)
        self.assertLess(tm.ELECTRODE_POSITIONS['Pz'][1], 0.0)

    def test_dragon_channels_are_unique_and_fully_positioned(self):
        # The literal list above enumerates 23 names (the brief's "21 sites"
        # text is a miscount; the tuple and the layout are the fixed contract).
        self.assertEqual(tm.DRAGON_CHANNELS, DRAGON_LITERAL)
        self.assertEqual(len(tm.DRAGON_CHANNELS), 23)
        self.assertEqual(len(tm.DRAGON_CHANNELS), len(set(tm.DRAGON_CHANNELS)))
        self.assertEqual(set(tm.DRAGON_CHANNELS), set(tm.ELECTRODE_POSITIONS))

    def test_brainbit_positions_match_the_same_named_dragon_positions(self):
        self.assertEqual(tm.BRAINBIT_CHANNELS, ('O1', 'O2', 'T3', 'T4'))
        # Pinned schematic coordinates, shared by both channel sets.
        pinned = {'O1': (-0.25, -0.80), 'O2': (0.25, -0.80),
                  'T3': (-0.94, 0.0), 'T4': (0.94, 0.0)}
        for name in tm.BRAINBIT_CHANNELS:
            with self.subTest(name=name):
                self.assertIn(name, tm.DRAGON_CHANNELS)
                self.assertEqual(tm.ELECTRODE_POSITIONS[name], pinned[name])
        dragon = dict(zip(tm.DRAGON_CHANNELS,
                          tm.electrode_xy(tm.DRAGON_CHANNELS)))
        brainbit = tm.electrode_xy(tm.BRAINBIT_CHANNELS)
        for name, position in zip(tm.BRAINBIT_CHANNELS, brainbit):
            with self.subTest(name=name):
                self.assertTrue(np.array_equal(position, dragon[name]))

    def test_head_outline_is_the_unit_circle(self):
        for n in (2, 181, 360):
            with self.subTest(n=n):
                x, y = tm.head_outline(n)
                self.assertEqual(np.asarray(x).shape, (n,))
                self.assertEqual(np.asarray(y).shape, (n,))
                self.assertTrue(np.allclose(x * x + y * y, 1.0, atol=1e-12))
                self.assertAlmostEqual(float(x[0]), 1.0, places=12)
                self.assertAlmostEqual(float(y[0]), 0.0, places=12)
                self.assertAlmostEqual(float(x[-1]), 1.0, places=9)
                self.assertAlmostEqual(float(y[-1]), 0.0, places=9)
        for bad in (1, 0, 2.5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    tm.head_outline(bad)

    def test_electrode_xy_follows_the_given_order(self):
        positions = tm.electrode_xy(tm.DRAGON_CHANNELS)
        self.assertEqual(positions.shape, (23, 2))
        self.assertTrue(np.issubdtype(positions.dtype, np.floating))
        for index, name in enumerate(tm.DRAGON_CHANNELS):
            with self.subTest(name=name):
                self.assertEqual(tuple(positions[index]),
                                 tm.ELECTRODE_POSITIONS[name])
        self.assertEqual(tm.electrode_xy(()).shape, (0, 2))
        with self.assertRaises(ValueError):
            tm.electrode_xy('Cz')        # a bare string is not a channel list

    def test_electrode_xy_reports_unknown_names(self):
        with self.assertRaises(ValueError) as context:
            tm.electrode_xy(['Cz', 'Fc1', 'Xyz'])
        message = str(context.exception)
        self.assertIn('Fc1', message)
        self.assertIn('Xyz', message)
        self.assertNotIn('Cz', message)

    # ---- gridded output contract ------------------------------------------
    def test_map_shape_dtype_and_complement_of_head_mask(self):
        field = tm.interpolate_topomap([1.0, 2.0, 3.0, 4.0], tm.BRAINBIT_CHANNELS)
        self.assertEqual(field.shape, (GRID_N, GRID_N))
        self.assertTrue(np.issubdtype(field.dtype, np.floating))
        mask = tm.head_mask(GRID_N)
        self.assertEqual(mask.shape, (GRID_N, GRID_N))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(np.array_equal(np.isnan(field), ~mask))
        self.assertFalse(mask[0, 0])
        self.assertTrue(np.isnan(field[0, 0]))          # corner: x^2 + y^2 = 2
        self.assertTrue(mask[GRID_N // 2, GRID_N // 2])  # near the centre

    def test_radius_rescales_the_sampled_head_circle(self):
        names = ('O1', 'O2', 'T3')
        values = [1.0, 2.0, 3.0]
        for radius in (0.5, 1.0, 2.0):
            with self.subTest(radius=radius):
                field = tm.interpolate_topomap(values, names, radius=radius)
                mask = tm.head_mask(GRID_N, radius)
                self.assertTrue(np.array_equal(np.isnan(field), ~mask))
                self.assertTrue(np.isfinite(field[mask]).all())
                self.assertTrue(np.isnan(field[0, 0]))
                self.assertFalse(mask[0, 0])
                # The disc covers the same fraction of the sampled square for
                # every radius: radius rescales the coordinates, not the shape.
                self.assertGreater(float(mask.mean()), 0.7)
                self.assertLess(float(mask.mean()), 0.8)
        for bad in (0.0, -1.0, float('nan')):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    tm.head_mask(GRID_N, bad)

    def test_constant_field_is_reproduced_exactly(self):
        for constant in (0.0, 5.0, -2.5, 1e-9):
            for names in (tm.DRAGON_CHANNELS, tm.BRAINBIT_CHANNELS):
                with self.subTest(constant=constant, channels=len(names)):
                    field = tm.interpolate_topomap([constant] * len(names), names)
                    inside = tm.head_mask()
                    self.assertTrue(np.isfinite(field[inside]).all())
                    self.assertLessEqual(
                        float(np.abs(field[inside] - constant).max()), 1e-6)
                    self.assertTrue(np.isnan(field[0, 0]))
                    self.assertTrue(np.isnan(field[-1, -1]))

    def test_single_channel_gives_a_constant_map(self):
        field = tm.interpolate_topomap([3.5], ('Cz',))
        inside = tm.head_mask()
        self.assertTrue(np.isfinite(field[inside]).all())
        self.assertTrue(np.allclose(field[inside], 3.5, atol=1e-6))
        self.assertTrue(np.isnan(field[0, 0]))

    def test_dragon_map_is_finite_inside_including_cz(self):
        names = tm.DRAGON_CHANNELS
        positions = tm.electrode_xy(names)
        values = [1.0 + 0.5 * y + 0.25 * np.sin(3.0 * x)
                  for x, y in positions]
        for grid_n in (GRID_N, 65):
            with self.subTest(grid_n=grid_n):
                field = tm.interpolate_topomap(values, names, grid_n=grid_n)
                mask = tm.head_mask(grid_n)
                self.assertTrue(np.isfinite(field[mask]).all())
                self.assertTrue(np.array_equal(np.isnan(field), ~mask))
        cz = tm.interpolate_topomap(values, names, grid_n=65)
        middle = 65 // 2
        self.assertTrue(tm.head_mask(65)[middle, middle])
        self.assertTrue(np.isfinite(cz[middle, middle]))

    def test_brainbit_map_is_finite_everywhere_inside(self):
        # Four electrodes at the back of the head: the interpolator must
        # extrapolate smoothly and stay finite over the whole head disc.
        field = tm.interpolate_topomap([1.0, 2.5, 4.0, 8.0], tm.BRAINBIT_CHANNELS)
        inside = tm.head_mask()
        self.assertTrue(np.isfinite(field[inside]).all())
        self.assertGreater(float(np.ptp(field[inside])), 0.0)
        self.assertTrue(np.array_equal(np.isnan(field), ~inside))

    def test_electrode_samples_match_supplied_values(self):
        for names in (tm.DRAGON_CHANNELS, tm.BRAINBIT_CHANNELS):
            with self.subTest(channels=len(names)):
                values = self.smooth_field(names)
                span = float(np.ptp(values))
                self.assertGreater(span, 0.0)
                field = tm.interpolate_topomap(values, names)
                for name, value in zip(names, values):
                    with self.subTest(name=name):
                        x, y = tm.ELECTRODE_POSITIONS[name]
                        node = self.node_value(field, x, y)
                        self.assertTrue(np.isfinite(node))
                        self.assertLessEqual(abs(node - value), 0.15 * span)

    def test_row_zero_is_nose_up(self):
        values = [10.0 if name == 'Fpz' else 1.0 for name in tm.DRAGON_CHANNELS]
        field = tm.interpolate_topomap(values, tm.DRAGON_CHANNELS)
        top = field[:GRID_N // 2]
        bottom = field[GRID_N // 2:]
        self.assertTrue(np.isfinite(top).any())
        self.assertTrue(np.isfinite(bottom).any())
        self.assertGreater(float(np.nanmean(top)), float(np.nanmean(bottom)))
        row, column = np.unravel_index(int(np.nanargmax(field)), field.shape)
        self.assertLess(row, GRID_N // 2)
        self.assertAlmostEqual(column, (GRID_N - 1) / 2, delta=1)
        oz_values = [10.0 if name == 'Oz' else 1.0 for name in tm.DRAGON_CHANNELS]
        oz_field = tm.interpolate_topomap(oz_values, tm.DRAGON_CHANNELS)
        last_row, _ = np.unravel_index(int(np.nanargmax(oz_field)), oz_field.shape)
        self.assertGreater(last_row, GRID_N // 2)

    # ---- validation --------------------------------------------------------
    def test_interpolate_rejects_bad_values_and_grid(self):
        names = ('O1', 'O2', 'T3')
        for values in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [], 5.0, None,
                       [1.0, 2.0, np.nan], [1.0, np.inf, 3.0], ['a', 'b', 'c']):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    tm.interpolate_topomap(values, names)
        with self.assertRaises(ValueError):
            tm.interpolate_topomap([], ())
        for grid_n in (0, 7, -8, 8.0, True, None):
            with self.subTest(grid_n=grid_n):
                with self.assertRaises(ValueError):
                    tm.interpolate_topomap([1.0], ('Cz',), grid_n=grid_n)
        for radius in (0.0, -1.0, float('nan'), float('inf'), 'wide'):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    tm.interpolate_topomap([1.0], ('Cz',), radius=radius)

    def test_interpolate_rejects_unknown_channel_names(self):
        with self.assertRaises(ValueError) as context:
            tm.interpolate_topomap([1.0, 2.0, 3.0, 4.0], ('O1', 'O2', 'T3', 'nope'))
        self.assertIn('nope', str(context.exception))

    # ---- band extraction ---------------------------------------------------
    def test_topomap_band_accepts_scalar_and_mapping_forms(self):
        names = tm.DRAGON_CHANNELS
        inside = tm.head_mask()
        scalars = tm.topomap_band({name: 4.0 for name in names}, 'Alpha', names)
        self.assertTrue(np.isfinite(scalars[inside]).all())
        self.assertTrue(np.allclose(scalars[inside], 4.0, atol=1e-6))
        mappings = {name: {'Alpha': 4.0 if name == 'Cz' else 1.0} for name in names}
        field = tm.topomap_band(mappings, 'Alpha', names)
        self.assertTrue(np.isfinite(field[inside]).all())
        cz = self.node_value(field, *tm.ELECTRODE_POSITIONS['Cz'])
        self.assertLessEqual(abs(cz - 4.0), 0.15 * 3.0)
        other_band = tm.topomap_band(mappings, 'Beta', names)
        self.assertEqual(other_band.shape, (GRID_N, GRID_N))
        self.assertTrue(np.isnan(other_band).all())
        mixed = dict(mappings)
        mixed['Oz'] = 2.0                      # scalar entries mix with mappings
        self.assertTrue(np.isfinite(tm.topomap_band(mixed, 'Alpha', names)[inside]).all())

    def test_topomap_band_drops_missing_and_nonfinite_channels(self):
        channels = ('O1', 'O2', 'T3', 'T4')
        powers = {'O1': float('nan'), 'O2': {'Alpha': 2.0}, 'T3': 3.0,
                  'T4': {'Alpha': float('inf')}}
        field = tm.topomap_band(powers, 'Alpha', channels)   # 2 usable -> all NaN
        self.assertEqual(field.shape, (GRID_N, GRID_N))
        self.assertTrue(np.isnan(field).all())
        powers['T4'] = 4.0                                   # 3 usable -> map
        field = tm.topomap_band(powers, 'Alpha', channels)
        self.assertEqual(field.shape, (GRID_N, GRID_N))
        self.assertTrue(np.isfinite(field[tm.head_mask()]).all())
        self.assertFalse(np.isnan(field[tm.head_mask()]).any())

    def test_topomap_band_returns_all_nan_below_three_usable_channels(self):
        channels = ('O1', 'O2', 'T3', 'T4')
        cases = (
            {},                                                   # none supplied
            {'O1': 1.0},                                          # one
            {'O1': 1.0, 'O2': 2.0},                               # two
            {'O1': 1.0, 'O2': 2.0, 'T3': None},                   # two finite
            {'O1': 1.0, 'O2': 2.0, 'T3': {'Beta': 3.0}},           # wrong band key
            {'O1': True, 'O2': 2.0},                              # bool is not a value
            {'O1': 1.0, 'O2': 2.0, 'T3': 'high'},                 # not a number
        )
        for powers in cases:
            with self.subTest(powers=powers):
                field = tm.topomap_band(powers, 'Alpha', channels)
                self.assertEqual(field.shape, (GRID_N, GRID_N))
                self.assertTrue(np.issubdtype(field.dtype, np.floating))
                self.assertTrue(np.isnan(field).all())
        three = tm.topomap_band({'O1': 1.0, 'O2': 2.0, 'T3': 3.0}, 'Alpha', channels)
        self.assertTrue(np.isfinite(three[tm.head_mask()]).all())

    def test_topomap_band_validates_configuration(self):
        names = ('O1', 'O2', 'T3')
        powers = dict.fromkeys(names, 2.0)
        with self.assertRaises(ValueError):
            tm.topomap_band(powers, 'Alpha', names, grid_n=7)
        with self.assertRaises(ValueError):
            tm.topomap_band(powers, 'Alpha', names, radius=0.0)
        for band in ('', None, 3):
            with self.subTest(band=band):
                with self.assertRaises(ValueError):
                    tm.topomap_band(powers, band, names)
        with self.assertRaises(ValueError):
            tm.topomap_band(powers, 'Alpha', 'O1')
        # An unknown name is never placed on a guessed coordinate, even with
        # three otherwise usable values that would allow a map.
        unknown = dict.fromkeys(('O1', 'O2', 'Xyz'), 2.0)
        with self.assertRaises(ValueError) as context:
            tm.topomap_band(unknown, 'Alpha', ('O1', 'O2', 'Xyz'))
        self.assertIn('Xyz', str(context.exception))


if __name__ == '__main__':
    unittest.main()
