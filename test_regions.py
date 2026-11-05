"""Tests for regions.py: ten-region roll-ups over the qEEG grid.

TEST ONLY. Grids are hand-built synthetic z maps; no recording and no
normative database are involved. These tests pin the region contract in
``docs/QEEG-NORMS-SPEC.md`` section 2.
"""
import unittest

import qeeg
import regions


def make_grid(by_band, row='absolute'):
    """A grid-shaped mapping with z values for one row of one or more bands."""
    bands = list(by_band)
    rows = {name: {band: {} for band in bands} for name in qeeg.ROWS}
    for band, zmap in by_band.items():
        for channel, z in zmap.items():
            rows[row][band][channel] = {'value': None if z is None else z, 'z': z}
    channels = sorted({name for zmap in by_band.values() for name in zmap})
    return {'rows': rows, 'bands': bands, 'channels': channels,
            'reference': {'kind': 'none', 'label': 'raw', 'n': 0},
            'limits': {'z': [-3, 3]}, 'notes': []}


class RegionDefinitionTests(unittest.TestCase):
    def test_all_eleven_listed_regions_exist(self):
        expected = {'left_lateral', 'right_lateral', 'left_medial',
                    'right_medial', 'left_anterior', 'right_anterior',
                    'left_central', 'right_central', 'left_posterior',
                    'right_posterior', 'midline'}
        self.assertEqual(set(regions.REGIONS), expected)
        self.assertEqual(regions.REGIONS['left_lateral'], ('F7', 'T3', 'T5'))
        self.assertEqual(regions.REGIONS['left_medial'],
                         ('Fp1', 'F3', 'C3', 'P3', 'O1'))
        self.assertEqual(regions.REGIONS['midline'], ('Fz', 'Cz', 'Pz'))

    def test_every_region_site_is_a_10_20_name(self):
        from topomap import ELECTRODE_POSITIONS
        for region, sites in regions.REGIONS.items():
            with self.subTest(region=region):
                for site in sites:
                    self.assertIn(site, ELECTRODE_POSITIONS)
        # Explicitly: the sites are the ones qeeg's pairs know about.
        known = {site for pair in qeeg.HOMOLOGOUS_PAIRS for site in pair}
        known |= {'Fz', 'Cz', 'Pz'}
        for sites in regions.REGIONS.values():
            for site in sites:
                self.assertIn(site, known)

    def test_regions_for_site(self):
        self.assertEqual(regions.regions_for_site('Fp1'),
                         ['left_medial', 'left_anterior'])
        self.assertEqual(regions.regions_for_site('Oz'), [])
        with self.assertRaises(ValueError):
            regions.region_sites('not_a_region')


class RegionTableTests(unittest.TestCase):
    def test_left_medial_flags_with_three_sites_at_2_5(self):
        grid = make_grid({'Alpha': {'F3': 2.5, 'C3': 2.5, 'P3': 2.5}})
        table = regions.region_table(grid)
        self.assertIn('Alpha', table)
        entry = table['Alpha']['left_medial']
        self.assertAlmostEqual(entry['mean_z'], 2.5)
        self.assertEqual(entry['n_sites'], 3)
        self.assertTrue(entry['flag'])
        self.assertEqual(entry['sites'], ('F3', 'C3', 'P3'))

    def test_single_site_region_is_omitted(self):
        grid = make_grid({'Alpha': {'P3': 3.0}})
        table = regions.region_table(grid)
        self.assertNotIn('left_medial', table.get('Alpha', {}))
        self.assertNotIn('left_posterior', table.get('Alpha', {}))
        self.assertEqual(table, {})

    def test_flag_needs_two_sites_and_z_of_2(self):
        weak = regions.region_table(
            make_grid({'Alpha': {'F3': 1.9, 'C3': 1.9}}))
        self.assertFalse(weak['Alpha']['left_medial']['flag'])
        strong = regions.region_table(
            make_grid({'Alpha': {'F3': 2.0, 'C3': 2.0}}))
        self.assertTrue(strong['Alpha']['left_medial']['flag'])

    def test_negative_cluster_flags_too(self):
        grid = make_grid({'Beta': {'F8': -2.2, 'T4': -2.2, 'T6': -2.2}})
        entry = regions.region_table(grid)['Beta']['right_lateral']
        self.assertAlmostEqual(entry['mean_z'], -2.2)
        self.assertTrue(entry['flag'])

    def test_none_z_is_not_a_site_and_nothing_is_filled_in(self):
        grid = make_grid({'Alpha': {'F3': None, 'C3': 2.5, 'P3': 2.5}})
        entry = regions.region_table(grid)['Alpha']['left_medial']
        self.assertEqual(entry['n_sites'], 2)
        self.assertNotIn('F3', entry['sites'])
        self.assertNotIn('Fp1', entry['sites'])
        self.assertNotIn('O1', entry['sites'])

    def test_row_selection(self):
        grid = make_grid({'Alpha': {'F3': 2.5, 'C3': 2.5, 'P3': 2.5}})
        self.assertEqual(regions.region_table(grid, row='coherence'), {})
        pooled = regions.region_table(grid, row=None)
        self.assertAlmostEqual(pooled['Alpha']['left_medial']['mean_z'], 2.5)

    def test_missing_row_and_band_are_empty_not_invented(self):
        grid = make_grid({'Alpha': {'F3': 2.5, 'C3': 2.5}})
        self.assertEqual(regions.region_table(grid, row='phase'), {})
        self.assertEqual(regions.region_table({'rows': {}}), {})

    def test_bad_grid_raises(self):
        with self.assertRaises(ValueError):
            regions.region_table('not a grid')
        with self.assertRaises(ValueError):
            regions.region_table({'channels': []})


class RegionRowsTests(unittest.TestCase):
    def test_flat_rows_are_sorted_by_absolute_z(self):
        grid = make_grid({'Alpha': {'F3': 2.5, 'C3': 2.5, 'P3': 2.5,
                                    'F8': -3.0, 'T4': -3.0, 'T6': -3.0}})
        items = regions.region_rows(grid)
        self.assertTrue(items)
        self.assertEqual(set(items[0]), {'band', 'region', 'mean_z',
                                         'n_sites', 'flag'})
        self.assertEqual(items[0]['region'], 'right_lateral')
        self.assertAlmostEqual(items[0]['mean_z'], -3.0)
        self.assertEqual(items[1]['region'], 'left_medial')
        self.assertTrue(all(item['band'] == 'Alpha' for item in items))

    def test_rows_carry_band_order(self):
        grid = make_grid({'Alpha': {'F3': 2.0, 'C3': 2.0},
                          'Beta': {'F3': 2.0, 'C3': 2.0}})
        items = regions.region_rows(grid)
        self.assertEqual([item['band'] for item in items],
                         ['Alpha', 'Beta'])
        self.assertEqual(set(regions.band_order(grid)), {'Alpha', 'Beta'})


if __name__ == '__main__':
    unittest.main()
