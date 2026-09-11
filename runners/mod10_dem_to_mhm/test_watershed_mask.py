import importlib.util
from pathlib import Path
import sys
import unittest

import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box


MODULE_PATH = Path(__file__).with_name("main.py")
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("mod10_main", MODULE_PATH)
mod10 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod10)


class WatershedMaskTests(unittest.TestCase):
    def test_rasterize_domain_mask_uses_l0_grid_and_all_touched(self):
        domain = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(1.5, 1.5, 2.5, 2.5)], crs="EPSG:5070"
        )

        mask = mod10._rasterize_domain_mask(
            domain, shape=(4, 4), transform=from_origin(0, 4, 1, 1)
        )

        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.sum(), 4)
        self.assertTrue(mask[1:3, 1:3].all())

    def test_preserves_internal_directions_and_multiple_exits(self):
        mask = np.array(
            [
                [False, True, True, False],
                [False, True, True, True],
                [False, False, True, True],
            ]
        )
        fdir = np.array(
            [
                [-9999, 1, 64, -9999],
                [-9999, 1, 4, 1],
                [-9999, -9999, 2, 4],
            ],
            dtype=np.int32,
        )

        result, outlet_count = mod10._set_boundary_outlets(fdir, mask)

        self.assertEqual(result[0, 1], 1)
        self.assertEqual(result[1, 1], 1)
        self.assertEqual(result[1, 2], 4)
        self.assertEqual(result[0, 2], 0)  # north edge
        self.assertEqual(result[1, 3], 0)  # east edge
        self.assertEqual(result[2, 2], 0)  # diagonal edge
        self.assertEqual(result[2, 3], 0)  # south edge
        self.assertEqual(outlet_count, 4)
        self.assertTrue(np.array_equal(result[~mask], fdir[~mask]))

    def test_counts_existing_outlet(self):
        mask = np.ones((1, 2), dtype=bool)
        fdir = np.array([[1, 0]], dtype=np.int32)

        result, outlet_count = mod10._set_boundary_outlets(fdir, mask)

        self.assertTrue(np.array_equal(result, fdir))
        self.assertEqual(outlet_count, 1)

    def test_rejects_invalid_codes(self):
        mask = np.ones((1, 1), dtype=bool)

        with self.assertRaisesRegex(ValueError, "Invalid ArcGIS flow-direction codes"):
            mod10._set_boundary_outlets(np.array([[3]], dtype=np.int32), mask)

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            mod10._set_boundary_outlets(
                np.ones((2, 2), dtype=np.int32), np.ones((1, 1), dtype=bool)
            )


if __name__ == "__main__":
    unittest.main()
