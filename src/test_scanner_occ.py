"""Tests for OccupancyMap and LaserScannerOcc."""
import math
import os
import pathlib
import tempfile
import unittest

import numpy as np

from basic_map.map_occupancy import OccupancyMap
from basic_laser_scanner.laser_scanner_occ import LaserScannerOcc
from basic_laser_scanner.laser_output import LaserScanOutput


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_SPARSE = os.path.join(ROOT_DIR, "config", "sparse_scanner_spec.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_map(size: int = 100, res: float = 0.1) -> OccupancyMap:
    """Create a square room with walls on all four edges.

    Map size: ``size × size`` pixels.
    Real-world size: ``size * res × size * res`` metres.
    """
    grid = np.zeros((size, size), dtype=bool)
    # Walls on all four edges (1 pixel thick)
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True
    grid[60:80, 60:80] = True
    return OccupancyMap.from_numpy(grid, resolution=res, origin=(0.0, 0.0))


# ---------------------------------------------------------------------------
# OccupancyMap tests
# ---------------------------------------------------------------------------

class TestOccupancyMapFromNumpy(unittest.TestCase):

    def setUp(self):
        self.occ = _simple_map(size=100, res=0.1)

    def test_shape(self):
        self.assertEqual(self.occ.height, 100)
        self.assertEqual(self.occ.width, 100)

    def test_grid_type(self):
        self.assertEqual(self.occ.grid.dtype, bool)

    def test_walls_are_occupied(self):
        # Top/bottom walls
        self.assertTrue(self.occ.is_occupied(0, 50))
        self.assertTrue(self.occ.is_occupied(99, 50))
        # Left/right walls
        self.assertTrue(self.occ.is_occupied(50, 0))
        self.assertTrue(self.occ.is_occupied(50, 99))

    def test_interior_is_free(self):
        self.assertFalse(self.occ.is_occupied(50, 50))

    def test_internal_obstacle_occupied(self):
        self.assertTrue(self.occ.is_occupied(60, 60))

    def test_out_of_bounds_returns_false(self):
        self.assertFalse(self.occ.is_occupied(-1, 0))
        self.assertFalse(self.occ.is_occupied(0, 200))


class TestOccupancyMapCoordinateTransform(unittest.TestCase):

    def setUp(self):
        # 100×100 grid, 0.1 m/px, origin at (0,0) = bottom-left world.
        self.occ = OccupancyMap.from_numpy(
            np.zeros((100, 100), dtype=bool), resolution=0.1, origin=(0.0, 0.0)
        )

    def test_world_to_pixel_origin(self):
        # World (0,0) → bottom-left pixel = (row=99, col=0)
        row, col = self.occ.world_to_pixel(0.0, 0.0)
        self.assertEqual(row, 99)
        self.assertEqual(col, 0)

    def test_world_to_pixel_top_right(self):
        # World (9.9, 9.9) → top-right pixel ≈ (row=0, col=99)
        row, col = self.occ.world_to_pixel(9.9, 9.9)
        self.assertEqual(row, 0)
        self.assertEqual(col, 99)

    def test_pixel_to_world_round_trip(self):
        for (ri, ci) in [(0, 0), (50, 50), (99, 0), (0, 99)]:
            x, y = self.occ.pixel_to_world(ri, ci)
            ri2, ci2 = self.occ.world_to_pixel(x, y)
            self.assertEqual(ri2, ri, msg=f"row mismatch for ({ri},{ci})")
            self.assertEqual(ci2, ci, msg=f"col mismatch for ({ri},{ci})")


class TestOccupancyMapFromNpy(unittest.TestCase):

    def test_round_trip_npy(self):
        grid = np.array([[0, 1], [1, 0]], dtype=bool)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            path = f.name
        try:
            np.save(path, grid)
            occ = OccupancyMap.from_npy(path, resolution=0.5, origin=(1.0, 2.0))
            np.testing.assert_array_equal(occ.grid, grid)
            self.assertAlmostEqual(occ.resolution, 0.5)
        finally:
            os.unlink(path)


class TestOccupancyMapFromImage(unittest.TestCase):

    def test_from_png_image(self):
        """Save a PNG and reload it; check that dark pixels become occupied."""
        import matplotlib.pyplot as plt  # type: ignore

        grid_orig = np.zeros((20, 20), dtype=np.uint8)
        grid_orig[0, :] = 255   # white row  → free
        grid_orig[10, :] = 0    # black row  → occupied (dark)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            plt.imsave(path, grid_orig, cmap="gray", vmin=0, vmax=255)
            occ = OccupancyMap.from_image(path, threshold=0.5)
            # Row 0 in the saved image (top) = white → free
            self.assertFalse(occ.grid[0, 0])
            # Row 10 = black → occupied
            self.assertTrue(occ.grid[10, 0])
        finally:
            os.unlink(path)


class TestOccupancyMapInvalidInput(unittest.TestCase):

    def test_3d_grid_raises(self):
        with self.assertRaises(ValueError):
            OccupancyMap.from_numpy(np.zeros((10, 10, 3)))

    def test_zero_resolution_raises(self):
        with self.assertRaises(ValueError):
            OccupancyMap(np.zeros((10, 10), dtype=bool), resolution=0.0)


# ---------------------------------------------------------------------------
# LaserScannerOcc tests
# ---------------------------------------------------------------------------

class TestLaserScannerOccInterface(unittest.TestCase):

    def test_scan_before_load_map_raises(self):
        scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_scan_before_load_scanner_raises(self):
        scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        scanner.load_map(_simple_map())
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_invalid_state_raises(self):
        scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        scanner.load_map(_simple_map())
        scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0])  # missing heading

    def test_wrong_map_type_raises(self):
        scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        with self.assertRaises(TypeError):
            scanner.load_map("not_a_map")  # type: ignore


class TestLaserScannerOccOutput(unittest.TestCase):

    def setUp(self):
        self.occ = _simple_map(size=100, res=0.1)
        self.scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        self.scanner.load_map(self.occ)
        # Place scanner near centre of the room (world coords).
        # Room is 10×10 m; centre is (5, 5).
        state = [5.0, 5.0, 0.0]
        self.scanner.load_scanner((state[0], state[1]), state[2])
        self.scan = self.scanner.scan(0.0, state)

    def test_output_type(self):
        self.assertIsInstance(self.scan, LaserScanOutput)

    def test_ranges_count_matches_angles(self):
        self.assertEqual(len(self.scan.ranges), len(self.scan.angles))

    def test_beam_count_matches_angles(self):
        self.assertEqual(len(self.scan.beam_end_points), len(self.scan.angles))

    def test_ranges_within_bounds(self):
        for r in self.scan.ranges:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, self.scan.range_max + 1e-6)

    def test_ranges_are_finite(self):
        for r in self.scan.ranges:
            self.assertTrue(math.isfinite(r))


class TestLaserScannerOccKnownRange(unittest.TestCase):
    """Check that ray-casting gives physically plausible distances."""

    def _scan_at(self, x, y, heading_deg):
        occ = _simple_map(size=100, res=0.1)
        scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        scanner.load_map(occ)
        state = [x, y, math.radians(heading_deg)]
        scanner.load_scanner((x, y), state[2])
        return scanner.scan(0.0, state)

    def test_centre_symmetric_scan(self):
        """Scanner at centre (5, 5), heading 0°.

        The room is 10 × 10 m; walls are 1 pixel = 0.1 m thick.
        From (5, 5) heading east (0°) the right wall face is at x ≈ 9.9,
        so the range to the east wall ≈ 4.9 m.
        """
        scan = self._scan_at(5.0, 5.0, 0)
        # Find the 0° beam (relative angle ≈ 0).
        idx = next(i for i, a in enumerate(scan.angles) if abs(a) < 0.01)
        # Expect roughly 4.9 m (wall at x=9.9, scanner at x=5.0).
        self.assertAlmostEqual(scan.ranges[idx], 4.9, delta=0.15)

    def test_range_to_internal_obstacle(self):
        """Scanner at (2, 3) heading east (0°), internal obstacle starts at
        pixel col=60 → world x = 0 + 60*0.1 = 6.0.
        The obstacle occupies rows [60:80] which maps to y=[2.0, 4.0).
        At y=3.0 (row=69), the beam hits the west face at x=6.0 → range=4.0 m."""
        scan = self._scan_at(2.0, 3.0, 0)
        idx = next(i for i, a in enumerate(scan.angles) if abs(a) < 0.01)
        # obstacle west edge ≈ x=6.0; scanner at x=2.0 → range ≈ 4.0 m
        self.assertAlmostEqual(scan.ranges[idx], 4.0, delta=0.2)

    def test_symmetric_left_right(self):
        """From centre heading east, ±90° beams should reach the north/south
        walls symmetrically and have similar ranges."""
        scan = self._scan_at(5.0, 5.0, 0)
        idx_plus = next(
            i for i, a in enumerate(scan.angles) if abs(a - math.pi / 2) < 0.01
        )
        idx_minus = next(
            i for i, a in enumerate(scan.angles) if abs(a + math.pi / 2) < 0.01
        )
        self.assertAlmostEqual(scan.ranges[idx_plus], scan.ranges[idx_minus], delta=0.2)


class TestLaserScannerOccGeometricConsistency(unittest.TestCase):
    """Compare OccupancyMap scanner against GeometricMap scanner.

    A simple rectangular room with one box obstacle is created both as a
    GeometricMap and as a matching high-resolution OccupancyMap.  Both
    scanners should agree on the scan ranges within a tolerance determined
    by the occupancy grid resolution.
    """

    def test_ranges_agree_with_geo_scanner(self):
        from basic_map.map_geometric import GeometricMap
        from basic_laser_scanner.laser_scanner_geo import LaserScannerGeo

        # Simple rectangular room with one axis-aligned box obstacle.
        # Room:  x ∈ [0, 10], y ∈ [0, 10]  (boundary just outside)
        boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        obstacle_list = [[[4.0, 4.0], [4.0, 6.0], [6.0, 6.0], [6.0, 4.0]]]
        geo_map = GeometricMap.from_raw(boundary, obstacle_list)

        # Build a matching occupancy map by rasterising manually.
        # Resolution: 0.05 m/px → 200×200 grid for a 10×10 m room.
        res = 0.05
        origin = (0.0, 0.0)
        H = W = int(10.0 / res)  # 200
        grid = np.zeros((H, W), dtype=bool)
        # Boundary walls (1-pixel border)
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True
        # Box obstacle at world x=[4,6], y=[4,6]
        # pixel col: 4/0.05=80 … 6/0.05=120
        # pixel row: H-1-(6/0.05) … H-1-(4/0.05) = 199-120=79 … 199-80=119
        col_lo, col_hi = int(4.0 / res), int(6.0 / res)   # 80, 120
        row_lo, row_hi = H - int(6.0 / res), H - int(4.0 / res)  # 79, 119
        grid[row_lo:row_hi, col_lo:col_hi] = True

        occ_map = OccupancyMap.from_numpy(grid, resolution=res, origin=origin)

        state = [2.0, 2.0, math.radians(45)]

        # Geometric scanner
        geo_scanner = LaserScannerGeo.from_yaml(CONFIG_SPARSE)
        geo_scanner.load_map(*geo_map())
        geo_scanner.load_scanner((state[0], state[1]), state[2])
        geo_scan = geo_scanner.scan(0.0, state)

        # Occupancy scanner
        occ_scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)
        occ_scanner.load_map(occ_map)
        occ_scanner.load_scanner((state[0], state[1]), state[2])
        occ_scan = occ_scanner.scan(0.0, state)

        # Ranges should agree within a generous tolerance given the coarse grid.
        tolerance = 0.3  # metres
        for i, (rg, ro) in enumerate(zip(geo_scan.ranges, occ_scan.ranges)):
            self.assertAlmostEqual(
                rg, ro, delta=tolerance,
                msg=f"Range mismatch at beam {i}: geo={rg:.3f}, occ={ro:.3f}",
            )


if __name__ == "__main__":
    unittest.main()
