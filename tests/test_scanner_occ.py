"""Tests for OccupancyMap and LaserScannerOcc."""
import math
import os
import pathlib
import tempfile
import unittest

import numpy as np

from basic_map.map_occupancy import OccupancyMap
from laser_scanner.laser_scanner_occ import LaserScannerOcc
from laser_scanner import LaserScanOutput


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_SPARSE = os.path.join(ROOT_DIR, "config", "sparse_scanner_spec.yaml")


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

class TestLaserScannerOcc(unittest.TestCase):

    def setUp(self):
        self.occ = _simple_map(size=100, res=0.1)
        self.scanner = LaserScannerOcc.from_yaml(CONFIG_SPARSE)

    def test_from_yaml_builds_scanner(self):
        self.assertIsInstance(self.scanner, LaserScannerOcc)

    def test_load_map_type_error(self):
        with self.assertRaises(TypeError):
            self.scanner.load_map(np.zeros((10, 10), dtype=bool))  # type: ignore[arg-type]

    def test_scan_before_load_map_raises(self):
        self.scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            self.scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_scan_before_load_scanner_raises(self):
        self.scanner.load_map(self.occ)
        with self.assertRaises(ValueError):
            self.scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_scan_invalid_state_raises(self):
        self.scanner.load_map(self.occ)
        self.scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            self.scanner.scan(0.0, [1.0, 1.0])

    def test_scan_output_shape_and_type(self):
        self.scanner.load_map(self.occ)
        self.scanner.load_scanner((1.0, 1.0), 0.0)

        scan = self.scanner.scan(1.23, [1.0, 1.0, 0.0])

        self.assertIsInstance(scan, LaserScanOutput)
        self.assertEqual(len(scan.ranges), len(scan.angles))
        self.assertEqual(len(scan.beam_end_points), len(scan.angles))
        self.assertAlmostEqual(scan.timestamp, 1.23, places=6)
        for r in scan.ranges:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, scan.range_max + 1e-9)

    def test_zero_degree_beam_hits_east_wall(self):
        """From (1.0, 1.0) heading east, 0-degree beam should hit x=9.9 wall."""
        self.scanner.load_map(self.occ)
        self.scanner.load_scanner((1.0, 1.0), 0.0)

        scan = self.scanner.scan(0.0, [1.0, 1.0, 0.0])
        idx = next(i for i, a in enumerate(scan.angles) if abs(a) < 0.01)

        # Right wall starts at the west edge of col=99 => x=9.85.
        self.assertAlmostEqual(scan.ranges[idx], 8.85, delta=0.06)

    def test_ninety_degree_beam_hits_north_wall(self):
        """From (1.0, 1.0) heading north, +90-degree beam hits y=9.85 wall edge."""
        self.scanner.load_map(self.occ)
        self.scanner.load_scanner((1.0, 1.0), 0.0)

        scan = self.scanner.scan(0.0, [1.0, 1.0, 0.0])
        idx = next(i for i, a in enumerate(scan.angles) if abs(a - math.pi / 2) < 0.01)

        self.assertAlmostEqual(scan.ranges[idx], 8.85, delta=0.06)
