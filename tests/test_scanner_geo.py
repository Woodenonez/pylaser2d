"""Tests for GeometricMap and LaserScannerGeo."""
import math
import os
import pathlib
import unittest

from basic_map.map_geometric import GeometricMap
from laser_scanner.laser_scanner_geo import LaserScannerGeo
from laser_scanner import LaserScanOutput


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_DENSE = os.path.join(ROOT_DIR, "config", "dense_scanner_spec.yaml")
CONFIG_SPARSE = os.path.join(ROOT_DIR, "config", "sparse_scanner_spec.yaml")
MAP_1 = os.path.join(ROOT_DIR, "data", "test_map_1", "map.json")
MAP_2 = os.path.join(ROOT_DIR, "data", "test_map_2", "map.json")


def _build_scanner(config_fpath: str, map_fpath: str,
                   state: list[float]) -> tuple[LaserScannerGeo, LaserScanOutput]:
    map_obj = GeometricMap.from_json(map_fpath)
    scanner = LaserScannerGeo.from_yaml(config_fpath)
    scanner.load_map(*map_obj())
    scanner.load_scanner((state[0], state[1]), state[2])
    laser_scan = scanner.scan(0.0, state)
    return scanner, laser_scan


class TestLaserScannerGeoInterface(unittest.TestCase):
    """Verify that LaserScannerGeo raises the right errors and accepts valid input."""

    def test_scan_before_load_map_raises(self):
        scanner = LaserScannerGeo.from_yaml(CONFIG_DENSE)
        scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_scan_before_load_scanner_raises(self):
        map_obj = GeometricMap.from_json(MAP_1)
        scanner = LaserScannerGeo.from_yaml(CONFIG_DENSE)
        scanner.load_map(*map_obj())
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0, 0.0])

    def test_invalid_state_raises(self):
        map_obj = GeometricMap.from_json(MAP_1)
        scanner = LaserScannerGeo.from_yaml(CONFIG_DENSE)
        scanner.load_map(*map_obj())
        scanner.load_scanner((1.0, 1.0), 0.0)
        with self.assertRaises(ValueError):
            scanner.scan(0.0, [1.0, 1.0])  # only 2 elements


class TestLaserScannerGeoOutput(unittest.TestCase):
    """Verify output properties of the scan."""

    def setUp(self):
        self.state = [1.0, 1.0, math.radians(45)]
        self.scanner, self.scan = _build_scanner(CONFIG_DENSE, MAP_1, self.state)

    def test_output_type(self):
        self.assertIsInstance(self.scan, LaserScanOutput)

    def test_ranges_count_matches_angles(self):
        self.assertEqual(len(self.scan.ranges), len(self.scan.angles))

    def test_beam_end_points_count_matches_angles(self):
        self.assertEqual(len(self.scan.beam_end_points), len(self.scan.angles))

    def test_ranges_within_bounds(self):
        for r in self.scan.ranges:
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, self.scan.range_max + 1e-9)

    def test_scan_inside_boundary_produces_finite_ranges(self):
        for r in self.scan.ranges:
            self.assertTrue(math.isfinite(r))


class TestLaserScannerGeoSparseMap(unittest.TestCase):
    """Dense-map scanner with a simpler box obstacle and specific known range."""

    def test_ranges_are_positive(self):
        state = [1.0, 1.0, 0.0]
        _, scan = _build_scanner(CONFIG_SPARSE, MAP_2, state)
        for r in scan.ranges:
            self.assertGreater(r, 0.0)

    def test_straight_beam_hits_obstacle(self):
        """Pointing straight toward the L-shaped obstacle in map_1 from
        (1.0, 4.7) heading east (0°), the leftmost obstacle edge is at
        x=6.0, so the range must be ≈5.0 m."""
        # The obstacle has a vertical face at x=6.0 spanning y=[4.5, 5.0].
        # From (1.0, 4.7) heading east (0°), the beam hits x=6.0 → range=5.0.
        state = [1.0, 4.7, 0.0]
        _, scan = _build_scanner(CONFIG_SPARSE, MAP_1, state)
        idx = next(i for i, a in enumerate(scan.angles) if abs(a) < 0.01)
        self.assertAlmostEqual(scan.ranges[idx], 5.0, delta=0.05)


class TestLaserScannerGeoFromYaml(unittest.TestCase):
    def test_from_yaml_builds_scanner(self):
        scanner = LaserScannerGeo.from_yaml(CONFIG_DENSE)
        self.assertIsInstance(scanner, LaserScannerGeo)


if __name__ == "__main__":
    unittest.main()
