from laser_scanner import LaserScanner
from basic_map import GeometricMap

scanner = LaserScanner.from_yaml("config/dense_scanner_spec.yaml")
geo_map = GeometricMap.from_json("data/test_map_1/map.json")

scanner.load_map(geo_map)
scanner.load_scanner((1.0, 1.0), 0.0)
scan = scanner.scan(0.0, [1.0, 1.0, 0.0])

print(scan.ranges[:5])