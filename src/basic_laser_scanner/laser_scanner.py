# Backward-compatible re-export.  New code should import LaserScannerGeo
# from laser_scanner_geo directly.
from .laser_scanner_geo import LaserScannerGeo as LaserScanner

__all__ = ["LaserScanner"]
