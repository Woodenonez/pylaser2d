import os 
import math
import pathlib

from basic_laser_scanner.laser_scanner import LaserScanner
from basic_laser_scanner.laser_output import LaserScanOutput
from basic_map.map_geometric import GeometricMap

CFG_FNAME_1 = "dense_scanner_spec.yaml"
MAP_NAME_1 = "test_map_1"

CFG_FNAME_2 = "sparse_scanner_spec.yaml"
MAP_NAME_2 = "test_map_2"

def test_scanner(config_file_name: str, map_name: str) -> tuple[LaserScanner, LaserScanOutput]:
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    config_fpath = os.path.join(root_dir, "config", config_file_name)
    map_path = os.path.join(root_dir, "data", map_name, "map.json")

    current_time = 0.0
    current_state = [1.0, 1.0, math.radians(45)]

    map_obj = GeometricMap.from_json(map_path)

    scanner = LaserScanner.from_yaml(config_fpath)
    scanner.load_map(*map_obj())
    scanner.load_scanner((current_state[0], current_state[1]), current_state[2])

    laser_scan = scanner.scan(current_time, current_state)
    return scanner, laser_scan

if __name__ == "__main__":
    from typing import cast
    import matplotlib.pyplot as plt # type: ignore
    from matplotlib.axes import Axes # type: ignore

    scanner_1, laser_scan_1 = test_scanner(CFG_FNAME_1, MAP_NAME_1)
    scanner_2, laser_scan_2 = test_scanner(CFG_FNAME_2, MAP_NAME_2)

    print("--- Sparse Scanner Output ---")
    print("Ranges:", laser_scan_2.ranges)
    print("Beam End Points:", laser_scan_2.beam_end_points)

    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1 = cast(Axes, ax1)
    ax2 = cast(Axes, ax2)

    ax1.axis('equal')
    ax1.set_title("Dense Scanner")
    scanner_1.plot(ax1, with_map=True)

    ax2.axis('equal')
    ax2.set_title("Sparse Scanner")
    scanner_2.plot(ax2, with_map=True)

    fig.tight_layout()

    plt.show()


