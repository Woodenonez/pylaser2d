"""Demo: robot motion with laser scanning on geometric or occupancy maps."""

from __future__ import annotations

import argparse
import os
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.artist import Artist

from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap
from laser_scanner import LaserScanner, LaserScanOutput


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "dense_scanner_spec.yaml")
MAP_PATH = os.path.join(ROOT_DIR, "data", "test_map_1", "map.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Laser scan demo for geo or occ maps")
    parser.add_argument(
        "--map-type",
        choices=["geo", "occ"],
        default="geo",
        help="Map backend to visualize.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Animation frame stride. Smaller is smoother but heavier.",
    )
    return parser.parse_args()


def create_simple_occupancy_map(size: int = 100, resolution: float = 0.1) -> OccupancyMap:
    """Create a simple occupancy map with walls and a few obstacles."""
    grid = np.zeros((size, size), dtype=bool)

    grid[0:2, :] = True
    grid[-2:, :] = True
    grid[:, 0:2] = True
    grid[:, -2:] = True

    grid[40:60, 30:35] = True
    grid[55:60, 30:50] = True
    grid[20:30, 60:70] = True

    return OccupancyMap.from_numpy(grid, resolution=resolution, origin=(0.0, 0.0))


def build_path_geo(map_obj: GeometricMap, num_points: int = 100) -> list[tuple[float, float]]:
    """Create a circular path inside a geometric map boundary."""
    boundary = np.array(map_obj.boundary_coords)
    center_x = float(np.mean(boundary[:, 0]))
    center_y = float(np.mean(boundary[:, 1]))
    radius = 1.5
    t_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    return [(center_x + radius * np.cos(t), center_y + radius * np.sin(t)) for t in t_vals]


def build_path_occ(occ_map: OccupancyMap, num_points: int = 150) -> list[tuple[float, float]]:
    """Create a circular path inside an occupancy map."""
    width_m = occ_map.width * occ_map.resolution
    height_m = occ_map.height * occ_map.resolution
    center_x = width_m / 2.0
    center_y = height_m / 2.0
    radius = min(width_m, height_m) / 4.0
    t_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    return [(center_x + radius * np.cos(t), center_y + radius * np.sin(t)) for t in t_vals]


def _plot_beams(ax: plt.Axes, robot_pos, robot_heading, laser_scan: LaserScanOutput, robot_radius: float, arrow_length: float) -> None:
    ranges = laser_scan.ranges
    range_max = laser_scan.range_max
    beam_endpoints = laser_scan.beam_end_points

    cmap = plt.get_cmap("cool")
    for endpoint, beam_range in zip(beam_endpoints, ranges):
        color_val = beam_range / range_max if range_max > 0.0 else 0.0
        color = cmap(1 - color_val)
        ax.plot(
            [robot_pos[0], endpoint[0]],
            [robot_pos[1], endpoint[1]],
            color=color,
            alpha=0.3,
            linewidth=0.5,
        )
        ax.plot(endpoint[0], endpoint[1], "o", color=color, markersize=2)

    circle = plt.Circle(robot_pos, robot_radius, color="red", alpha=0.8, label="Robot")
    ax.add_patch(circle)

    arrow_end_x = robot_pos[0] + arrow_length * np.cos(robot_heading)
    arrow_end_y = robot_pos[1] + arrow_length * np.sin(robot_heading)
    arrow = FancyArrowPatch(
        robot_pos,
        (arrow_end_x, arrow_end_y),
        arrowstyle="->",
        mutation_scale=18,
        color="red",
        linewidth=2,
    )
    ax.add_patch(arrow)


def visualize_geo(ax: plt.Axes, map_obj: GeometricMap, laser_scan, robot_pos, robot_heading) -> None:
    """Render one geometric-map scan frame."""
    ax.clear()

    boundary = np.array(map_obj.boundary_coords + [map_obj.boundary_coords[0]])
    ax.plot(boundary[:, 0], boundary[:, 1], "k-", linewidth=2, label="Boundary")

    for obstacle in map_obj.obstacle_coords_list:
        obs_array = np.array(obstacle + [obstacle[0]])
        ax.fill(obs_array[:, 0], obs_array[:, 1], color="gray", alpha=0.5)
        ax.plot(obs_array[:, 0], obs_array[:, 1], "k-", linewidth=1)

    _plot_beams(ax, robot_pos, robot_heading, laser_scan, robot_radius=0.15, arrow_length=0.3)

    boundary_array = np.array(map_obj.boundary_coords)
    margin = 0.5
    ax.set_xlim(boundary_array[:, 0].min() - margin, boundary_array[:, 0].max() + margin)
    ax.set_ylim(boundary_array[:, 1].min() - margin, boundary_array[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    ax.set_title(
        "Geometric Map - Laser Scan Demo\n"
        f"Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Heading: {np.degrees(robot_heading):.1f}°"
    )


def visualize_occ(ax: plt.Axes, occ_map: OccupancyMap, laser_scan, robot_pos, robot_heading) -> None:
    """Render one occupancy-map scan frame with cell-center aligned display."""
    ax.clear()

    ox, oy = occ_map.origin
    res = occ_map.resolution
    extent = (
        ox - 0.5 * res,
        ox + (occ_map.width - 0.5) * res,
        oy - 0.5 * res,
        oy + (occ_map.height - 0.5) * res,
    )
    cmap = ListedColormap(["lightgray", "darkgray"])
    ax.imshow(np.flipud(occ_map.grid), cmap=cmap, extent=extent, origin="lower", alpha=0.8)

    _plot_beams(ax, robot_pos, robot_heading, laser_scan, robot_radius=0.1, arrow_length=0.2)

    width_m = occ_map.width * res
    height_m = occ_map.height * res
    ax.set_xlim(ox - 0.5 * res, ox + (occ_map.width - 0.5) * res)
    ax.set_ylim(oy - 0.5 * res, oy + (occ_map.height - 0.5) * res)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    ax.set_title(
        "Occupancy Map - Laser Scan Demo\n"
        f"Robot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Heading: {np.degrees(robot_heading):.1f}°"
    )


def main() -> None:
    args = _parse_args()
    scanner = LaserScanner.from_yaml(CONFIG_PATH)

    if args.map_type == "geo":
        map_obj = GeometricMap.from_json(MAP_PATH)
        scanner.load_map(map_obj)
        waypoints = build_path_geo(map_obj)
        renderer = lambda ax, scan, pos, heading: visualize_geo(ax, map_obj, scan, pos, heading)
        frame_stride = max(1, args.step)
    else:
        map_obj = create_simple_occupancy_map(size=100, resolution=0.1)
        scanner.load_map(map_obj)
        waypoints = build_path_occ(map_obj)
        renderer = lambda ax, scan, pos, heading: visualize_occ(ax, map_obj, scan, pos, heading)
        frame_stride = max(1, args.step)

    start_pos = waypoints[0]
    scanner.load_scanner(start_pos, 0.0)

    fig, ax = plt.subplots(figsize=(10, 10))

    def update_frame(frame_idx: int) -> list[Artist]:
        waypoint = waypoints[frame_idx]

        if frame_idx < len(waypoints) - 1:
            next_point = waypoints[frame_idx + 1]
            heading = float(np.arctan2(
                next_point[1] - waypoint[1],
                next_point[0] - waypoint[0],
            ))
        else:
            heading = 0.0

        state = [waypoint[0], waypoint[1], heading]
        laser_scan = scanner.scan(float(frame_idx) * 0.1, state)
        renderer(ax, laser_scan, waypoint, heading)

        return []

    anim = FuncAnimation(
        fig,
        update_frame,
        frames=range(0, len(waypoints), frame_stride),
        repeat=True,
        repeat_delay=1000,
        blit=False,
    )

    _ = anim
    plt.show()


if __name__ == "__main__":
    main()