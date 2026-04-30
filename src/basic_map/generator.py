"""Random map generation helpers for geometric and occupancy maps."""

from __future__ import annotations

from typing import Any, Literal
import random

import numpy as np

from .map_geometric import GeometricMap
from .map_occupancy import OccupancyMap


def _grow_blob(
    grid: np.ndarray,
    rng: random.Random,
    start_row: int,
    start_col: int,
    target_cells: int,
) -> int:
    """Grow one irregular obstacle blob using a randomized frontier expansion."""
    height, width = grid.shape
    frontier = [(start_row, start_col)]
    blob_cells: set[tuple[int, int]] = set()

    while frontier and len(blob_cells) < target_cells:
        index = rng.randrange(len(frontier))
        row, col = frontier.pop(index)

        if row <= 0 or row >= height - 1 or col <= 0 or col >= width - 1:
            continue
        if grid[row, col] or (row, col) in blob_cells:
            continue

        blob_cells.add((row, col))

        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
            (row - 1, col - 1),
            (row - 1, col + 1),
            (row + 1, col - 1),
            (row + 1, col + 1),
        ]
        rng.shuffle(neighbors)
        for next_row, next_col in neighbors:
            if next_row <= 0 or next_row >= height - 1 or next_col <= 0 or next_col >= width - 1:
                continue
            if grid[next_row, next_col] or (next_row, next_col) in blob_cells:
                continue
            if rng.random() < 0.82:
                frontier.append((next_row, next_col))

        if not frontier and len(blob_cells) < target_cells:
            row2, col2 = rng.choice(tuple(blob_cells))
            frontier.extend(
                [
                    (row2 - 1, col2),
                    (row2 + 1, col2),
                    (row2, col2 - 1),
                    (row2, col2 + 1),
                ]
            )

    for row, col in blob_cells:
        grid[row, col] = True

    return len(blob_cells)


def _rectangles_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad: float = 0.0,
) -> bool:
    """Return True if two axis-aligned rectangles overlap (with optional padding)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        (ax1 + pad) <= bx0
        or (bx1 + pad) <= ax0
        or (ay1 + pad) <= by0
        or (by1 + pad) <= ay0
    )


def gen_random_geo_map(
    width: float = 10.0,
    height: float = 10.0,
    num_obstacles: int = 8,
    min_obstacle_size: float = 0.5,
    max_obstacle_size: float = 2.0,
    border_margin: float = 0.35,
    obstacle_margin: float = 0.15,
    seed: int | None = None,
    max_attempts_per_obstacle: int = 150,
) -> GeometricMap:
    """Generate a random geometric map with axis-aligned rectangular obstacles.

    Args:
        width: Map width in metres.
        height: Map height in metres.
        num_obstacles: Number of obstacles to place.
        min_obstacle_size: Minimum obstacle side length.
        max_obstacle_size: Maximum obstacle side length.
        border_margin: Minimum clearance from outer boundary.
        obstacle_margin: Minimum clearance between obstacles.
        seed: Random seed for reproducible generation.
        max_attempts_per_obstacle: Max retries for placing each obstacle.
    """
    if width <= 0.0 or height <= 0.0:
        raise ValueError("width and height must be positive.")
    if num_obstacles < 0:
        raise ValueError("num_obstacles must be >= 0.")
    if min_obstacle_size <= 0.0 or max_obstacle_size <= 0.0:
        raise ValueError("Obstacle sizes must be positive.")
    if min_obstacle_size > max_obstacle_size:
        raise ValueError("min_obstacle_size must be <= max_obstacle_size.")

    rng = random.Random(seed)

    boundary = [
        (0.0, 0.0),
        (width, 0.0),
        (width, height),
        (0.0, height),
    ]

    placed_rects: list[tuple[float, float, float, float]] = []
    obstacles: list[list[tuple[float, float]]] = []

    for _ in range(num_obstacles):
        placed = False
        for _ in range(max_attempts_per_obstacle):
            w = rng.uniform(min_obstacle_size, max_obstacle_size)
            h = rng.uniform(min_obstacle_size, max_obstacle_size)

            x_min = border_margin
            x_max = width - border_margin - w
            y_min = border_margin
            y_max = height - border_margin - h

            if x_max <= x_min or y_max <= y_min:
                break

            x0 = rng.uniform(x_min, x_max)
            y0 = rng.uniform(y_min, y_max)
            x1 = x0 + w
            y1 = y0 + h
            rect = (x0, y0, x1, y1)

            if any(_rectangles_overlap(rect, r, pad=obstacle_margin) for r in placed_rects):
                continue

            placed_rects.append(rect)
            obstacles.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            placed = True
            break

        if not placed:
            # Stop early if additional obstacles cannot be placed with constraints.
            break

    return GeometricMap.from_raw(boundary, obstacles)


def gen_random_occ_map(
    width: int = 120,
    height: int = 120,
    resolution: float = 0.1,
    origin: tuple[float, float] = (0.0, 0.0),
    target_occupancy: float = 0.2,
    min_rect_size: int = 4,
    max_rect_size: int = 22,
    add_border_walls: bool = True,
    seed: int | None = None,
    max_rectangles: int = 250,
) -> tuple[OccupancyMap, dict[str, Any]]:
    """Generate a random occupancy map and return map plus generation metadata.

    Args:
        width: Grid width in pixels.
        height: Grid height in pixels.
        resolution: Metres per pixel.
        origin: World coordinate of bottom-left pixel.
        target_occupancy: Target occupied ratio in [0, 1].
        min_rect_size: Minimum irregular obstacle scale in pixels.
        max_rect_size: Maximum irregular obstacle scale in pixels.
        add_border_walls: Whether to mark 1-pixel border as occupied.
        seed: Random seed for reproducibility.
        max_rectangles: Maximum number of irregular obstacle blobs to attempt.
    """
    if width <= 3 or height <= 3:
        raise ValueError("width and height must be > 3.")
    if not (0.0 <= target_occupancy <= 1.0):
        raise ValueError("target_occupancy must be in [0, 1].")
    if min_rect_size <= 0 or max_rect_size <= 0:
        raise ValueError("Rectangle sizes must be positive integers.")
    if min_rect_size > max_rect_size:
        raise ValueError("min_rect_size must be <= max_rect_size.")

    rng = random.Random(seed)
    grid = np.zeros((height, width), dtype=bool)

    if add_border_walls:
        grid[0, :] = True
        grid[-1, :] = True
        grid[:, 0] = True
        grid[:, -1] = True

    target_pixels = int(target_occupancy * grid.size)
    blobs_used = 0

    while int(grid.sum()) < target_pixels and blobs_used < max_rectangles:
        start_col = rng.randint(1, width - 2)
        start_row = rng.randint(1, height - 2)

        if grid[start_row, start_col]:
            blobs_used += 1
            continue

        blob_scale = rng.randint(min_rect_size, max_rect_size)
        target_blob_cells = rng.randint(
            max(4, blob_scale * blob_scale // 2),
            max(5, blob_scale * blob_scale),
        )
        _grow_blob(grid, rng, start_row, start_col, target_blob_cells)
        blobs_used += 1

    occ_map = OccupancyMap.from_numpy(grid, resolution=resolution, origin=origin)
    params: dict[str, Any] = {
        "width_px": width,
        "height_px": height,
        "resolution": resolution,
        "origin": origin,
        "target_occupancy": target_occupancy,
        "actual_occupancy": float(grid.mean()),
        "rectangles_used": blobs_used,
        "blobs_used": blobs_used,
        "obstacle_style": "irregular_blob",
        "add_border_walls": add_border_walls,
        "seed": seed,
    }
    return occ_map, params


def gen_random_map(
    map_type: Literal["geo", "occ"] = "geo",
    **kwargs: Any,
) -> tuple[GeometricMap | OccupancyMap, dict[str, Any]]:
    """Generate a random map based on map_type.

    Args:
        map_type: "geo" for geometric map, "occ" for occupancy map.
        **kwargs: Passed through to the chosen generator.

    Returns:
        Tuple of (map_object, metadata).
        - geo: (GeometricMap, {"map_type": "geo", ...})
        - occ: (OccupancyMap, {"map_type": "occ", ...})
    """
    if map_type == "geo":
        geo_map = gen_random_geo_map(**kwargs)
        meta = {
            "map_type": "geo",
            "width": kwargs.get("width", 10.0),
            "height": kwargs.get("height", 10.0),
            "num_obstacles_requested": kwargs.get("num_obstacles", 8),
            "num_obstacles_actual": len(geo_map.obstacle_coords_list),
            "seed": kwargs.get("seed"),
        }
        return geo_map, meta

    if map_type == "occ":
        occ_map, meta = gen_random_occ_map(**kwargs)
        meta["map_type"] = "occ"
        return occ_map, meta

    raise ValueError(f"Unsupported map_type: {map_type}. Use 'geo' or 'occ'.")
