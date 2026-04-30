"""Demo: generate and visualize a random geometric or occupancy map."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from basic_map.generator import gen_random_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random map generation demo")
    parser.add_argument(
        "--map-type",
        choices=["geo", "occ"],
        default="geo",
        help="Type of map to generate.",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=3,
        help="Complexity level (1-6). Higher means denser/more obstacles.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=10,
        help="Width of the map.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=10,
        help="Height of the map.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    complexity = max(1, min(6, args.complexity))

    if args.map_type == "geo":
        width = float(args.width)
        height = float(args.height)
        num_obstacles = 2 + complexity * 2
        geo_map, meta = gen_random_map(
            map_type="geo",
            width=width,
            height=height,
            num_obstacles=num_obstacles,
            min_obstacle_size=0.5,
            max_obstacle_size=1.2 + 0.25 * complexity,
            seed=args.seed,
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        geo_map.plot(ax)  # type: ignore[attr-defined]
        ax.set_aspect("equal")
        ax.set_title(
            "Random Geometric Map\n"
            f"obstacles={meta['num_obstacles_actual']} (requested={meta['num_obstacles_requested']}), "
            f"seed={meta['seed']}"
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return

    # Occupancy-map branch
    width_px = args.width * 10  # Convert from meters to pixels (assuming 0.1 m/px)
    height_px = args.height * 10
    target_occupancy = 0.06 + 0.045 * complexity
    occ_map, meta = gen_random_map(
        map_type="occ",
        width=width_px,
        height=height_px,
        resolution=0.1,
        target_occupancy=min(target_occupancy, 0.45),
        min_rect_size=3,
        max_rect_size=8 + 3 * complexity,
        seed=args.seed,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    occ_map.plot(ax)  # type: ignore[attr-defined]
    ax.set_aspect("equal")
    ax.set_title(
        "Random Occupancy Map\n"
        f"target={meta['target_occupancy']:.2f}, actual={meta['actual_occupancy']:.2f}, "
        f"blobs={meta['blobs_used']}, seed={meta['seed']}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
