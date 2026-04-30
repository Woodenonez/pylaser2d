import math
import yaml  # type: ignore
from typing import Optional

import numpy as np
from matplotlib.axes import Axes  # type: ignore

from .laser_output import LaserScanOutput, LaserScanConfig
from basic_map.map_occupancy import OccupancyMap


PathNode = tuple[float, float]


class LaserScannerOcc:
    """Laser scanner that operates on an :class:`~basic_map.map_occupancy.OccupancyMap`.

    The scanner accepts robot states in **real-world coordinates** (metres,
    standard right-hand frame: *x* right, *y* up).  Internally, ray-casting is
    performed on the pixel grid using an efficient DDA (Digital Differential
    Analyser) algorithm, which avoids expensive geometric intersection tests.

    The public interface (``load_map`` / ``load_scanner`` / ``scan`` / ``plot``)
    mirrors :class:`~basic_laser_scanner.laser_scanner_geo.LaserScannerGeo` so
    the two scanners are interchangeable.

    Args:
        config: Scanner configuration dictionary.  Same keys as for
            :class:`~basic_laser_scanner.laser_scanner_geo.LaserScannerGeo`.
    """

    def __init__(self, config: LaserScanConfig) -> None:
        self._cfg = config
        self._map_prepared = False
        self._scanner_prepared = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> list[float]:
        return self._state

    @property
    def position(self) -> tuple[float, float]:
        return (self._state[0], self._state[1])

    @property
    def heading(self) -> float:
        return self._state[2]

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_fpath: str) -> "LaserScannerOcc":
        """Create a scanner from a YAML configuration file."""
        with open(config_fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_map(self, occupancy_map: OccupancyMap) -> None:
        """Attach an occupancy map to this scanner.

        Args:
            occupancy_map: The :class:`~basic_map.map_occupancy.OccupancyMap`
                to scan against.
        """
        if not isinstance(occupancy_map, OccupancyMap):
            raise TypeError(
                f"Expected OccupancyMap, got {type(occupancy_map).__name__}."
            )
        self._map = occupancy_map
        self._map_prepared = True

    def load_scanner(
        self, init_position: PathNode, init_heading: float
    ) -> None:
        """Initialise the scanner state.

        Args:
            init_position: Starting position ``(x, y)`` in world coordinates.
            init_heading: Starting heading in radians.
        """
        self.laser_scan = LaserScanOutput.from_config(self._cfg)
        self._state: list[float] = [
            init_position[0], init_position[1], init_heading
        ]
        self._scanner_prepared = True

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(
        self, current_time: float, current_state: list[float]
    ) -> LaserScanOutput:
        """Perform a laser scan at the given state.

        Args:
            current_time: Timestamp for the scan.
            current_state: ``[x, y, heading]`` in world coordinates / radians.

        Returns:
            :class:`~basic_laser_scanner.laser_output.LaserScanOutput` with
            ranges and beam end-points expressed in world coordinates.
        """
        if not self._map_prepared:
            raise ValueError("Map not prepared.")
        if not self._scanner_prepared:
            raise ValueError("Scanner not prepared.")
        if not isinstance(current_state, list) or len(current_state) != 3:
            raise ValueError("current_state must be a list of [x, y, heading].")

        self._state = current_state
        self.laser_scan.timestamp = current_time
        self.laser_scan.init_beams(self.position, self.heading)

        x, y = self.position
        occ = self._map

        # Convert scanner position to pixel coordinates (floating-point).
        #
        # OccupancyMap uses world coordinates at pixel centres
        #   x = ox + col * res
        #   y = oy + (H - 1 - row) * res
        # while DDA boundary stepping assumes integer coordinates denote
        # *cell edges*. Shift by +0.5 so centres map to half-integers and
        # edges map to integers; this removes a systematic half-cell offset.
        ox, oy = occ.origin
        res = occ.resolution
        c0 = (x - ox) / res + 0.5
        r0 = (occ.height - 1) - (y - oy) / res + 0.5

        # Maximum range in pixels.
        max_range_px = self.laser_scan.range_max / res

        new_ranges = self.laser_scan.ranges.copy()
        new_beams = self.laser_scan.beam_end_points.copy()

        for i, relative_angle in enumerate(self.laser_scan.angles):
            world_angle = self.heading + relative_angle
            # Direction in pixel space (col right, row *downward* ⇒ y-flip).
            dc = math.cos(world_angle)
            dr = -math.sin(world_angle)

            dist_px = self._dda_cast(r0, c0, dr, dc, max_range_px, occ)
            dist_world = dist_px * res

            new_ranges[i] = dist_world
            new_beams[i] = (
                x + dist_world * math.cos(world_angle),
                y + dist_world * math.sin(world_angle),
            )

        self.laser_scan.update_ranges_and_beams(current_time, new_ranges, new_beams)
        return self.laser_scan

    # ------------------------------------------------------------------
    # Internal ray-casting (DDA)
    # ------------------------------------------------------------------

    @staticmethod
    def _dda_cast(
        r0: float,
        c0: float,
        dr: float,
        dc: float,
        max_dist_px: float,
        occ: OccupancyMap,
    ) -> float:
        """Cast a single ray using the DDA algorithm.

        The ray starts at continuous pixel coordinates ``(r0, c0)`` and
        travels in direction ``(dr, dc)`` (both components normalised so
        that ``||(dr, dc)|| == 1`` in pixel space).

        Returns the distance in pixels to the first occupied cell (or
        ``max_dist_px`` if no obstacle is found within range).
        """
        H = occ.height
        W = occ.width
        grid = occ.grid

        # ------------------------------------------------------------------
        # Initialise DDA accumulators.
        # ------------------------------------------------------------------
        # Distance along the ray to the next row/column boundary.
        EPS = 1e-9

        if abs(dc) < EPS:
            t_next_c = math.inf
            dt_c = math.inf
        else:
            dt_c = abs(1.0 / dc)
            if dc > 0:
                t_next_c = (math.floor(c0) + 1 - c0) / dc
            else:
                t_next_c = (c0 - math.ceil(c0) + 1) / (-dc)

        if abs(dr) < EPS:
            t_next_r = math.inf
            dt_r = math.inf
        else:
            dt_r = abs(1.0 / dr)
            if dr > 0:
                t_next_r = (math.floor(r0) + 1 - r0) / dr
            else:
                t_next_r = (r0 - math.ceil(r0) + 1) / (-dr)

        ci = int(c0)
        ri = int(r0)
        step_c = 1 if dc >= 0 else -1
        step_r = 1 if dr >= 0 else -1
        t = 0.0

        # ------------------------------------------------------------------
        # March along the ray.
        # ------------------------------------------------------------------
        while t <= max_dist_px:
            # Clamp indices for the boundary check.
            if ri < 0 or ri >= H or ci < 0 or ci >= W:
                # The ray has left the map — return current distance.
                return min(t, max_dist_px)
            if grid[ri, ci]:
                # Hit an occupied cell.
                return t

            # Advance to the next grid-cell boundary.
            if t_next_c < t_next_r:
                t = t_next_c
                t_next_c += dt_c
                ci += step_c
            else:
                t = t_next_r
                t_next_r += dt_r
                ri += step_r

        return max_dist_px

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax: Axes, with_map: bool = False) -> None:
        """Plot the current scan and optionally the underlying map.

        Args:
            ax: Matplotlib axes to draw on.
            with_map: If ``True``, draw the occupancy map in the background.
        """
        if not self._map_prepared:
            raise ValueError("Map not prepared.")
        if not self._scanner_prepared:
            raise ValueError("Scanner not prepared.")

        if with_map:
            self._map.plot(ax)

        for beam in self.laser_scan.beam_end_points:
            ax.plot(
                [self.position[0], beam[0]],
                [self.position[1], beam[1]],
                "gray",
                linestyle="-",
            )
        ax.plot(*self.position, "ro")
