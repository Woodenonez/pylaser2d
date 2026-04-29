"""Unified laser-scanner entry point.

Selects the appropriate backend automatically based on the type of map
passed to :meth:`LaserScanner.load_map`:

* :class:`~basic_map.map_geometric.GeometricMap` → :class:`~basic_laser_scanner.laser_scanner_geo.LaserScannerGeo`
* :class:`~basic_map.map_occupancy.OccupancyMap`  → :class:`~basic_laser_scanner.laser_scanner_occ.LaserScannerOcc`

Both backends expose the same public interface
(``load_map`` / ``load_scanner`` / ``scan`` / ``plot``), so after
construction the returned :class:`LaserScanner` object can be used
identically regardless of the underlying map type.

Example::

    scanner = LaserScanner.from_yaml("scanner.yaml")

    # -- geometric map --
    geo_map = GeometricMap.from_json("map.json")
    scanner.load_map(geo_map)

    # -- occupancy map --
    occ_map = OccupancyMap.from_image("map.png", resolution=0.05)
    scanner.load_map(occ_map)

    scanner.load_scanner((x, y), heading)
    scan_output = scanner.scan(t, [x, y, heading])
"""
from __future__ import annotations

import yaml  # type: ignore
from typing import Union

from .laser_output import LaserScanOutput, LaserScanConfig
from .laser_scanner_geo import LaserScannerGeo
from .laser_scanner_occ import LaserScannerOcc


# ---------------------------------------------------------------------------
# Public type alias for the supported map objects
# ---------------------------------------------------------------------------
# Import here for type-checking only; the actual isinstance checks below use
# the same classes at runtime.
from basic_map.map_geometric import GeometricMap
from basic_map.map_occupancy import OccupancyMap

SupportedMap = Union[GeometricMap, OccupancyMap]

PathNode = tuple[float, float]


class LaserScanner:
    """Unified laser-scanner entry point.

    Internally delegates all work to either :class:`LaserScannerGeo` or
    :class:`LaserScannerOcc` once a map has been loaded via
    :meth:`load_map`.

    Args:
        config: Scanner configuration dictionary with keys ``angle_min``,
            ``angle_max``, ``angle_increment``, ``range_min``, ``range_max``,
            and ``frame_id``.
    """

    def __init__(self, config: LaserScanConfig) -> None:
        self._cfg = config
        self._backend: LaserScannerGeo | LaserScannerOcc | None = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_fpath: str) -> "LaserScanner":
        """Create a :class:`LaserScanner` from a YAML configuration file.

        Args:
            config_fpath: Path to the YAML file.
        """
        with open(config_fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ------------------------------------------------------------------
    # Map loading (selects the backend)
    # ------------------------------------------------------------------

    def load_map(self, map_obj: SupportedMap) -> None:
        """Attach a map and select the matching scanner backend.

        Args:
            map_obj: Either a :class:`~basic_map.map_geometric.GeometricMap`
                (selects :class:`LaserScannerGeo`) or an
                :class:`~basic_map.map_occupancy.OccupancyMap`
                (selects :class:`LaserScannerOcc`).

        Raises:
            TypeError: If *map_obj* is not one of the supported map types.
        """
        if isinstance(map_obj, GeometricMap):
            backend: LaserScannerGeo | LaserScannerOcc = LaserScannerGeo(self._cfg)
            backend.load_map(*map_obj())
        elif isinstance(map_obj, OccupancyMap):
            backend = LaserScannerOcc(self._cfg)
            backend.load_map(map_obj)
        else:
            raise TypeError(
                f"Unsupported map type: {type(map_obj).__name__}. "
                "Expected GeometricMap or OccupancyMap."
            )
        self._backend = backend

    # ------------------------------------------------------------------
    # Scanner initialisation / scanning
    # ------------------------------------------------------------------

    def load_scanner(self, init_position: PathNode, init_heading: float) -> None:
        """Initialise the scanner position and heading.

        Args:
            init_position: Starting position ``(x, y)`` in world coordinates.
            init_heading: Starting heading in radians.

        Raises:
            ValueError: If :meth:`load_map` has not been called first.
        """
        self._require_backend()
        self._backend.load_scanner(init_position, init_heading)  # type: ignore[union-attr]

    def scan(self, current_time: float, current_state: list[float]) -> LaserScanOutput:
        """Perform a laser scan at the given state.

        Args:
            current_time: Timestamp for the scan.
            current_state: ``[x, y, heading]`` in world coordinates / radians.

        Returns:
            :class:`~basic_laser_scanner.laser_output.LaserScanOutput`.

        Raises:
            ValueError: If :meth:`load_map` or :meth:`load_scanner` has not
                been called.
        """
        self._require_backend()
        return self._backend.scan(current_time, current_state)  # type: ignore[union-attr]

    def plot(self, ax, with_map: bool = False) -> None:
        """Plot the current scan and optionally the map.

        Args:
            ax: Matplotlib :class:`~matplotlib.axes.Axes` to draw on.
            with_map: If ``True``, draw the map in the background.

        Raises:
            ValueError: If :meth:`load_map` or :meth:`load_scanner` has not
                been called.
        """
        self._require_backend()
        self._backend.plot(ax, with_map=with_map)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_backend(self) -> None:
        if self._backend is None:
            raise ValueError("No map loaded. Call load_map() before using the scanner.")

    # ------------------------------------------------------------------
    # Pass-through properties (available after load_map + load_scanner)
    # ------------------------------------------------------------------

    @property
    def state(self) -> list[float]:
        self._require_backend()
        return self._backend.state  # type: ignore[union-attr]

    @property
    def position(self):
        self._require_backend()
        return self._backend.position  # type: ignore[union-attr]

    @property
    def heading(self) -> float:
        self._require_backend()
        return self._backend.heading  # type: ignore[union-attr]

    @property
    def laser_scan(self) -> LaserScanOutput:
        self._require_backend()
        return self._backend.laser_scan  # type: ignore[union-attr]
