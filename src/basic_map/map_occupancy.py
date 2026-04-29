import numpy as np
import matplotlib.image as mpimg  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from matplotlib.axes import Axes  # type: ignore
from typing import Optional


class OccupancyMap:
    """An occupancy map that can be loaded from an image file or a numpy matrix.

    The internal grid is a 2-D boolean array with shape ``(height, width)``,
    where ``True`` means *occupied* and ``False`` means *free*.

    The map supports an optional real-world coordinate transform defined by
    two parameters:

    * ``resolution`` – metres per pixel (default ``1.0``).
    * ``origin`` – ``(ox, oy)`` real-world coordinates (metres) of the
      **bottom-left** pixel (row = ``height - 1``, col = ``0``).
      Defaults to ``(0.0, 0.0)``.

    With these parameters the conversion between pixel indices and world
    coordinates follows the standard map convention used in robotics (e.g. ROS
    `nav_msgs/OccupancyGrid`):

    .. code-block:: text

        col  =  (x - ox) / resolution
        row  =  (height - 1) - (y - oy) / resolution

        x    =  ox + col * resolution
        y    =  oy + (height - 1 - row) * resolution
    """

    def __init__(
        self,
        grid: np.ndarray,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Initialise directly from a boolean (or binary) numpy array.

        Args:
            grid: 2-D array of shape ``(H, W)``.  Non-zero values are treated
                as *occupied*.
            resolution: Metres per pixel.
            origin: Real-world ``(x, y)`` of the bottom-left pixel.
        """
        if grid.ndim != 2:
            raise ValueError(f"grid must be 2-D, got shape {grid.shape}.")
        if resolution <= 0.0:
            raise ValueError(f"resolution must be positive, got {resolution}.")
        self._grid: np.ndarray = grid.astype(bool)
        self._resolution = float(resolution)
        self._origin = (float(origin[0]), float(origin[1]))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Boolean occupancy grid, shape ``(height, width)``."""
        return self._grid

    @property
    def height(self) -> int:
        return self._grid.shape[0]

    @property
    def width(self) -> int:
        return self._grid.shape[1]

    @property
    def resolution(self) -> float:
        """Metres per pixel."""
        return self._resolution

    @property
    def origin(self) -> tuple[float, float]:
        """Real-world ``(x, y)`` of the bottom-left pixel."""
        return self._origin

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_image(
        cls,
        image_path: str,
        threshold: float = 0.5,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
        occupied_dark: bool = True,
    ) -> "OccupancyMap":
        """Load an occupancy map from an image file.

        Supported formats are those handled by ``matplotlib.image.imread``
        (PNG, JPEG, …).  The image is converted to a single-channel
        grayscale value in ``[0, 1]`` before applying the threshold.

        Args:
            image_path: Path to the image file.
            threshold: Normalised grayscale threshold in ``[0, 1]``.
                Pixels *below* this value are occupied when
                ``occupied_dark=True`` (default), and pixels *above* when
                ``occupied_dark=False``.
            resolution: Metres per pixel.
            origin: Real-world ``(x, y)`` of the bottom-left pixel.
            occupied_dark: If ``True`` (default), dark pixels represent
                obstacles (value < threshold).  Set to ``False`` to invert
                (light pixels are obstacles).
        """
        raw = mpimg.imread(image_path)
        # Normalise to [0, 1] float
        if raw.dtype == np.uint8:
            raw = raw.astype(np.float32) / 255.0
        # Convert to greyscale if RGB / RGBA
        if raw.ndim == 3:
            raw = raw[..., :3].mean(axis=-1)
        if occupied_dark:
            grid = raw < threshold
        else:
            grid = raw >= threshold
        return cls(grid, resolution=resolution, origin=origin)

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "OccupancyMap":
        """Load an occupancy map from a numpy array.

        Args:
            array: 2-D array of shape ``(H, W)``.  Non-zero → occupied.
            resolution: Metres per pixel.
            origin: Real-world ``(x, y)`` of the bottom-left pixel.
        """
        return cls(array, resolution=resolution, origin=origin)

    @classmethod
    def from_npy(
        cls,
        npy_path: str,
        resolution: float = 1.0,
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> "OccupancyMap":
        """Load an occupancy map from a ``.npy`` file (saved numpy matrix).

        Args:
            npy_path: Path to a ``.npy`` file containing a 2-D boolean/int
                array.
            resolution: Metres per pixel.
            origin: Real-world ``(x, y)`` of the bottom-left pixel.
        """
        array = np.load(npy_path)
        return cls(array, resolution=resolution, origin=origin)

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Convert real-world ``(x, y)`` to pixel ``(row, col)``.

        Returns:
            ``(row, col)`` as integers (rounded to nearest pixel).
        """
        ox, oy = self._origin
        col = int(round((x - ox) / self._resolution))
        row = int(round((self.height - 1) - (y - oy) / self._resolution))
        return row, col

    def pixel_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Convert pixel ``(row, col)`` to real-world ``(x, y)``."""
        ox, oy = self._origin
        x = ox + col * self._resolution
        y = oy + (self.height - 1 - row) * self._resolution
        return x, y

    def is_occupied(self, row: int, col: int) -> bool:
        """Return ``True`` if the pixel at ``(row, col)`` is occupied."""
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        return bool(self._grid[row, col])

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Return ``True`` if ``(row, col)`` is inside the map."""
        return 0 <= row < self.height and 0 <= col < self.width

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot(self, ax: Axes, **imshow_kwargs) -> None:
        """Plot the occupancy map on the given axes.

        Occupied cells are shown in black and free cells in white.  The
        world-coordinate extents are used for axis limits.
        """
        ox, oy = self._origin
        res = self._resolution
        extent = [
            ox,
            ox + self.width * res,
            oy,
            oy + self.height * res,
        ]
        defaults = {"cmap": "gray_r", "vmin": 0, "vmax": 1, "origin": "lower"}
        defaults.update(imshow_kwargs)
        ax.imshow(self._grid.astype(np.uint8), extent=extent, **defaults)


if __name__ == "__main__":
    # Quick smoke-test: build a simple map programmatically.
    H, W = 200, 200
    grid = np.zeros((H, W), dtype=bool)
    # Border walls
    grid[0, :] = True
    grid[-1, :] = True
    grid[:, 0] = True
    grid[:, -1] = True
    # A rectangular obstacle
    grid[60:100, 80:130] = True

    occ_map = OccupancyMap.from_numpy(grid, resolution=0.05, origin=(0.0, 0.0))
    fig, ax = plt.subplots()
    occ_map.plot(ax)
    ax.set_title("OccupancyMap smoke-test")
    plt.show()
