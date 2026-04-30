from dataclasses import dataclass, field
import math
import warnings
from typing import Any, TypedDict

import numpy as np


class LaserScanConfig(TypedDict):
    angle_min: float
    angle_max: float
    angle_increment: float
    range_min: float
    range_max: float
    frame_id: str


@dataclass
class LaserScanOutput:
    """
    Data class for the output of the laser scanner (in 2D).
    Similar to the sensor_msgs/LaserScan message in ROS.
    """
    timestamp: float = 0.0
    frame_id: str = ""
    _state: list[float] = field(default_factory=list)

    angle_min: float = -math.pi/2
    angle_max: float = math.pi/2
    angle_increment: float = math.pi/180
    _angles: tuple[float, ...] = field(default_factory=tuple)

    range_min: float = 0.0
    range_max: float = 10.0
    _ranges: list[float] = field(default_factory=list)
    _beam_end_points: list[tuple[float, float]] = field(default_factory=list)

    __fronzen__ = False

    @property
    def angles(self) -> tuple[float, ...]:
        return self._angles

    @property
    def angles_deg(self) -> tuple[float, ...]:
        return tuple([round(math.degrees(angle), 4) for angle in self.angles])
    
    @property
    def state(self) -> list[float]:
        return self._state
    
    @property
    def ranges(self) -> list[float]:
        return self._ranges
    
    @property
    def beam_end_points(self) -> list[tuple[float, float]]:
        return self._beam_end_points

    def __post_init__(self):
        if self.angle_min > self.angle_max:
            raise ValueError("angle_min should be less than angle_max.")
        if (self.angle_increment <= 0.0) or (self.angle_increment > self.angle_max - self.angle_min):
            raise ValueError("angle_increment should be positive and less than angle_max - angle_min.")
        if (self.angle_min >= 0.0) or (self.angle_max <= 0.0):
            warnings.warn("Assumeing the heading direction is 0.0 rad, angle_min is expected to be negative and angle_max is expected to be positive.")

        self._angles = tuple(np.arange(self.angle_min, self.angle_max+self.angle_increment/2, self.angle_increment).tolist())
        self.__fronzen__ = True

    def __setattr__(self, __name: str, __value: Any) -> None:
        if self.__fronzen__ and (not hasattr(self, __name)):
            raise AttributeError(f"Cannot set attribute {__name} of {self.__class__.__name__}")
        super().__setattr__(__name, __value)

    @classmethod
    def from_config(cls, config: LaserScanConfig) -> 'LaserScanOutput':
        return cls(angle_min=config['angle_min'], 
                   angle_max=config['angle_max'], 
                   angle_increment=config['angle_increment'], 
                   range_min=config['range_min'], 
                   range_max=config['range_max'], 
                   frame_id=config['frame_id'])

    def init_beams(self, current_position: tuple[float, float], heading: float):
        """Initialize the beams of the laser scanner.
        """
        self._state = [current_position[0], current_position[1], heading]
        self._beam_end_points = []
        self._ranges = [self.range_max] * len(self.angles)
        for angle in self.angles:
            x = current_position[0] + self.range_max * math.cos(heading + angle)
            y = current_position[1] + self.range_max * math.sin(heading + angle)
            self.beam_end_points.append((x, y))

    def update_ranges_and_beams(self, current_time: float, new_ranges: list[float], new_beam_end_points: list[tuple[float, float]]):
        """Update the ranges and beam end points of the laser scanner.

        Args:
            current_time: The current time.
            new_ranges: The new ranges. The length should be the same as the original ranges.
            new_beam_end_points: The new beam end points. The length should be the same as the original beam end points.
        """
        if len(new_ranges) != len(self._ranges):
            raise ValueError(f"The length of new_ranges should be {len(self._ranges)}, got {len(new_ranges)}.")
        if len(new_beam_end_points) != len(self._beam_end_points):
            raise ValueError(f"The length of new_beam_end_points should be {len(self._beam_end_points)}, got {len(new_beam_end_points)}.")
        self.timestamp = current_time
        self._ranges = new_ranges
        self._beam_end_points = new_beam_end_points


if __name__ == "__main__":
    # Test the class
    import matplotlib.pyplot as plt # type: ignore

    laser_scan = LaserScanOutput(angle_increment=math.pi/30)
    print(laser_scan.angles_deg)
    print(laser_scan.ranges)

    laser_scan.init_beams((1.0, 1.0), 0.0)

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.plot(1.0, 1.0, 'ro')
    for beam in laser_scan.beam_end_points:
        ax.plot([1.0, beam[0]], [1.0, beam[1]], 'k-')
    plt.show()


