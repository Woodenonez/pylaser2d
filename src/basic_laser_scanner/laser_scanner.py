import math
import yaml # type: ignore
import warnings
from typing import TypedDict

import numpy as np
from shapely.geometry import Point, LineString, Polygon # type: ignore
from shapely.geometry.base import BaseGeometry # type: ignore
from matplotlib.axes import Axes # type: ignore

from .laser_output import LaserScanOutput, LaserScanConfig


PathNode = tuple[float, float]


class BasicMap(TypedDict):
    boundary_coords: list[PathNode]
    obstacle_list: list[list[PathNode]]


class LaserScanner:
    def __init__(self, config: LaserScanConfig) -> None:
        """Initializes the laser scanner with the given configuration.

        Args:
            config: The configuration for the laser scanner. It should be a dictionary with the following
                keys: "angle_min", "angle_max", "angle_increment", "range_min", "range_max", and "frame_id".
        """
        self._cfg = config
        self._map_prepared = False
        self._scanner_prepared = False

    @property
    def state(self):
        return self._state
    
    @property
    def position(self):
        return self._state[:2]
    
    @property
    def heading(self):
        return self._state[2]
    
    @classmethod
    def from_yaml(cls, config_fpath: str) -> 'LaserScanner':
        with open(config_fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def load_map(self, boundary_coords: list[PathNode], obstacle_list: list[list[PathNode]]) -> None:
        self._map = BasicMap(boundary_coords=boundary_coords, obstacle_list=obstacle_list)
        self._map_prepared = True

    def load_scanner(self, init_position: PathNode, init_heading: float) -> None:
        """Initialize the laser scanner.

        Args:
            init_position: The initial position of the robot, (x, y).
            init_heading: The initial heading of the robot, in radian.
        """
        self.laser_scan = LaserScanOutput.from_config(self._cfg)
        self._state = [init_position[0], init_position[1], init_heading]
        self._scanner_prepared = True

    def scan(self, current_time: float, current_state: list[float]) -> LaserScanOutput:
        """Scan the environment with the laser scanner.

        Args:
            current_time: The current time.
            current_state: The current state of the robot. It should be [x, y, heading].

        Raises:
            ValueError: If the map or scanner is not prepared.
            ValueError: If the current state is invalid.

        Returns:
            LaserScanOutput: The laser scan output.
        """
        if not self._map_prepared:
            raise ValueError("Map not prepared.")
        if not self._scanner_prepared:
            raise ValueError("Scanner not prepared.")
        if not isinstance(current_state, list) or len(current_state)!=3:
            raise ValueError("Invalid current state.")
        
        self._state = current_state
        self.laser_scan.timestamp = current_time
        self.laser_scan.init_beams(self.position, self.heading)

        if not Polygon(self._map['boundary_coords']).contains(Point(current_state[0], current_state[1])):
            warnings.warn("Scanner is outside the boundary!")
            return self.laser_scan

        _x, _y = self.position
        new_ranges = self.laser_scan.ranges.copy()
        new_beams = self.laser_scan.beam_end_points.copy()
        geometries = [Polygon(obstacle) for obstacle in self._map['obstacle_list']]
        geometries.append(LineString(self._map['boundary_coords'] + [self._map['boundary_coords'][0]]))
        for i, relative_angle in enumerate(self.laser_scan.angles):
            angle = self.heading + relative_angle
            beam = LineString([(_x, _y), (_x+self.laser_scan.range_max*math.cos(angle), _y+self.laser_scan.range_max*math.sin(angle))])
            closest_distance = self.laser_scan.range_max
            for geo in geometries:
                if not geo.is_valid:
                    geo = geo.buffer(0)
                if beam.intersects(geo):
                    intersection:BaseGeometry = beam.intersection(geo)
                    if not intersection.is_empty:
                        distance = intersection.distance(Point(_x, _y))
                        if distance < closest_distance:
                            closest_distance = distance
            new_ranges[i] = closest_distance
            new_beams[i] = (_x+closest_distance*math.cos(angle), _y+closest_distance*math.sin(angle))

        self.laser_scan.update_ranges_and_beams(current_time, new_ranges, new_beams)
        return self.laser_scan
    
    def plot(self, ax: Axes, with_map=False):
        if not self._map_prepared:
            raise ValueError("Map not prepared.")
        if not self._scanner_prepared:
            raise ValueError("Scanner not prepared.")
        
        if with_map:
            boundary_plot = np.array(self._map['boundary_coords']+[self._map['boundary_coords'][0]])
            ax.plot(boundary_plot[:,0], boundary_plot[:,1], 'b-')
            for obstacle in self._map['obstacle_list']:
                obstacle_plot = np.array(obstacle+[obstacle[0]])
                ax.plot(obstacle_plot[:,0], obstacle_plot[:,1], 'r-')
            ax.set_xlim(min(np.array(self._map['boundary_coords'])[:,0])-1, max(np.array(self._map['boundary_coords'])[:,0])+1)
            ax.set_ylim(min(np.array(self._map['boundary_coords'])[:,1])-1, max(np.array(self._map['boundary_coords'])[:,1])+1)


        for beam in self.laser_scan.beam_end_points:
            ax.plot([self.position[0], beam[0]], [self.position[1], beam[1]], 'gray', linestyle='-')
        ax.plot(*self.position, 'ro')
        
        
        