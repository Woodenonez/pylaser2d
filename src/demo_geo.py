"""Demo: Robot moving around a geometric map with laser scanning visualization.

This demo simulates a robot navigating through a geometric map while performing
laser scans. The visualization shows:
- The map boundary and obstacles
- The robot position and heading (red arrow)
- Laser scan beams (colored by range)
- Beam endpoints
"""

import math
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation

from basic_laser_scanner.laser_scanner_geo import LaserScannerGeo
from basic_map.map_geometric import GeometricMap


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "dense_scanner_spec.yaml")
MAP_PATH = os.path.join(ROOT_DIR, "data", "test_map_1", "map.json")


def create_robot_path(start, end, num_points=50):
    """Create a smooth path from start to end position."""
    path = np.linspace(start, end, num_points)
    return path


def simulate_robot_path(map_obj, scanner):
    """Simulate robot moving in a circular/figure-8 pattern around the map."""
    boundary = np.array(map_obj.boundary_coords)
    
    # Find reasonable waypoints inside the boundary
    center_x = np.mean(boundary[:, 0])
    center_y = np.mean(boundary[:, 1])
    
    # Create waypoints for figure-8 movement
    radius = 1.5
    t_vals = np.linspace(0, 2 * np.pi, 100)
    x_vals = center_x + radius * np.cos(t_vals)
    y_vals = center_y + radius * np.sin(t_vals)
    
    waypoints = list(zip(x_vals, y_vals))
    return waypoints


def visualize_laser_scan(ax, map_obj, laser_scan, robot_pos, robot_heading):
    """Visualize a single laser scan on the given axes."""
    ax.clear()
    
    # Draw map boundary
    boundary = np.array(map_obj.boundary_coords + [map_obj.boundary_coords[0]])
    ax.plot(boundary[:, 0], boundary[:, 1], 'k-', linewidth=2, label='Boundary')
    
    # Draw obstacles
    for i, obstacle in enumerate(map_obj.obstacle_coords_list):
        obs_array = np.array(obstacle + [obstacle[0]])
        ax.fill(obs_array[:, 0], obs_array[:, 1], color='gray', alpha=0.5)
        ax.plot(obs_array[:, 0], obs_array[:, 1], 'k-', linewidth=1)
    
    # Draw laser beams with color gradient based on range
    ranges = laser_scan.ranges
    range_max = laser_scan.range_max
    beam_endpoints = laser_scan.beam_end_points
    
    for i, (endpoint, r) in enumerate(zip(beam_endpoints, ranges)):
        # Color based on range (normalized to [0, 1])
        color_val = r / range_max
        # Color from blue (near) to red (far)
        color = plt.cm.cool(1 - color_val)
        
        ax.plot([robot_pos[0], endpoint[0]], [robot_pos[1], endpoint[1]], 
               color=color, alpha=0.3, linewidth=0.5)
        ax.plot(endpoint[0], endpoint[1], 'o', color=color, markersize=2)
    
    # Draw robot as a circle with heading arrow
    circle = plt.Circle(robot_pos, 0.15, color='red', alpha=0.7, label='Robot')
    ax.add_patch(circle)
    
    # Draw heading arrow
    arrow_length = 0.3
    arrow_end_x = robot_pos[0] + arrow_length * np.cos(robot_heading)
    arrow_end_y = robot_pos[1] + arrow_length * np.sin(robot_heading)
    arrow = FancyArrowPatch(robot_pos, (arrow_end_x, arrow_end_y),
                           arrowstyle='->', mutation_scale=20, color='red', linewidth=2)
    ax.add_patch(arrow)
    
    # Set axis properties
    boundary_array = np.array(map_obj.boundary_coords)
    margin = 0.5
    ax.set_xlim(boundary_array[:, 0].min() - margin, boundary_array[:, 0].max() + margin)
    ax.set_ylim(boundary_array[:, 1].min() - margin, boundary_array[:, 1].max() + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.set_title(f'Geometric Map - Laser Scan Demo\nRobot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Heading: {np.degrees(robot_heading):.1f}°')


def main():
    """Run the geometric map laser scan demo."""
    print("Loading geometric map...")
    map_obj = GeometricMap.from_json(MAP_PATH)
    
    print("Creating laser scanner...")
    scanner = LaserScannerGeo.from_yaml(CONFIG_PATH)
    boundary_coords, obstacle_coords = map_obj()
    scanner.load_map(boundary_coords, obstacle_coords)
    
    # Initialize scanner at a starting position
    start_pos = (1.0, 1.0)
    start_heading = 0.0
    scanner.load_scanner(start_pos, start_heading)
    
    # Generate robot path
    print("Generating robot path...")
    waypoints = simulate_robot_path(map_obj, scanner)
    
    # Create figure for visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update_frame(frame_idx):
        """Update function for animation."""
        if frame_idx >= len(waypoints):
            return
        
        # Get current waypoint
        waypoint = waypoints[frame_idx]
        
        # Calculate heading towards next waypoint
        if frame_idx < len(waypoints) - 1:
            next_point = waypoints[frame_idx + 1]
            heading = np.arctan2(next_point[1] - waypoint[1], 
                               next_point[0] - waypoint[0])
        else:
            heading = 0.0
        
        # Perform scan
        state = [waypoint[0], waypoint[1], heading]
        laser_scan = scanner.scan(float(frame_idx) * 0.1, state)
        
        # Visualize
        visualize_laser_scan(ax, map_obj, laser_scan, waypoint, heading)
    
    # Create animation
    print("Creating animation (showing every 5th frame for performance)...")
    anim = FuncAnimation(fig, update_frame, frames=range(0, len(waypoints), 5),
                        repeat=True, repeat_delay=1000, blit=False)
    
    print("Displaying animation...")
    print("Press Ctrl+C to stop.")
    plt.show()


if __name__ == "__main__":
    main()
