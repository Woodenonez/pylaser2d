"""Demo: Robot moving around an occupancy map with laser scanning visualization.

This demo simulates a robot navigating through an occupancy map while performing
laser scans. The visualization shows:
- The occupancy grid (occupied cells in dark, free in light)
- The robot position and heading (red arrow)
- Laser scan beams (colored by range)
- Beam endpoints
"""

import math
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from basic_laser_scanner.laser_scanner_occ import LaserScannerOcc
from basic_map.map_occupancy import OccupancyMap


ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "dense_scanner_spec.yaml")


def create_simple_occupancy_map(size=100, resolution=0.1):
    """Create a simple occupancy map with walls and an internal obstacle.
    
    Args:
        size: Grid size (size x size pixels)
        resolution: Metres per pixel
    
    Returns:
        OccupancyMap object
    """
    grid = np.zeros((size, size), dtype=bool)
    
    # Create walls on all four edges (2 pixels thick)
    grid[0:2, :] = True
    grid[-2:, :] = True
    grid[:, 0:2] = True
    grid[:, -2:] = True
    
    # Add an internal obstacle (L-shaped)
    grid[40:60, 30:35] = True  # Vertical part
    grid[55:60, 30:50] = True  # Horizontal part
    
    # Add another obstacle
    grid[20:30, 60:70] = True
    
    return OccupancyMap.from_numpy(grid, resolution=resolution, origin=(0.0, 0.0))


def simulate_robot_path_occupancy(occ_map, num_points=100):
    """Simulate robot moving in a circular path inside the occupancy map."""
    # Get the map bounds
    width_m = occ_map.width * occ_map.resolution
    height_m = occ_map.height * occ_map.resolution
    
    # Center and radius for circular motion
    center_x = width_m / 2
    center_y = height_m / 2
    radius = min(width_m, height_m) / 4
    
    # Create circular path
    t_vals = np.linspace(0, 2 * np.pi, num_points)
    x_vals = center_x + radius * np.cos(t_vals)
    y_vals = center_y + radius * np.sin(t_vals)
    
    waypoints = list(zip(x_vals, y_vals))
    return waypoints


def visualize_laser_scan_occ(ax, occ_map, laser_scan, robot_pos, robot_heading):
    """Visualize a single laser scan on an occupancy map."""
    ax.clear()
    
    # Create custom colormap: free=light gray, occupied=dark gray
    colors = ['lightgray', 'darkgray']
    cmap = ListedColormap(colors)
    
    # Display the occupancy grid (flip Y to match world coordinates)
    grid_display = np.flip(occ_map.grid, axis=0)  # Flip for matplotlib display
    ax.imshow(grid_display, cmap=cmap, extent=[0, occ_map.width * occ_map.resolution,
                                                0, occ_map.height * occ_map.resolution],
             origin='lower', alpha=0.8)
    
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
    circle = plt.Circle(robot_pos, 0.1, color='red', alpha=0.9, label='Robot')
    ax.add_patch(circle)
    
    # Draw heading arrow
    arrow_length = 0.2
    arrow_end_x = robot_pos[0] + arrow_length * np.cos(robot_heading)
    arrow_end_y = robot_pos[1] + arrow_length * np.sin(robot_heading)
    arrow = FancyArrowPatch(robot_pos, (arrow_end_x, arrow_end_y),
                           arrowstyle='->', mutation_scale=15, color='red', linewidth=2)
    ax.add_patch(arrow)
    
    # Set axis properties
    width_m = occ_map.width * occ_map.resolution
    height_m = occ_map.height * occ_map.resolution
    ax.set_xlim(0, width_m)
    ax.set_ylim(0, height_m)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='upper right')
    ax.set_title(f'Occupancy Map - Laser Scan Demo\nRobot at ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), Heading: {np.degrees(robot_heading):.1f}°')


def main():
    """Run the occupancy map laser scan demo."""
    print("Creating occupancy map...")
    occ_map = create_simple_occupancy_map(size=100, resolution=0.1)
    
    print("Creating laser scanner...")
    scanner = LaserScannerOcc.from_yaml(CONFIG_PATH)
    scanner.load_map(occ_map)
    
    # Initialize scanner at a starting position
    start_pos = (2.5, 2.5)
    start_heading = 0.0
    scanner.load_scanner(start_pos, start_heading)
    
    # Generate robot path
    print("Generating robot path...")
    waypoints = simulate_robot_path_occupancy(occ_map, num_points=150)
    
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
        visualize_laser_scan_occ(ax, occ_map, laser_scan, waypoint, heading)
    
    # Create animation
    print("Creating animation (showing every 3rd frame for performance)...")
    anim = FuncAnimation(fig, update_frame, frames=range(0, len(waypoints), 3),
                        repeat=True, repeat_delay=1000, blit=False)
    
    print("Displaying animation...")
    print("Press Ctrl+C to stop.")
    plt.show()


if __name__ == "__main__":
    main()
