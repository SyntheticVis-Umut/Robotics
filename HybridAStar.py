"""
Maze Game with Hybrid A* Pathfinding Algorithm

This file implements a maze game where a creature uses the Hybrid A* algorithm 
to find the shortest path from start to exit.

Hybrid A* is designed for vehicles with turning constraints (like cars), combining
discrete graph search with continuous motion planning.

Features:
- Hybrid A* pathfinding with vehicle dynamics
- PNG image maze loading
- Real-time visualization of path exploration
- Creature animation following the optimal path with orientation
"""

import sys
import os
import math
import glob
import heapq
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree
import cv2

# Add the pythonrobotics path to import Hybrid A* algorithm
sys.path.insert(0, '/Users/umutozdemir/Desktop/pythonrobotics')
from PathPlanning.HybridAStar import hybrid_a_star
from PathPlanning.HybridAStar.car import check_car_collision, BUBBLE_R, MAX_STEER, WB
from PathPlanning.HybridAStar.dynamic_programming_heuristic import calc_distance_heuristic

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    # Check if GPU is actually usable
    try:
        _ = cp.array([1, 2, 3])
        GPU_USABLE = True
        print("[GPU] ✓ GPU acceleration available and working (CuPy)")
        print(f"[GPU] Device: {cp.cuda.Device().id if hasattr(cp.cuda, 'Device') else 'Unknown'}")
    except Exception as e:
        GPU_USABLE = False
        print(f"[GPU] ✗ GPU available but not usable: {e}")
        print("[GPU] Falling back to CPU")
except ImportError:
    GPU_AVAILABLE = False
    GPU_USABLE = False
    print("[GPU] ✗ GPU acceleration not available")
    print("[GPU] Install cupy for GPU support: pip install cupy-cuda11x")

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("[Numba] ✓ JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    print("[Numba] ✗ JIT compilation not available")
    print("[Numba] Install numba for speedup: pip install numba")

# Maze configuration
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
CELL_SIZE = 1.0
ROBOT_RADIUS = 0.3

# Hybrid A* specific parameters
# Scale factor to make maze large enough for vehicle (vehicle needs ~2.4 units radius)
MAZE_SCALE = 5.0  # Scale maze coordinates by this factor
XY_GRID_RESOLUTION = 2.0  # Grid resolution for Hybrid A*
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # Yaw angle resolution (15 degrees)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "path_config.json")


def load_planner_config():
    """Load planner configuration from JSON; fall back to defaults if missing/invalid."""
    defaults = {
        "edge_detection": "none",
        "image": "map.png",
    }
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in defaults:
            if k in data:
                defaults[k] = data[k]
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[Config] Warning: failed to load path_config.json: {e}")
    return defaults


class Maze:
    """Simple maze with walls and open spaces"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = set()
        # Start at center by default
        self.start = (width / 2.0, height / 2.0)
        self.exit = (width - 1.5, height - 1.5)  # Slightly offset from walls
        
    def add_wall(self, x, y):
        """Add a wall at position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.walls.add((x, y))
    
    def add_wall_line(self, start_x, start_y, length, horizontal=True):
        """Add a line of walls"""
        for i in range(length):
            if horizontal:
                self.add_wall(start_x + i, start_y)
            else:
                self.add_wall(start_x, start_y + i)
    
    def is_wall(self, x, y):
        """Check if position (x, y) is a wall"""
        return (x, y) in self.walls
    
    def get_obstacle_list(self, scale=1.0):
        """Convert walls to obstacle lists for Hybrid A* planner"""
        ox, oy = [], []
        
        # Add border walls - create a solid border (scaled)
        # Top and bottom borders
        for i in range(self.width):
            ox.append(float(i) * scale)
            oy.append(0.0)
            ox.append(float(i) * scale)
            oy.append(float(self.height - 1) * scale)
        # Left and right borders
        for i in range(self.height):
            ox.append(0.0)
            oy.append(float(i) * scale)
            ox.append(float(self.width - 1) * scale)
            oy.append(float(i) * scale)
        
        # Add internal walls (scaled)
        for wall_x, wall_y in self.walls:
            # Only add if not on border (to avoid duplicates)
            if 0 < wall_x < self.width - 1 and 0 < wall_y < self.height - 1:
                ox.append(float(wall_x) * scale)
                oy.append(float(wall_y) * scale)
        
        return ox, oy
    
    @staticmethod
    def from_image(image_path, start_pos=None, exit_pos=None):
        """
        Create a maze from a PNG image
        Black pixels (or dark pixels) = walls
        White pixels (or light pixels) = open paths
        
        Args:
            image_path: Path to PNG image
            start_pos: Tuple (x, y) for start position, or None for auto-detection
            exit_pos: Tuple (x, y) for exit position, or None for auto-detection
        
        Returns:
            Maze object
        """
        # Load image
        img = Image.open(image_path)
        # Convert to grayscale
        img_gray = img.convert('L')
        # Convert to numpy array
        img_array = np.array(img_gray)
        
        height, width = img_array.shape
        
        # Create maze
        maze = Maze(width, height)
        
        # Threshold: pixels darker than 128 are considered walls
        threshold = 128
        
        # Find walls
        for y in range(height):
            for x in range(width):
                pixel_value = img_array[y, x]
                if pixel_value < threshold:
                    maze.add_wall(x, y)
        
        # Auto-detect start and exit if not provided
        if start_pos is None:
            # Place start at the center of the maze
            center_x = width // 2
            center_y = height // 2
            
            # Check if center is a valid (non-wall) position
            if center_x < width and center_y < height and img_array[center_y, center_x] >= threshold:
                maze.start = (float(center_x), float(center_y))
            else:
                # Center is a wall, find nearest valid position
                found_start = False
                max_search_radius = min(width, height) // 2
                
                for radius in range(1, max_search_radius + 1):
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            if abs(dx) == radius or abs(dy) == radius:
                                x = center_x + dx
                                y = center_y + dy
                                
                                if 0 <= x < width and 0 <= y < height:
                                    if img_array[y, x] >= threshold:
                                        maze.start = (float(x), float(y))
                                        found_start = True
                                        break
                        if found_start:
                            break
                    if found_start:
                        break
                
                if not found_start:
                    for y in range(height):
                        for x in range(width):
                            if img_array[y, x] >= threshold:
                                maze.start = (float(x), float(y))
                                found_start = True
                                break
                        if found_start:
                            break
                    if not found_start:
                        maze.start = (1.0, 1.0)
        else:
            maze.start = start_pos
        
        if exit_pos is None:
            # Find last white pixel from bottom-right (scanning right to left, bottom to top)
            found_exit = False
            for y in range(height - 1, -1, -1):
                for x in range(width - 1, -1, -1):
                    if img_array[y, x] >= threshold:
                        maze.exit = (float(x), float(y))
                        found_exit = True
                        break
                if found_exit:
                    break
            if not found_exit:
                maze.exit = (float(width - 1), float(height - 1))
        else:
            maze.exit = exit_pos
        
        return maze


def create_maze():
    """Create a simple maze with guaranteed path from start to exit"""
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    # Set start to center
    maze.start = (MAZE_WIDTH / 2.0, MAZE_HEIGHT / 2.0)
    
    # Create a simpler maze with a clear winding path
    # Horizontal walls with gaps
    maze.add_wall_line(3, 2, 3, horizontal=True)
    maze.add_wall_line(8, 2, 4, horizontal=True)
    maze.add_wall_line(2, 4, 3, horizontal=True)
    maze.add_wall_line(7, 4, 3, horizontal=True)
    maze.add_wall_line(12, 4, 3, horizontal=True)
    maze.add_wall_line(4, 6, 3, horizontal=True)
    maze.add_wall_line(9, 6, 3, horizontal=True)
    maze.add_wall_line(3, 8, 3, horizontal=True)
    maze.add_wall_line(8, 8, 3, horizontal=True)
    maze.add_wall_line(13, 8, 3, horizontal=True)
    maze.add_wall_line(2, 10, 3, horizontal=True)
    maze.add_wall_line(7, 10, 3, horizontal=True)
    maze.add_wall_line(12, 10, 3, horizontal=True)
    maze.add_wall_line(4, 12, 3, horizontal=True)
    maze.add_wall_line(9, 12, 3, horizontal=True)
    maze.add_wall_line(14, 12, 3, horizontal=True)
    maze.add_wall_line(3, 14, 3, horizontal=True)
    maze.add_wall_line(8, 14, 3, horizontal=True)
    maze.add_wall_line(13, 14, 3, horizontal=True)
    maze.add_wall_line(2, 16, 3, horizontal=True)
    maze.add_wall_line(7, 16, 3, horizontal=True)
    maze.add_wall_line(12, 16, 3, horizontal=True)
    
    # Vertical walls with gaps
    maze.add_wall_line(5, 1, 2, horizontal=False)
    maze.add_wall_line(10, 1, 2, horizontal=False)
    maze.add_wall_line(15, 1, 2, horizontal=False)
    maze.add_wall_line(6, 3, 2, horizontal=False)
    maze.add_wall_line(11, 3, 2, horizontal=False)
    maze.add_wall_line(16, 3, 2, horizontal=False)
    maze.add_wall_line(7, 5, 2, horizontal=False)
    maze.add_wall_line(12, 5, 2, horizontal=False)
    maze.add_wall_line(17, 5, 2, horizontal=False)
    maze.add_wall_line(6, 7, 2, horizontal=False)
    maze.add_wall_line(11, 7, 2, horizontal=False)
    maze.add_wall_line(16, 7, 2, horizontal=False)
    maze.add_wall_line(5, 9, 2, horizontal=False)
    maze.add_wall_line(10, 9, 2, horizontal=False)
    maze.add_wall_line(15, 9, 2, horizontal=False)
    maze.add_wall_line(8, 11, 2, horizontal=False)
    maze.add_wall_line(13, 11, 2, horizontal=False)
    maze.add_wall_line(18, 11, 2, horizontal=False)
    maze.add_wall_line(7, 13, 2, horizontal=False)
    maze.add_wall_line(12, 13, 2, horizontal=False)
    maze.add_wall_line(17, 13, 2, horizontal=False)
    maze.add_wall_line(6, 15, 2, horizontal=False)
    maze.add_wall_line(11, 15, 2, horizontal=False)
    maze.add_wall_line(16, 15, 2, horizontal=False)
    
    return maze


class HybridAStarPlannerWithTracking:
    """Hybrid A* planner with exploration tracking for visualization"""
    
    def __init__(self, ox, oy, resolution, yaw_resolution):
        self.ox = ox
        self.oy = oy
        self.resolution = resolution
        self.yaw_resolution = yaw_resolution
        self.explored_nodes = []
        self.open_set_history = []
        
        # Temporarily disable animation in hybrid_a_star module
        original_show = hybrid_a_star.show_animation
        hybrid_a_star.show_animation = False
        
        self.explored_nodes = []
        self.open_set_history = []
    
    def planning(self, sx, sy, syaw, gx, gy, gyaw):
        """Hybrid A* path search with exploration tracking"""
        print(f"[Hybrid A*] Starting pathfinding from ({sx:.1f}, {sy:.1f}) to ({gx:.1f}, {gy:.1f})")
        if NUMBA_AVAILABLE:
            print("[Hybrid A*] Using CPU pathfinding with Numba JIT compilation")
        else:
            print("[Hybrid A*] Using optimized CPU pathfinding")
        
        start = [sx, sy, syaw]
        goal = [gx, gy, gyaw]
        
        # Call the original Hybrid A* planning function
        path = hybrid_a_star.hybrid_a_star_planning(
            start, goal, self.ox, self.oy, 
            self.resolution, self.yaw_resolution
        )
        
        if path and path.x_list:
            print(f"[Hybrid A*] ✓ Goal found! Path length: {len(path.x_list)} points")
            print(f"[Hybrid A*] Path cost: {path.cost:.2f}")
            return path.x_list, path.y_list, path.yaw_list
        else:
            print("[Hybrid A*] ✗ No path found")
            return [], [], []


class Creature:
    """Creature that navigates the maze using Hybrid A*"""
    
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.path_yaw = []  # Orientation angles
        self.path_index = 0
        self.found_path = False
        
    def find_path(self):
        """Use Hybrid A* to find path from start to exit"""
        # Get obstacles (scaled up for vehicle size)
        ox, oy = self.maze.get_obstacle_list(scale=MAZE_SCALE)
        
        # Create Hybrid A* planner
        planner = HybridAStarPlannerWithTracking(ox, oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        
        # Find path - Hybrid A* needs start and goal with orientation (yaw)
        # Scale start and goal positions
        sx, sy = self.maze.start[0] * MAZE_SCALE, self.maze.start[1] * MAZE_SCALE
        gx, gy = self.maze.exit[0] * MAZE_SCALE, self.maze.exit[1] * MAZE_SCALE
        
        # Set initial and goal orientations (pointing toward goal)
        start_yaw = math.atan2(gy - sy, gx - sx)
        goal_yaw = math.atan2(gy - sy, gx - sx)  # Point toward start initially, will adjust
        
        rx, ry, ryaw = planner.planning(sx, sy, start_yaw, gx, gy, goal_yaw)
        
        if rx and ry:
            # Scale path coordinates back down for visualization
            scaled_path = [(x / MAZE_SCALE, y / MAZE_SCALE) for x, y in zip(rx, ry)]
            self.current_path = scaled_path
            self.path_yaw = ryaw
            self.found_path = True
            return True
        return False
    
    def get_current_position(self):
        """Get current position of creature"""
        if self.current_path and self.path_index < len(self.current_path):
            return self.current_path[self.path_index]
        return self.maze.start
    
    def get_current_yaw(self):
        """Get current orientation of creature"""
        if self.path_yaw and self.path_index < len(self.path_yaw):
            return self.path_yaw[self.path_index]
        return 0.0
    
    def move_next(self):
        """Move to next position in path"""
        if self.path_index < len(self.current_path) - 1:
            self.path_index += 1
            return True
        return False


class MazeGame:
    """Main game class with visualization"""
    
    def __init__(self, maze=None):
        if maze is None:
            self.maze = create_maze()
        else:
            self.maze = maze
        self.creature = Creature(self.maze)
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.animation = None
        self.phase = 'moving'  # Hybrid A* doesn't track exploration the same way
        self.frame_count = 0
    
    def _draw_walls_as_rectangles(self):
        """Draw walls as continuous rectangles instead of individual cells"""
        # Create a 2D array to mark wall cells
        wall_map = np.zeros((self.maze.width, self.maze.height), dtype=bool)
        for wall_x, wall_y in self.maze.walls:
            if 0 <= wall_x < self.maze.width and 0 <= wall_y < self.maze.height:
                wall_map[int(wall_x), int(wall_y)] = True
        
        # Find horizontal wall segments
        visited = np.zeros_like(wall_map)
        rectangles = []
        
        for y in range(self.maze.height):
            for x in range(self.maze.width):
                if wall_map[x, y] and not visited[x, y]:
                    # Find horizontal segment
                    start_x = x
                    end_x = x
                    while end_x < self.maze.width and wall_map[end_x, y] and not visited[end_x, y]:
                        visited[end_x, y] = True
                        end_x += 1
                    
                    # Check if we can extend vertically
                    height = 1
                    can_extend = True
                    while can_extend and y + height < self.maze.height:
                        for check_x in range(start_x, end_x):
                            if not wall_map[check_x, y + height] or visited[check_x, y + height]:
                                can_extend = False
                                break
                        if can_extend:
                            for check_x in range(start_x, end_x):
                                visited[check_x, y + height] = True
                            height += 1
                    
                    # Create rectangle
                    rect = patches.Rectangle(
                        (start_x - 0.5, y - 0.5),
                        end_x - start_x,
                        height,
                        linewidth=0,
                        edgecolor='black',
                        facecolor='black',
                        zorder=1
                    )
                    rectangles.append(rect)
        
        # Draw all rectangles
        for rect in rectangles:
            self.ax.add_patch(rect)
        
        # Add label only once
        if rectangles:
            rectangles[0].set_label('Walls')
        
    def draw_maze(self):
        """Draw the maze"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.maze.width - 0.5)
        self.ax.set_ylim(-0.5, self.maze.height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw walls as continuous rectangles
        self._draw_walls_as_rectangles()
        
        # Draw start position
        sx, sy = self.maze.start
        self.ax.scatter(sx, sy, c='green', s=300, marker='o', 
                       edgecolors='darkgreen', linewidths=2, label='Start', zorder=5)
        
        # Draw exit position
        ex, ey = self.maze.exit
        self.ax.scatter(ex, ey, c='red', s=300, marker='s', 
                       edgecolors='darkred', linewidths=2, label='Exit', zorder=5)
        
        # Draw final path
        if self.creature.found_path and self.creature.current_path:
            path_x = [p[0] for p in self.creature.current_path]
            path_y = [p[1] for p in self.creature.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Path', zorder=4)
        
        # Draw creature with orientation
        cx, cy = self.creature.get_current_position()
        yaw = self.creature.get_current_yaw()
        
        # Draw creature as an arrow showing direction
        creature_circle = plt.Circle((cx, cy), 0.4, color='blue', zorder=10)
        self.ax.add_patch(creature_circle)
        
        # Draw arrow showing direction
        arrow_length = 0.6
        dx = arrow_length * math.cos(yaw)
        dy = arrow_length * math.sin(yaw)
        self.ax.arrow(cx, cy, dx, dy, head_width=0.2, head_length=0.15, 
                     fc='darkblue', ec='darkblue', zorder=11)
        
        # Add eyes
        eye_offset = 0.15
        eye_dx = eye_offset * math.cos(yaw + math.pi/2)
        eye_dy = eye_offset * math.sin(yaw + math.pi/2)
        self.ax.scatter([cx + eye_dx, cx - eye_dx], [cy + eye_dy, cy - eye_dy], 
                       c='white', s=40, zorder=12)
        
        title = f'Maze Game - Hybrid A* Pathfinding (Moving to Exit)'
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
    
    def animate(self, frame):
        """Animation function"""
        self.draw_maze()
        
        # Move creature along path
        if self.creature.found_path:
            if frame > 0 and frame % 2 == 0:  # Move every 2 frames
                self.creature.move_next()
        
        return []
    
    def run(self):
        """Run the game"""
        print("Finding path using Hybrid A* algorithm...")
        success = self.creature.find_path()
        
        if not success:
            print("[Hybrid A*] No path found!")
            self.draw_maze()
            plt.show()
            return
        
        print(f"[Hybrid A*] Path found! Path length: {len(self.creature.current_path)} points")
        print("Starting animation...")
        print("Watch the creature follow the path with vehicle-like turning constraints")
        
        # Create animation
        total_frames = len(self.creature.current_path) * 2 + 20
        self.animation = FuncAnimation(self.fig, self.animate, frames=total_frames,
                                      interval=100, repeat=True, blit=False)
        
        plt.tight_layout()
        plt.show()


def apply_edge_detection(image_path, edge_type, output_path):
    """
    Apply edge detection to an image and save the result.
    
    Args:
        image_path: Path to input image
        edge_type: 'canny' or 'sobel'
        output_path: Path to save the processed image
    
    Returns:
        Path to the processed image
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if edge_type == 'canny':
        # Apply Gaussian Blur to reduce noise
        blur = cv2.GaussianBlur(img, (5, 5), 1.4)
        # Apply Canny Edge Detector
        edges = cv2.Canny(blur, threshold1=100, threshold2=200)
        # Invert colors before saving
        edges = cv2.bitwise_not(edges)
        cv2.imwrite(output_path, edges)
        print(f"[Edge Detection] Applied Canny edge detection, saved to: {os.path.basename(output_path)}")
    elif edge_type == 'sobel':
        # Apply Sobel operator
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges
        # Compute gradient magnitude
        gradient_magnitude = cv2.magnitude(sobelx, sobely)
        # Convert to uint8
        gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
        # Invert colors before saving
        gradient_magnitude = cv2.bitwise_not(gradient_magnitude)
        cv2.imwrite(output_path, gradient_magnitude)
        print(f"[Edge Detection] Applied Sobel edge detection, saved to: {os.path.basename(output_path)}")
    else:
        raise ValueError(f"Unknown edge detection type: {edge_type}")
    
    return output_path


def load_maze_from_maps():
    """Check maps folder for PNG files and load the first one found"""
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    
    if not os.path.exists(maps_dir):
        print(f"Maps directory not found: {maps_dir}")
        return None
    
    # Load configuration
    config = load_planner_config()
    edge_detection = config.get("edge_detection", "none")
    image_name = config.get("image", "map.png")
    
    # Find all PNG files in maps directory
    png_files = glob.glob(os.path.join(maps_dir, '*.png'))
    png_files.extend(glob.glob(os.path.join(maps_dir, '*.PNG')))
    
    if not png_files:
        print(f"No PNG files found in {maps_dir}")
        return None
    
    # Use the specified image from config, or fall back to first PNG file found
    specified_image = os.path.join(maps_dir, image_name)
    if os.path.exists(specified_image):
        original_image_path = specified_image
    else:
        print(f"[Config] Specified image '{image_name}' not found, using first available PNG file")
        original_image_path = png_files[0]
    
    # Apply edge detection if specified
    if edge_detection == 'none':
        image_path = original_image_path
        print(f"Loading maze from image: {os.path.basename(image_path)}")
    elif edge_detection in ['canny', 'sobel']:
        # Create output path for edge-detected image
        output_filename = f"{edge_detection}.png"
        output_path = os.path.join(maps_dir, output_filename)
        
        # Apply edge detection
        try:
            image_path = apply_edge_detection(original_image_path, edge_detection, output_path)
            print(f"Loading maze from edge-detected image: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"[Edge Detection] Error applying {edge_detection}: {e}")
            print(f"[Edge Detection] Falling back to original image")
            image_path = original_image_path
    else:
        print(f"[Edge Detection] Unknown edge detection type: {edge_detection}, using original image")
        image_path = original_image_path
    
    try:
        maze = Maze.from_image(image_path)
        print(f"Successfully loaded maze: {maze.width}x{maze.height}")
        print(f"Start position: {maze.start}")
        print(f"Exit position: {maze.exit}")
        return maze
    except Exception as e:
        print(f"Error loading maze from image: {e}")
        return None


def main():
    """Main function"""
    print("=" * 50)
    print("Maze Game with Hybrid A* Pathfinding")
    print("=" * 50)
    
    # Try to load maze from PNG image in maps folder
    maze = load_maze_from_maps()
    
    if maze is None:
        # Use default generated maze
        print("Using default generated maze")
        print(f"Maze size: {MAZE_WIDTH}x{MAZE_HEIGHT}")
        maze = create_maze()
        print(f"Start: {maze.start}")
        print(f"Exit: {maze.exit}")
    else:
        print(f"Maze loaded from image: {maze.width}x{maze.height}")
    
    print("=" * 50)
    
    game = MazeGame(maze)
    game.run()


if __name__ == '__main__':
    main()

