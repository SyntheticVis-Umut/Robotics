"""
Maze Game with A* Pathfinding Algorithm and Catmull-Rom Spline Smoothing

This file implements a maze pathfinding visualization using the A* algorithm 
to find the shortest path from start to exit, then applies Catmull-Rom spline
smoothing for a smoother path visualization.

The A* algorithm is used for optimal pathfinding, preventing leaks through
wall edges and finding the most efficient route through the maze. The path
is then smoothed using Catmull-Rom spline interpolation for better visualization.

Features:
- A* pathfinding with wall edge leak prevention
- Catmull-Rom spline path smoothing
- Collision detection for smoothed paths
- PNG image maze loading with Canny edge detection
- Visualization of path exploration and final path
- Static path visualization with explored nodes
"""

import sys
import os
import math
import glob
import heapq
import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

# Add the pythonrobotics path to import A* algorithm
def find_pythonrobotics_path():
    """Dynamically find the pythonrobotics directory path."""
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Strategy 1: Check if pythonrobotics is a sibling directory
    parent_dir = os.path.dirname(current_dir)
    sibling_path = os.path.join(parent_dir, 'pythonrobotics')
    if os.path.exists(sibling_path) and os.path.isdir(sibling_path):
        return sibling_path
    
    # Strategy 2: Check if pythonrobotics is in the current directory's parent
    # (for cases where Robotics is inside pythonrobotics)
    grandparent_dir = os.path.dirname(parent_dir)
    grandparent_path = os.path.join(grandparent_dir, 'pythonrobotics')
    if os.path.exists(grandparent_path) and os.path.isdir(grandparent_path):
        return grandparent_path
    
    # Strategy 3: Search up the directory tree
    search_dir = current_dir
    for _ in range(5):  # Search up to 5 levels
        search_dir = os.path.dirname(search_dir)
        pythonrobotics_path = os.path.join(search_dir, 'pythonrobotics')
        if os.path.exists(pythonrobotics_path) and os.path.isdir(pythonrobotics_path):
            return pythonrobotics_path
    
    # Strategy 4: Check common locations
    home_dir = os.path.expanduser('~')
    common_paths = [
        os.path.join(home_dir, 'Desktop', 'pythonrobotics'),
        os.path.join(home_dir, 'Documents', 'pythonrobotics'),
        os.path.join(home_dir, 'pythonrobotics'),
    ]
    for path in common_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    # If not found, raise an error
    raise FileNotFoundError(
        "Could not find pythonrobotics directory. Please ensure it's accessible. "
        f"Searched from: {current_dir}"
    )

pythonrobotics_path = find_pythonrobotics_path()
sys.path.insert(0, pythonrobotics_path)
from PathPlanning.AStar import a_star
from PathPlanning.Catmull_RomSplinePath.catmull_rom_spline_path import catmull_rom_spline
from Mapping.DistanceMap.distance_map import compute_udf_scipy, compute_sdf_scipy

# Maze configuration
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
CELL_SIZE = 1.0
ROBOT_RADIUS = 0.3  # Increased to prevent edge leaks, but still small enough for paths

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "path_config.json")


def load_planner_config():
    """Load planner configuration from JSON; fall back to defaults if missing/invalid."""
    defaults = {
        "use_distance_cost": True,
        "num_points_per_segment": 15,
        "show_distance_map": True,
        "smoothing_enabled": True,
        "distance_heat_gamma": 1.0,
        "detection_mode": "canny",
        "canny_threshold1": 100,
        "canny_threshold2": 200,
        "sobel_threshold": 50,  # Threshold for Sobel edge detection (0-255)
        "map_file": None,
        "grid_resolution": 1.0,  # Node grid cell size in pixels. 1.0 = 1 pixel per node, 10.0 = 10x10 pixels per node
        "obstacle_map_resolution": 1.0,  # Obstacle map resolution for collision detection. 1.0 = full pixel resolution (max sensitivity)
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
        self.canny_edge_map = None  # Store Canny edge detection result as numpy array
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
        # If Canny edge map is available, use it directly
        if self.canny_edge_map is not None:
            if 0 <= int(y) < self.height and 0 <= int(x) < self.width:
                return self.canny_edge_map[int(y), int(x)] > 0
        # Otherwise, use traditional walls set
        return (x, y) in self.walls
    
    def get_valid_positions(self):
        """Get all valid (non-wall) positions in the maze"""
        # If Canny edge map is available, use numpy vectorized operations
        if self.canny_edge_map is not None:
            # Use numpy to find all free space pixels efficiently
            free_coords = np.argwhere(self.canny_edge_map == 0)  # Returns (y, x) coordinates
            # Convert to (x, y) format and convert to list of tuples
            valid_positions = [(float(x), float(y)) for y, x in free_coords]
        else:
            # Traditional method: check walls set
            valid_positions = []
            for x in range(self.width):
                for y in range(self.height):
                    if not self.is_wall(x, y):
                        valid_positions.append((float(x), float(y)))
        return valid_positions
    
    def set_random_start_exit(self, min_distance=None):
        """
        Set random start and exit positions from valid positions.
        
        Args:
            min_distance: Minimum distance between start and exit (optional)
        """
        valid_positions = self.get_valid_positions()
        
        if len(valid_positions) < 2:
            # Fallback to default positions if not enough valid positions
            self.start = (1.0, 1.0)
            self.exit = (float(self.width - 1), float(self.height - 1))
            return
        
        # Try to find positions that are far apart
        max_attempts = 100
        for attempt in range(max_attempts):
            start_pos = random.choice(valid_positions)
            exit_pos = random.choice(valid_positions)
            
            # Make sure they're different
            if start_pos == exit_pos:
                continue
            
            # Check minimum distance if specified
            if min_distance is not None:
                dx = start_pos[0] - exit_pos[0]
                dy = start_pos[1] - exit_pos[1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < min_distance:
                    continue
            
            self.start = start_pos
            self.exit = exit_pos
            return
        
        # Fallback: just pick any two different positions
        if len(valid_positions) >= 2:
            self.start = valid_positions[0]
            self.exit = valid_positions[-1]
        else:
            self.start = (1.0, 1.0)
            self.exit = (float(self.width - 1), float(self.height - 1))
    
    def get_obstacle_list(self):
        """Convert walls to obstacle lists for A* planner"""
        # If Canny edge map is available, use numpy vectorized operations
        if self.canny_edge_map is not None:
            # Use numpy to find all edge pixels efficiently
            edge_coords = np.argwhere(self.canny_edge_map > 0)  # Returns (y, x) coordinates
            # Convert to (x, y) format
            oy, ox = edge_coords[:, 0], edge_coords[:, 1]
            ox = ox.tolist()
            oy = oy.tolist()
            
            # Add border walls
            border_x = list(range(self.width)) + list(range(self.width)) + [0] * self.height + [self.width - 1] * self.height
            border_y = [0] * self.width + [self.height - 1] * self.width + list(range(self.height)) + list(range(self.height))
            ox.extend(border_x)
            oy.extend(border_y)
        else:
            # Traditional method: build from walls set
            ox, oy = [], []
        # Add border walls
        for i in range(self.width):
            ox.append(i)
            oy.append(0)
            ox.append(i)
            oy.append(self.height - 1)
        for i in range(self.height):
            ox.append(0)
            oy.append(i)
            ox.append(self.width - 1)
            oy.append(i)
        
        # Add internal walls
        for wall_x, wall_y in self.walls:
            ox.append(wall_x)
            oy.append(wall_y)
        
        return ox, oy
    
    def get_obstacle_map_2d(self):
        """
        Create a 2D boolean array representing obstacles for distance mapping.
        Returns a numpy array where True = obstacle, False = free space.
        Shape: (width, height) - x, y coordinates
        """
        # If Canny edge map is available, use it directly
        if self.canny_edge_map is not None:
            # Canny edge map is (height, width) format (y, x)
            # Convert to boolean: 0 = free space, >0 = edge/obstacle
            # Transpose to (width, height) format (x, y)
            obstacle_map = (self.canny_edge_map > 0).T.astype(bool)
            
            # Add border walls
            obstacle_map[:, 0] = True  # Top border
            obstacle_map[:, self.height - 1] = True  # Bottom border
            obstacle_map[0, :] = True  # Left border
            obstacle_map[self.width - 1, :] = True  # Right border
            
            return obstacle_map
        else:
            # Traditional method: build from walls set
            obstacle_map = np.zeros((self.width, self.height), dtype=bool)
            
            # Add border walls
            obstacle_map[:, 0] = True  # Top border
            obstacle_map[:, self.height - 1] = True  # Bottom border
            obstacle_map[0, :] = True  # Left border
            obstacle_map[self.width - 1, :] = True  # Right border
            
            # Add internal walls
            for wall_x, wall_y in self.walls:
                if 0 <= wall_x < self.width and 0 <= wall_y < self.height:
                    obstacle_map[int(wall_x), int(wall_y)] = True
            
            return obstacle_map
    
    def compute_distance_field(self, use_sdf=True):
        """
        Compute distance field from obstacles.
        
        Args:
            use_sdf: If True, compute Signed Distance Field (SDF).
                    If False, compute Unsigned Distance Field (UDF).
        
        Returns:
            Distance field array (2D numpy array)
        """
        obstacle_map = self.get_obstacle_map_2d()
        
        # Note: distance_map functions expect obstacles as 1, free space as 0
        # Our obstacle_map already has this format (True=1, False=0)
        obstacles_bool = obstacle_map.astype(int)
        
        if use_sdf:
            return compute_sdf_scipy(obstacles_bool)
        else:
            return compute_udf_scipy(obstacles_bool)
    
    @staticmethod
    def from_image(image_path, start_pos=None, exit_pos=None, detection_mode='canny', canny_threshold1=100, canny_threshold2=200, sobel_threshold=50):
        """
        Create a maze from a PNG image using edge detection (Canny or Sobel) or legacy threshold mode
        
        Args:
            image_path: Path to PNG image
            start_pos: Tuple (x, y) for start position, or None for auto-detection
            exit_pos: Tuple (x, y) for exit position, or None for auto-detection
            detection_mode: 'canny' for Canny edge detection, 'sobel' for Sobel edge detection, 'none' for legacy threshold mode
            canny_threshold1: Lower threshold for Canny edge detection (default: 100)
            canny_threshold2: Upper threshold for Canny edge detection (default: 200)
            sobel_threshold: Threshold for Sobel edge detection (default: 50)
        
        Returns:
            Maze object
        """
        # Load image using OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Flip vertically and horizontally so origin aligns with plot
        #img = np.flipud(img)   # bottom-to-top
        
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        height, width = img.shape
        
        # Create maze
        maze = Maze(width, height)
        
        if detection_mode == 'canny':
            # Apply Gaussian blur to reduce noise before edge detection
            blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
            
            # Apply Canny edge detection
            # Canny output: white pixels (255) = edges, black pixels (0) = non-edges
            edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
            
            # Store Canny edge map directly as numpy array
            # Canny output: white (255) = edges = obstacles, black (0) = free space
            maze.canny_edge_map = edges

            # Save Canny edge map for reference
            try:
                maps_dir = os.path.dirname(image_path)
                canny_output_path = os.path.join(maps_dir, 'canny.png')
                cv2.imwrite(canny_output_path, edges)
                print(f"Canny edge map saved to: {canny_output_path}")
            except Exception as e:
                print(f"[Warning] Failed to save Canny edge map: {e}")
        
        elif detection_mode == 'sobel':
            # Apply Gaussian blur to reduce noise before edge detection
            blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
            
            # Apply Sobel edge detection
            # Sobel computes gradient in x and y directions
            sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute gradient magnitude
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            # Normalize to 0-255 range
            sobel_magnitude = np.uint8(255 * sobel_magnitude / (np.max(sobel_magnitude) + 1e-8))
            
            # Apply threshold to create binary edge map
            # Pixels above threshold = edges = obstacles
            _, edges = cv2.threshold(sobel_magnitude, sobel_threshold, 255, cv2.THRESH_BINARY)
            
            # Store Sobel edge map directly as numpy array
            # Sobel output: white (255) = edges = obstacles, black (0) = free space
            maze.canny_edge_map = edges  # Reuse canny_edge_map variable for backward compatibility

            # Save Sobel edge map for reference
            try:
                maps_dir = os.path.dirname(image_path)
                sobel_output_path = os.path.join(maps_dir, 'sobel.png')
                cv2.imwrite(sobel_output_path, edges)
                print(f"Sobel edge map saved to: {sobel_output_path}")
            except Exception as e:
                print(f"[Warning] Failed to save Sobel edge map: {e}")
        
        else:
            # Legacy mode: black pixels = walls, white pixels = free paths
            # Convert to binary: threshold at 128 (black < 128 = wall, white >= 128 = free)
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            # In legacy mode, we convert to walls set
            for y in range(height):
                for x in range(width):
                    if binary[y, x] > 0:  # Wall (originally black pixel)
                        maze.add_wall(x, y)
            # No canny_edge_map in legacy mode
            maze.canny_edge_map = None
        
        # Auto-detect start and exit if not provided
        if start_pos is None or exit_pos is None:
            # Set random start and exit positions
            # Calculate minimum distance as 30% of maze diagonal
            diagonal = math.sqrt(width*width + height*height)
            min_distance = diagonal * 0.3
            maze.set_random_start_exit(min_distance=min_distance)
        else:
            maze.start = start_pos
            maze.exit = exit_pos
        
        return maze


def create_maze():
    """Create a simple maze with guaranteed path from start to exit"""
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    
    # Create a simpler maze with a clear winding path
    # We'll create walls but ensure there's always a path
    
    # Horizontal walls with gaps
    maze.add_wall_line(3, 2, 3, horizontal=True)  # Gap at x=6-7
    maze.add_wall_line(8, 2, 4, horizontal=True)  # Gap at x=12-13
    maze.add_wall_line(2, 4, 3, horizontal=True)  # Gap at x=5-6
    maze.add_wall_line(7, 4, 3, horizontal=True)  # Gap at x=10-11
    maze.add_wall_line(12, 4, 3, horizontal=True)  # Gap at x=15-16
    maze.add_wall_line(4, 6, 3, horizontal=True)  # Gap at x=7-8
    maze.add_wall_line(9, 6, 3, horizontal=True)  # Gap at x=12-13
    maze.add_wall_line(3, 8, 3, horizontal=True)  # Gap at x=6-7
    maze.add_wall_line(8, 8, 3, horizontal=True)  # Gap at x=11-12
    maze.add_wall_line(13, 8, 3, horizontal=True)  # Gap at x=16-17
    maze.add_wall_line(2, 10, 3, horizontal=True)  # Gap at x=5-6
    maze.add_wall_line(7, 10, 3, horizontal=True)  # Gap at x=10-11
    maze.add_wall_line(12, 10, 3, horizontal=True)  # Gap at x=15-16
    maze.add_wall_line(4, 12, 3, horizontal=True)  # Gap at x=7-8
    maze.add_wall_line(9, 12, 3, horizontal=True)  # Gap at x=12-13
    maze.add_wall_line(14, 12, 3, horizontal=True)  # Gap at x=17-18
    maze.add_wall_line(3, 14, 3, horizontal=True)  # Gap at x=6-7
    maze.add_wall_line(8, 14, 3, horizontal=True)  # Gap at x=11-12
    maze.add_wall_line(13, 14, 3, horizontal=True)  # Gap at x=16-17
    maze.add_wall_line(2, 16, 3, horizontal=True)  # Gap at x=5-6
    maze.add_wall_line(7, 16, 3, horizontal=True)  # Gap at x=10-11
    maze.add_wall_line(12, 16, 3, horizontal=True)  # Gap at x=15-16
    
    # Vertical walls with gaps
    maze.add_wall_line(5, 1, 2, horizontal=False)  # Gap at y=3-4
    maze.add_wall_line(10, 1, 2, horizontal=False)  # Gap at y=3-4
    maze.add_wall_line(15, 1, 2, horizontal=False)  # Gap at y=3-4
    maze.add_wall_line(6, 3, 2, horizontal=False)  # Gap at y=5-6
    maze.add_wall_line(11, 3, 2, horizontal=False)  # Gap at y=5-6
    maze.add_wall_line(16, 3, 2, horizontal=False)  # Gap at y=5-6
    maze.add_wall_line(7, 5, 2, horizontal=False)  # Gap at y=7-8
    maze.add_wall_line(12, 5, 2, horizontal=False)  # Gap at y=7-8
    maze.add_wall_line(17, 5, 2, horizontal=False)  # Gap at y=7-8
    maze.add_wall_line(6, 7, 2, horizontal=False)  # Gap at y=9-10
    maze.add_wall_line(11, 7, 2, horizontal=False)  # Gap at y=9-10
    maze.add_wall_line(16, 7, 2, horizontal=False)  # Gap at y=9-10
    maze.add_wall_line(5, 9, 2, horizontal=False)  # Gap at y=11-12
    maze.add_wall_line(10, 9, 2, horizontal=False)  # Gap at y=11-12
    maze.add_wall_line(15, 9, 2, horizontal=False)  # Gap at y=11-12
    maze.add_wall_line(8, 11, 2, horizontal=False)  # Gap at y=13-14
    maze.add_wall_line(13, 11, 2, horizontal=False)  # Gap at y=13-14
    maze.add_wall_line(18, 11, 2, horizontal=False)  # Gap at y=13-14
    maze.add_wall_line(7, 13, 2, horizontal=False)  # Gap at y=15-16
    maze.add_wall_line(12, 13, 2, horizontal=False)  # Gap at y=15-16
    maze.add_wall_line(17, 13, 2, horizontal=False)  # Gap at y=15-16
    maze.add_wall_line(6, 15, 2, horizontal=False)  # Gap at y=17-18
    maze.add_wall_line(11, 15, 2, horizontal=False)  # Gap at y=17-18
    maze.add_wall_line(16, 15, 2, horizontal=False)  # Gap at y=17-18
    
    # Set random start and exit positions
    diagonal = math.sqrt(MAZE_WIDTH*MAZE_WIDTH + MAZE_HEIGHT*MAZE_HEIGHT)
    min_distance = diagonal * 0.3
    maze.set_random_start_exit(min_distance=min_distance)
    
    return maze


class AStarPlannerWithTracking(a_star.AStarPlanner):
    """Optimized A* planner with GPU support, tracking, wall edge leak prevention, and distance field cost"""
    
    def calc_obstacle_map(self, ox, oy):
        """
        Optimized obstacle map calculation using vectorized operations.
        Much faster than the original triple-nested loop, especially for large maps.
        Uses GPU if available, otherwise vectorized CPU operations.
        
        Note: obstacle_map_resolution controls collision detection precision,
        while self.resolution controls node grid size.
        """
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        # Node grid size (for A* search space)
        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        # Obstacle map size (for collision detection precision)
        # Use obstacle_map_resolution for precise collision checking
        obstacle_x_width = round((self.max_x - self.min_x) / self.obstacle_map_resolution)
        obstacle_y_width = round((self.max_y - self.min_y) / self.obstacle_map_resolution)
        print(f"[Obstacle Map] Resolution: {self.obstacle_map_resolution}, Size: {obstacle_x_width}x{obstacle_y_width}")

        # Use vectorized operations for much faster obstacle map generation
        # Create obstacle map grid coordinates (at obstacle_map_resolution)
        obstacle_x_coords = np.arange(obstacle_x_width) * self.obstacle_map_resolution + self.min_x
        obstacle_y_coords = np.arange(obstacle_y_width) * self.obstacle_map_resolution + self.min_y
        
        # Create meshgrid for obstacle map (full resolution)
        X_obstacle, Y_obstacle = np.meshgrid(obstacle_x_coords, obstacle_y_coords, indexing='ij')
        
        # Also create node grid coordinates (at node grid resolution) for sampling
        x_coords = np.arange(self.x_width) * self.resolution + self.min_x
        y_coords = np.arange(self.y_width) * self.resolution + self.min_y
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Create meshgrid for all grid points
        X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
        
        # Convert obstacle points to numpy array
        ox_array = np.array(ox)
        oy_array = np.array(oy)
        
        # Use GPU if available for distance calculations
        if GPU_AVAILABLE and GPU_USABLE:
            try:
                num_obstacles = len(ox_array)
                batch_size = 2000  # Process obstacles in batches to avoid GPU memory issues
                
                print(f"[GPU] Computing obstacle map on GPU at {self.obstacle_map_resolution}px resolution (processing {num_obstacles} obstacles in batches)...")
                
                # Transfer obstacle map grid coordinates to GPU (full resolution for collision detection)
                X_obstacle_gpu = cp.asarray(X_obstacle)
                Y_obstacle_gpu = cp.asarray(Y_obstacle)
                rr_gpu = cp.float32(self.rr)
                
                # Initialize full-resolution obstacle map on GPU (all False initially)
                obstacle_map_full_gpu = cp.zeros((obstacle_x_width, obstacle_y_width), dtype=cp.bool_)
                
                # Process obstacles in batches
                num_batches = (num_obstacles + batch_size - 1) // batch_size
                print(f"[GPU] Processing {num_obstacles} obstacles in {num_batches} batches of ~{batch_size}")
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_obstacles)
                    batch_ox = ox_array[start_idx:end_idx]
                    batch_oy = oy_array[start_idx:end_idx]
                    
                    # Transfer batch to GPU
                    ox_batch_gpu = cp.asarray(batch_ox)
                    oy_batch_gpu = cp.asarray(batch_oy)
                    
                    # Compute distances using broadcasting on GPU for this batch
                    # Shape: (obstacle_x_width, obstacle_y_width, batch_size)
                    dx = X_obstacle_gpu[:, :, cp.newaxis] - ox_batch_gpu[cp.newaxis, cp.newaxis, :]
                    dy = Y_obstacle_gpu[:, :, cp.newaxis] - oy_batch_gpu[cp.newaxis, cp.newaxis, :]
                    distances = cp.sqrt(dx * dx + dy * dy)
                    
                    # Check if any obstacle in this batch is within robot radius
                    # Shape: (obstacle_x_width, obstacle_y_width)
                    batch_obstacle_map = cp.any(distances <= rr_gpu, axis=2)
                    
                    # Accumulate results using logical OR
                    obstacle_map_full_gpu = obstacle_map_full_gpu | batch_obstacle_map
                    
                    # Free GPU memory for this batch
                    del dx, dy, distances, batch_obstacle_map, ox_batch_gpu, oy_batch_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                
                # Transfer full-resolution obstacle map back to CPU
                obstacle_map_full = cp.asnumpy(obstacle_map_full_gpu)
                
                # Sample full-resolution obstacle map at node grid positions
                # Convert node grid positions to obstacle map indices
                node_x_indices = ((X - self.min_x) / self.obstacle_map_resolution).astype(int)
                node_y_indices = ((Y - self.min_y) / self.obstacle_map_resolution).astype(int)
                
                # Clip indices to valid range
                node_x_indices = np.clip(node_x_indices, 0, obstacle_x_width - 1)
                node_y_indices = np.clip(node_y_indices, 0, obstacle_y_width - 1)
                
                # Sample obstacle map at node grid positions
                obstacle_map = obstacle_map_full[node_x_indices, node_y_indices]
                
                # Clean up GPU memory
                del X_obstacle_gpu, Y_obstacle_gpu, obstacle_map_full_gpu
                cp.get_default_memory_pool().free_all_blocks()
                
                print("[GPU] ✓ Obstacle map computed on GPU at full resolution and sampled to node grid")
            except Exception as e:
                print(f"[GPU] ⚠ GPU computation failed: {e}, using CPU")
                # Fallback to CPU vectorized
                obstacle_map_full = self._calc_obstacle_map_cpu_vectorized(X_obstacle, Y_obstacle, ox_array, oy_array)
                
                # Sample full-resolution obstacle map at node grid positions
                node_x_indices = ((X - self.min_x) / self.obstacle_map_resolution).astype(int)
                node_y_indices = ((Y - self.min_y) / self.obstacle_map_resolution).astype(int)
                node_x_indices = np.clip(node_x_indices, 0, obstacle_x_width - 1)
                node_y_indices = np.clip(node_y_indices, 0, obstacle_y_width - 1)
                obstacle_map = obstacle_map_full[node_x_indices, node_y_indices]
        else:
            # Use CPU vectorized operations
            obstacle_map_full = self._calc_obstacle_map_cpu_vectorized(X_obstacle, Y_obstacle, ox_array, oy_array)
            
            # Sample full-resolution obstacle map at node grid positions
            node_x_indices = ((X - self.min_x) / self.obstacle_map_resolution).astype(int)
            node_y_indices = ((Y - self.min_y) / self.obstacle_map_resolution).astype(int)
            node_x_indices = np.clip(node_x_indices, 0, obstacle_x_width - 1)
            node_y_indices = np.clip(node_y_indices, 0, obstacle_y_width - 1)
            obstacle_map = obstacle_map_full[node_x_indices, node_y_indices]
        
        # Store full-resolution obstacle map for later use
        self.obstacle_map_full = obstacle_map_full
        self.obstacle_map_full_resolution = self.obstacle_map_resolution
        
        # Convert to list of lists format expected by parent class (node grid size)
        self.obstacle_map = [[bool(obstacle_map[ix][iy]) for iy in range(self.y_width)]
                             for ix in range(self.x_width)]
    
    def _calc_obstacle_map_cpu_vectorized(self, X, Y, ox_array, oy_array):
        """
        CPU-optimized obstacle map using vectorized NumPy operations.
        Uses broadcasting to compute all distances at once.
        """
        # Compute distances using broadcasting
        # Shape: (x_width, y_width, num_obstacles)
        dx = X[:, :, np.newaxis] - ox_array[np.newaxis, np.newaxis, :]
        dy = Y[:, :, np.newaxis] - oy_array[np.newaxis, np.newaxis, :]
        distances = np.sqrt(dx * dx + dy * dy)
        
        # Check if any obstacle is within robot radius
        # Shape: (x_width, y_width)
        obstacle_map = np.any(distances <= self.rr, axis=2)
        
        return obstacle_map
    
    def __init__(self, ox, oy, resolution, rr, distance_field=None, use_distance_cost=True, min_clearance_norm=0.50, obstacle_map_resolution=None):
        # Store obstacle map resolution (for collision detection precision)
        # If None, use same as node grid resolution
        self.obstacle_map_resolution = obstacle_map_resolution if obstacle_map_resolution is not None else resolution
        
        # Temporarily disable animation in parent class
        original_show = a_star.show_animation
        a_star.show_animation = False
        super().__init__(ox, oy, resolution, rr)
        a_star.show_animation = original_show
        
        self.explored_nodes = []
        self.open_set_history = []
        # Store original motion model
        self.original_motion = self.motion.copy()
        
        # Pre-compute motion model as numpy array for vectorization
        self.motion_array = np.array(self.motion, dtype=np.float32)
        
        # Distance field for safety-aware pathfinding
        # Store original distance field and maze info for mapping
        self.distance_field = distance_field
        self.use_distance_cost = use_distance_cost and (distance_field is not None)
        self.min_clearance_norm = min_clearance_norm  # Hard cutoff for minimum normalized clearance
        
        if self.use_distance_cost:
            print(f"[A*] ✓ Distance field cost enabled")
            # Normalize distance field to [0, 1] for cost calculation
            if distance_field is not None:
                max_dist = np.max(distance_field)
                if max_dist > 0:
                    self.distance_field_norm = distance_field / max_dist
                else:
                    self.distance_field_norm = distance_field
                    self.use_distance_cost = False
        else:
            self.distance_field_norm = None
        
        # Use GPU for obstacle map and distance field if available and usable
        self.use_gpu = False
        self.distance_field_norm_gpu = None
        if GPU_AVAILABLE and GPU_USABLE:
            try:
                self.obstacle_map_gpu = cp.asarray(self.obstacle_map)
                # Also create GPU version of distance field if available
                if self.distance_field_norm is not None:
                    self.distance_field_norm_gpu = cp.asarray(self.distance_field_norm)
                self.use_gpu = True
                print(f"[GPU] ✓ Using GPU for obstacle map processing")
                print(f"[GPU] Obstacle map size: {self.x_width}x{self.y_width}")
                if self.distance_field_norm_gpu is not None:
                    print(f"[GPU] ✓ Using GPU for distance field lookups")
            except Exception as e:
                self.use_gpu = False
                print(f"[GPU] ✗ Failed to initialize GPU arrays: {e}")
                print("[GPU] Falling back to CPU")
        else:
            self.use_gpu = False
            if not GPU_AVAILABLE:
                print("[GPU] ✗ GPU not available, using CPU for obstacle map")
            elif not GPU_USABLE:
                print("[GPU] ✗ GPU not usable, using CPU for obstacle map")
        
        # Convert obstacle map to numpy array for faster access
        self.obstacle_map_np = np.array(self.obstacle_map, dtype=bool)
        if not self.use_gpu:
            print("[CPU] ✓ Using optimized CPU (NumPy) for obstacle map")
    
    def _calc_obstacle_map_cpu_vectorized(self, X, Y, ox_array, oy_array):
        """
        CPU-optimized obstacle map using vectorized NumPy operations.
        Uses broadcasting to compute all distances at once.
        """
        # Compute distances using broadcasting
        # Shape: (x_width, y_width, num_obstacles)
        dx = X[:, :, np.newaxis] - ox_array[np.newaxis, np.newaxis, :]
        dy = Y[:, :, np.newaxis] - oy_array[np.newaxis, np.newaxis, :]
        distances = np.sqrt(dx * dx + dy * dy)
        
        # Check if any obstacle is within robot radius
        # Shape: (x_width, y_width)
        obstacle_map = np.any(distances <= self.rr, axis=2)
        
        return obstacle_map
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def verify_node_fast(node_x, node_y, x_width, y_width, obstacle_map_np, 
                         parent_x=None, parent_y=None):
        """
        Fast JIT-compiled node verification with wall edge leak prevention.
        """
        # Bounds check
        if node_x < 0 or node_x >= x_width or node_y < 0 or node_y >= y_width:
            return False

        # Collision check using numpy array (faster)
        if obstacle_map_np[node_x, node_y]:
            return False
        
        # Diagonal corner check
        if parent_x is not None and parent_y is not None:
            dx = node_x - parent_x
            dy = node_y - parent_y
            
            # If diagonal move, check adjacent cells
            if dx != 0 and dy != 0:
                adj1_x = parent_x + dx
                adj1_y = parent_y
                adj2_x = parent_x
                adj2_y = parent_y + dy
                
                if (0 <= adj1_x < x_width and 0 <= adj1_y < y_width and
                    0 <= adj2_x < x_width and 0 <= adj2_y < y_width):
                    if (obstacle_map_np[adj1_x, adj1_y] or 
                        obstacle_map_np[adj2_x, adj2_y]):
                        return False

        return True
    
    def verify_node(self, node, parent_node=None):
        """
        Verify node with additional check for diagonal movement through wall corners.
        Uses GPU acceleration if available, otherwise JIT-compiled version or fallback.
        """
        parent_x = parent_node.x if parent_node is not None else None
        parent_y = parent_node.y if parent_node is not None else None
        
        # Use GPU if available
        if self.use_gpu and GPU_AVAILABLE and GPU_USABLE:
            # Bounds check
            if node.x < 0 or node.x >= self.x_width or node.y < 0 or node.y >= self.y_width:
                return False
            
            # Collision check using GPU array
            try:
                # Use item() to get Python scalar directly from GPU (more efficient than asnumpy)
                if self.obstacle_map_gpu[node.x, node.y].item():
                    return False
            except Exception as e:
                # Fallback to CPU if GPU access fails
                if self.obstacle_map_np[node.x, node.y]:
                    return False
            
            # Diagonal corner check
            if parent_x is not None and parent_y is not None:
                dx = node.x - parent_x
                dy = node.y - parent_y
                
                # If diagonal move, check adjacent cells
                if dx != 0 and dy != 0:
                    adj1_x = parent_x + dx
                    adj1_y = parent_y
                    adj2_x = parent_x
                    adj2_y = parent_y + dy
                    
                    if (0 <= adj1_x < self.x_width and 0 <= adj1_y < self.y_width and
                        0 <= adj2_x < self.x_width and 0 <= adj2_y < self.y_width):
                        try:
                            # Use GPU for adjacent cell checks (batch check for efficiency)
                            adj1_val = self.obstacle_map_gpu[adj1_x, adj1_y].item()
                            adj2_val = self.obstacle_map_gpu[adj2_x, adj2_y].item()
                            if adj1_val or adj2_val:
                                return False
                        except Exception:
                            # Fallback to CPU
                            if (self.obstacle_map_np[adj1_x, adj1_y] or 
                                self.obstacle_map_np[adj2_x, adj2_y]):
                                return False
            
            return True
        
        # Use CPU with Numba JIT if available
        elif NUMBA_AVAILABLE:
            return self.verify_node_fast(node.x, node.y, self.x_width, self.y_width,
                                        self.obstacle_map_np, parent_x, parent_y)
        else:
            # Fallback to original method
            px = self.calc_grid_position(node.x, self.min_x)
            py = self.calc_grid_position(node.y, self.min_y)

            if px < self.min_x or py < self.min_y or px >= self.max_x or py >= self.max_y:
                return False

            if self.obstacle_map[node.x][node.y]:
                return False
            
            if parent_node is not None:
                dx = node.x - parent_node.x
                dy = node.y - parent_node.y
                
                if dx != 0 and dy != 0:
                    adj1_x = parent_node.x + dx
                    adj1_y = parent_node.y
                    adj2_x = parent_node.x
                    adj2_y = parent_node.y + dy
                    
                    if (0 <= adj1_x < self.x_width and 0 <= adj1_y < self.y_width and
                        0 <= adj2_x < self.x_width and 0 <= adj2_y < self.y_width):
                        if (self.obstacle_map[adj1_x][adj1_y] or 
                            self.obstacle_map[adj2_x][adj2_y]):
                            return False

            return True
        
    def planning(self, sx, sy, gx, gy):
        """Optimized A* path search with exploration tracking and priority queue"""
        print(f"[A*] Starting pathfinding from ({sx:.1f}, {sy:.1f}) to ({gx:.1f}, {gy:.1f})")
        if self.use_gpu:
            print("[A*] Using GPU-accelerated pathfinding")
        elif NUMBA_AVAILABLE:
            print("[A*] Using CPU pathfinding with Numba JIT compilation")
        else:
            print("[A*] Using optimized CPU pathfinding")
        
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # Use priority queue (heap) instead of dict + min() for O(log n) operations
        # Format: (f_cost, g_cost, node_id, node)
        open_heap = []
        open_set_dict = {}  # For O(1) lookup
        closed_set = {}
        
        start_id = self.calc_grid_index(start_node)
        start_f = 0.0 + self.calc_heuristic(goal_node, start_node)
        heapq.heappush(open_heap, (start_f, 0.0, start_id, start_node))
        open_set_dict[start_id] = start_node
        
        self.explored_nodes = []
        self.open_set_history = []
        
        # Batch processing parameters for GPU acceleration
        batch_size = 256 if (self.use_gpu and GPU_AVAILABLE and GPU_USABLE) else 1
        nodes_processed = 0

        while True:
            if len(open_heap) == 0:
                print("Open set is empty..")
                break

            # Batch processing: collect multiple nodes to process in parallel on GPU
            batch_nodes = []
            batch_f_costs = []
            
            # Collect a batch of nodes with similar f_costs (within tolerance)
            if batch_size > 1 and self.use_gpu and GPU_AVAILABLE and GPU_USABLE:
                if len(open_heap) > 0:
                    # Get the minimum f_cost
                    min_f_cost = open_heap[0][0]
                    f_cost_tolerance = 0.1  # Process nodes with f_cost within 0.1 of minimum
                    
                    # Collect up to batch_size nodes
                    temp_heap = []
                    while len(batch_nodes) < batch_size and len(open_heap) > 0:
                        f_cost, g_cost, c_id, node = heapq.heappop(open_heap)
                        
                        # Only batch nodes with similar f_costs to maintain near-optimality
                        if len(batch_nodes) == 0 or abs(f_cost - min_f_cost) <= f_cost_tolerance:
                            if c_id in open_set_dict and open_set_dict[c_id] == node:
                                batch_nodes.append((f_cost, g_cost, c_id, node))
                                batch_f_costs.append(f_cost)
                                del open_set_dict[c_id]
                            else:
                                # Node was updated, skip it
                                continue
                        else:
                            # f_cost too different, put it back
                            heapq.heappush(open_heap, (f_cost, g_cost, c_id, node))
                            break
                    
                    # Put remaining nodes back
                    for item in temp_heap:
                        heapq.heappush(open_heap, item)
            else:
                # Single node processing (CPU or small batches)
                if len(open_heap) > 0:
                    f_cost, g_cost, c_id, node = heapq.heappop(open_heap)
                    if c_id in open_set_dict and open_set_dict[c_id] == node:
                        batch_nodes.append((f_cost, g_cost, c_id, node))
                        batch_f_costs.append(f_cost)
                        del open_set_dict[c_id]
            
            if len(batch_nodes) == 0:
                continue
            
            # Process batch of nodes
            for f_cost, g_cost, c_id, current in batch_nodes:
                # Track explored nodes
                explored_pos = (self.calc_grid_position(current.x, self.min_x),
                               self.calc_grid_position(current.y, self.min_y))
                self.explored_nodes.append(explored_pos)
                nodes_processed += 1
                
                # Check for goal
                if current.x == goal_node.x and current.y == goal_node.y:
                    print(f"[A*] ✓ Goal found! Explored {len(self.explored_nodes)} nodes")
                    if self.use_gpu:
                        print("[A*] Pathfinding completed using GPU acceleration")
                    elif NUMBA_AVAILABLE:
                        print("[A*] Pathfinding completed using CPU with Numba JIT")
                    else:
                        print("[A*] Pathfinding completed using optimized CPU")
                    goal_node.parent_index = current.parent_index
                    goal_node.cost = current.cost
                    break
                
                # Add to closed set
                closed_set[c_id] = current
            
            # Check if goal was found
            if goal_node.parent_index != -1:
                break
            
            # Track open set (for visualization) - only once per batch
            open_positions = [(self.calc_grid_position(n.x, self.min_x),
                              self.calc_grid_position(n.y, self.min_y))
                             for n in open_set_dict.values()]
            self.open_set_history.append(open_positions.copy())
            
            # Batch process neighbors for all nodes in the batch
            if batch_size > 1 and self.use_gpu and GPU_AVAILABLE and GPU_USABLE:
                # Process all nodes in batch together on GPU
                self._process_batch_nodes_gpu(batch_nodes, goal_node, open_heap, open_set_dict, closed_set)
            else:
                # Process nodes individually
                for f_cost, g_cost, c_id, current in batch_nodes:
                    if current.x == goal_node.x and current.y == goal_node.y:
                        continue
                    
                    self._process_single_node(current, c_id, goal_node, open_heap, open_set_dict, closed_set)

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry
    
    def _process_single_node(self, current, c_id, goal_node, open_heap, open_set_dict, closed_set):
        """Process neighbors for a single node (CPU or fallback)"""
        # Expand grid search based on motion model
        neighbor_nodes = []
        neighbor_coords = []
        
        for i, motion_step in enumerate(self.motion):
            new_x = current.x + motion_step[0]
            new_y = current.y + motion_step[1]
            new_cost = current.cost + motion_step[2]
            
            # Bounds check
            if new_x < 0 or new_x >= self.x_width or new_y < 0 or new_y >= self.y_width:
                continue
            
            neighbor_nodes.append((new_x, new_y, new_cost, motion_step))
            neighbor_coords.append((new_x, new_y))
        
        # Batch verify all neighbors on GPU using CUDA
        valid_neighbors = []
        if self.use_gpu and GPU_AVAILABLE and GPU_USABLE and len(neighbor_coords) > 0:
            try:
                # Prepare all data on GPU (CUDA) - minimize CPU-GPU transfers
                n_neighbors = len(neighbor_coords)
                x_coords = cp.array([nc[0] for nc in neighbor_coords], dtype=cp.int32)
                y_coords = cp.array([nc[1] for nc in neighbor_coords], dtype=cp.int32)
                
                # Vectorized obstacle check on CUDA (all operations stay on GPU)
                obstacles_gpu = self.obstacle_map_gpu[x_coords, y_coords]
                
                # Prepare distance field coordinates on GPU if needed
                dist_values_gpu = None
                if self.use_distance_cost and self.distance_field_norm_gpu is not None:
                    df_width, df_height = self.distance_field_norm.shape
                    # Compute distance field coordinates on GPU (all operations on CUDA)
                    # Convert grid indices to world coordinates on GPU
                    px_gpu = x_coords.astype(cp.float32) * self.resolution + self.min_x
                    py_gpu = y_coords.astype(cp.float32) * self.resolution + self.min_y
                    
                    # Clip and convert to maze grid indices on GPU
                    df_x_gpu = cp.clip(px_gpu, 0, df_width - 1).astype(cp.int32)
                    df_y_gpu = cp.clip(py_gpu, 0, df_height - 1).astype(cp.int32)
                    
                    # Batch distance field lookup on CUDA
                    dist_values_gpu = self.distance_field_norm_gpu[df_x_gpu, df_y_gpu]
                
                # Prepare diagonal check data on GPU (all CUDA operations)
                # Compute deltas for all neighbors on GPU
                current_x_gpu = cp.array([current.x] * n_neighbors, dtype=cp.int32)
                current_y_gpu = cp.array([current.y] * n_neighbors, dtype=cp.int32)
                dx_gpu = x_coords - current_x_gpu
                dy_gpu = y_coords - current_y_gpu
                
                # Identify diagonal moves on GPU (dx != 0 AND dy != 0)
                is_diagonal = (dx_gpu != 0) & (dy_gpu != 0)
                
                # For diagonal moves, compute adjacent cell coordinates on GPU
                adj1_x_gpu = current_x_gpu + dx_gpu
                adj1_y_gpu = current_y_gpu
                adj2_x_gpu = current_x_gpu
                adj2_y_gpu = current_y_gpu + dy_gpu
                
                # Check bounds on GPU
                adj1_valid = (adj1_x_gpu >= 0) & (adj1_x_gpu < self.x_width) & \
                             (adj1_y_gpu >= 0) & (adj1_y_gpu < self.y_width)
                adj2_valid = (adj2_x_gpu >= 0) & (adj2_x_gpu < self.x_width) & \
                             (adj2_y_gpu >= 0) & (adj2_y_gpu < self.y_width)
                
                # Initialize obstacle arrays on GPU
                adj1_obstacles = cp.zeros(n_neighbors, dtype=cp.bool_)
                adj2_obstacles = cp.zeros(n_neighbors, dtype=cp.bool_)
                
                # Batch check adjacent cells on GPU using vectorized operations
                # Use conditional indexing: only check where valid
                valid_adj1_indices = cp.where(adj1_valid)[0]
                valid_adj2_indices = cp.where(adj2_valid)[0]
                
                if len(valid_adj1_indices) > 0:
                    adj1_x_valid = adj1_x_gpu[valid_adj1_indices]
                    adj1_y_valid = adj1_y_gpu[valid_adj1_indices]
                    adj1_obstacles[valid_adj1_indices] = self.obstacle_map_gpu[adj1_x_valid, adj1_y_valid]
                
                if len(valid_adj2_indices) > 0:
                    adj2_x_valid = adj2_x_gpu[valid_adj2_indices]
                    adj2_y_valid = adj2_y_gpu[valid_adj2_indices]
                    adj2_obstacles[valid_adj2_indices] = self.obstacle_map_gpu[adj2_x_valid, adj2_y_valid]
                
                # Combine all checks on GPU using vectorized CUDA operations
                # Valid if: not obstacle AND (distance OK) AND (not diagonal OR diagonal corners OK)
                valid_mask = ~obstacles_gpu  # Not an obstacle
                
                # Apply distance field check if enabled (all on GPU)
                if dist_values_gpu is not None:
                    valid_mask = valid_mask & (dist_values_gpu >= self.min_clearance_norm)
                
                # Apply diagonal corner check (all on GPU)
                # For diagonal moves, both adjacent cells must be free
                diagonal_corner_blocked = is_diagonal & (adj1_obstacles | adj2_obstacles)
                valid_mask = valid_mask & ~diagonal_corner_blocked
                
                # Single GPU-to-CPU transfer at the end (minimize transfers)
                valid_mask_cpu = cp.asnumpy(valid_mask)
                
                # Extract valid neighbors
                for idx, (new_x, new_y, new_cost, motion_step) in enumerate(neighbor_nodes):
                    if valid_mask_cpu[idx]:
                        valid_neighbors.append((new_x, new_y, new_cost))
                    
            except Exception as e:
                # Fallback to individual checks if batch fails
                for new_x, new_y, new_cost, motion_step in neighbor_nodes:
                    node = self.Node(new_x, new_y, new_cost, c_id)
                    if self.verify_node(node, current):
                        # Distance field check
                        if self.use_distance_cost and self.distance_field_norm is not None:
                            px = self.calc_grid_position(new_x, self.min_x)
                            py = self.calc_grid_position(new_y, self.min_y)
                            df_width, df_height = self.distance_field_norm.shape
                            maze_x = int(np.clip(px, 0, df_width - 1))
                            maze_y = int(np.clip(py, 0, df_height - 1))
                            if 0 <= maze_x < df_width and 0 <= maze_y < df_height:
                                dist_value = self.distance_field_norm[maze_x, maze_y]
                                if dist_value < self.min_clearance_norm:
                                    continue
                        valid_neighbors.append((new_x, new_y, new_cost))
        else:
            # CPU path: verify neighbors individually
            for new_x, new_y, new_cost, motion_step in neighbor_nodes:
                node = self.Node(new_x, new_y, new_cost, c_id)
                
                # Add clearance check based on distance field (if enabled)
                if self.use_distance_cost and self.distance_field_norm is not None:
                    px = self.calc_grid_position(new_x, self.min_x)
                    py = self.calc_grid_position(new_y, self.min_y)
                    df_width, df_height = self.distance_field_norm.shape
                    maze_x = int(np.clip(px, 0, df_width - 1))
                    maze_y = int(np.clip(py, 0, df_height - 1))
                    if 0 <= maze_x < df_width and 0 <= maze_y < df_height:
                        dist_value = self.distance_field_norm[maze_x, maze_y]
                        if dist_value < self.min_clearance_norm:
                            continue
                
                if self.verify_node(node, current):
                    valid_neighbors.append((new_x, new_y, new_cost))
        
        # Process valid neighbors - compute everything on GPU if available
        if self.use_gpu and GPU_AVAILABLE and GPU_USABLE and len(valid_neighbors) > 0:
            try:
                # Batch compute all values on GPU
                n_valid = len(valid_neighbors)
                valid_x_gpu = cp.array([vn[0] for vn in valid_neighbors], dtype=cp.int32)
                valid_y_gpu = cp.array([vn[1] for vn in valid_neighbors], dtype=cp.int32)
                valid_cost_gpu = cp.array([vn[2] for vn in valid_neighbors], dtype=cp.float32)
                
                # Compute grid indices on GPU
                # grid_index = y * x_width + x
                grid_indices_gpu = valid_y_gpu * self.x_width + valid_x_gpu
                
                # Compute grid positions (world coordinates) on GPU
                grid_pos_x_gpu = valid_x_gpu.astype(cp.float32) * self.resolution + self.min_x
                grid_pos_y_gpu = valid_y_gpu.astype(cp.float32) * self.resolution + self.min_y
                
                # Compute goal position on GPU
                goal_x_gpu = cp.array([goal_node.x] * n_valid, dtype=cp.int32)
                goal_y_gpu = cp.array([goal_node.y] * n_valid, dtype=cp.int32)
                goal_pos_x_gpu = goal_x_gpu.astype(cp.float32) * self.resolution + self.min_x
                goal_pos_y_gpu = goal_y_gpu.astype(cp.float32) * self.resolution + self.min_y
                
                # Compute heuristics on GPU (Euclidean distance)
                dx_gpu = grid_pos_x_gpu - goal_pos_x_gpu
                dy_gpu = grid_pos_y_gpu - goal_pos_y_gpu
                h_cost_gpu = cp.sqrt(dx_gpu * dx_gpu + dy_gpu * dy_gpu)
                
                # Compute f_cost on GPU
                f_cost_gpu = valid_cost_gpu + h_cost_gpu
                
                # Transfer results to CPU in one batch
                grid_indices_cpu = cp.asnumpy(grid_indices_gpu).astype(int)
                f_cost_cpu = cp.asnumpy(f_cost_gpu)
                h_cost_cpu = cp.asnumpy(h_cost_gpu)
                valid_cost_cpu = cp.asnumpy(valid_cost_gpu)
                
                # Process each neighbor with GPU-computed values
                for idx, (new_x, new_y, new_cost) in enumerate(valid_neighbors):
                    n_id = int(grid_indices_cpu[idx])
                    
                    if n_id in closed_set:
                        continue
                    
                    node = self.Node(new_x, new_y, valid_cost_cpu[idx], c_id)
                    f_cost = float(f_cost_cpu[idx])
                    
                    if n_id not in open_set_dict:
                        # New node discovered
                        open_set_dict[n_id] = node
                        heapq.heappush(open_heap, (f_cost, valid_cost_cpu[idx], n_id, node))
                    else:
                        # Update if we found a better path
                        existing_node = open_set_dict[n_id]
                        if existing_node.cost > valid_cost_cpu[idx]:
                            open_set_dict[n_id] = node
                            heapq.heappush(open_heap, (f_cost, valid_cost_cpu[idx], n_id, node))
                            
            except Exception as e:
                # Fallback to CPU computation
                for new_x, new_y, new_cost in valid_neighbors:
                    node = self.Node(new_x, new_y, new_cost, c_id)
                    n_id = self.calc_grid_index(node)

                    if n_id in closed_set:
                        continue

                    # Calculate heuristic and f_cost
                    h_cost = self.calc_heuristic(goal_node, node)
                    f_cost = new_cost + h_cost

                    if n_id not in open_set_dict:
                        # New node discovered
                        open_set_dict[n_id] = node
                        heapq.heappush(open_heap, (f_cost, new_cost, n_id, node))
                    else:
                        # Update if we found a better path
                        existing_node = open_set_dict[n_id]
                        if existing_node.cost > new_cost:
                            open_set_dict[n_id] = node
                            heapq.heappush(open_heap, (f_cost, new_cost, n_id, node))
        else:
            # CPU path: compute individually
            for new_x, new_y, new_cost in valid_neighbors:
                node = self.Node(new_x, new_y, new_cost, c_id)
                n_id = self.calc_grid_index(node)

                if n_id in closed_set:
                    continue

                # Calculate heuristic and f_cost
                h_cost = self.calc_heuristic(goal_node, node)
                f_cost = new_cost + h_cost

                if n_id not in open_set_dict:
                    # New node discovered
                    open_set_dict[n_id] = node
                    heapq.heappush(open_heap, (f_cost, new_cost, n_id, node))
                else:
                    # Update if we found a better path
                    existing_node = open_set_dict[n_id]
                    if existing_node.cost > new_cost:
                        open_set_dict[n_id] = node
                        heapq.heappush(open_heap, (f_cost, new_cost, n_id, node))
    
    def _process_batch_nodes_gpu(self, batch_nodes, goal_node, open_heap, open_set_dict, closed_set):
        """Process neighbors for multiple nodes in parallel on GPU"""
        # Collect all neighbors from all nodes in the batch
        all_neighbor_data = []  # List of (node_idx, new_x, new_y, new_cost, parent_c_id)
        
        for batch_idx, (f_cost, g_cost, c_id, current) in enumerate(batch_nodes):
            for motion_step in self.motion:
                new_x = current.x + motion_step[0]
                new_y = current.y + motion_step[1]
                new_cost = current.cost + motion_step[2]
                
                # Bounds check
                if new_x < 0 or new_x >= self.x_width or new_y < 0 or new_y >= self.y_width:
                    continue
                
                all_neighbor_data.append((batch_idx, new_x, new_y, new_cost, c_id, current.x, current.y))
        
        if len(all_neighbor_data) == 0:
            return
        
        try:
            # Prepare all data on GPU
            n_neighbors = len(all_neighbor_data)
            x_coords = cp.array([nd[1] for nd in all_neighbor_data], dtype=cp.int32)
            y_coords = cp.array([nd[2] for nd in all_neighbor_data], dtype=cp.int32)
            parent_x_coords = cp.array([nd[5] for nd in all_neighbor_data], dtype=cp.int32)
            parent_y_coords = cp.array([nd[6] for nd in all_neighbor_data], dtype=cp.int32)
            costs = cp.array([nd[3] for nd in all_neighbor_data], dtype=cp.float32)
            
            # Vectorized obstacle check on GPU
            obstacles_gpu = self.obstacle_map_gpu[x_coords, y_coords]
            
            # Distance field check if enabled
            dist_values_gpu = None
            if self.use_distance_cost and self.distance_field_norm_gpu is not None:
                df_width, df_height = self.distance_field_norm.shape
                px_gpu = x_coords.astype(cp.float32) * self.resolution + self.min_x
                py_gpu = y_coords.astype(cp.float32) * self.resolution + self.min_y
                df_x_gpu = cp.clip(px_gpu, 0, df_width - 1).astype(cp.int32)
                df_y_gpu = cp.clip(py_gpu, 0, df_height - 1).astype(cp.int32)
                dist_values_gpu = self.distance_field_norm_gpu[df_x_gpu, df_y_gpu]
            
            # Diagonal corner check
            dx_gpu = x_coords - parent_x_coords
            dy_gpu = y_coords - parent_y_coords
            is_diagonal = (dx_gpu != 0) & (dy_gpu != 0)
            
            adj1_x_gpu = parent_x_coords + dx_gpu
            adj1_y_gpu = parent_y_coords
            adj2_x_gpu = parent_x_coords
            adj2_y_gpu = parent_y_coords + dy_gpu
            
            adj1_valid = (adj1_x_gpu >= 0) & (adj1_x_gpu < self.x_width) & \
                         (adj1_y_gpu >= 0) & (adj1_y_gpu < self.y_width)
            adj2_valid = (adj2_x_gpu >= 0) & (adj2_x_gpu < self.x_width) & \
                         (adj2_y_gpu >= 0) & (adj2_y_gpu < self.y_width)
            
            adj1_obstacles = cp.zeros(n_neighbors, dtype=cp.bool_)
            adj2_obstacles = cp.zeros(n_neighbors, dtype=cp.bool_)
            
            valid_adj1_indices = cp.where(adj1_valid)[0]
            valid_adj2_indices = cp.where(adj2_valid)[0]
            
            if len(valid_adj1_indices) > 0:
                adj1_obstacles[valid_adj1_indices] = self.obstacle_map_gpu[
                    adj1_x_gpu[valid_adj1_indices], adj1_y_gpu[valid_adj1_indices]]
            
            if len(valid_adj2_indices) > 0:
                adj2_obstacles[valid_adj2_indices] = self.obstacle_map_gpu[
                    adj2_x_gpu[valid_adj2_indices], adj2_y_gpu[valid_adj2_indices]]
            
            # Combine all checks
            valid_mask = ~obstacles_gpu
            
            if dist_values_gpu is not None:
                valid_mask = valid_mask & (dist_values_gpu >= self.min_clearance_norm)
            
            diagonal_corner_blocked = is_diagonal & (adj1_obstacles | adj2_obstacles)
            valid_mask = valid_mask & ~diagonal_corner_blocked
            
            # Transfer to CPU
            valid_mask_cpu = cp.asnumpy(valid_mask)
            costs_cpu = cp.asnumpy(costs)
            
            # Compute heuristics and f_costs on GPU for valid neighbors
            valid_indices = np.where(valid_mask_cpu)[0]
            if len(valid_indices) > 0:
                valid_x = x_coords[valid_indices]
                valid_y = y_coords[valid_indices]
                valid_costs = costs_cpu[valid_indices]
                
                # Compute grid positions and heuristics on GPU
                grid_pos_x_gpu = valid_x.astype(cp.float32) * self.resolution + self.min_x
                grid_pos_y_gpu = valid_y.astype(cp.float32) * self.resolution + self.min_y
                goal_pos_x = goal_node.x * self.resolution + self.min_x
                goal_pos_y = goal_node.y * self.resolution + self.min_y
                
                dx_gpu = grid_pos_x_gpu - goal_pos_x
                dy_gpu = grid_pos_y_gpu - goal_pos_y
                h_cost_gpu = cp.sqrt(dx_gpu * dx_gpu + dy_gpu * dy_gpu)
                f_cost_gpu = cp.asarray(valid_costs) + h_cost_gpu
                
                # Transfer to CPU
                grid_indices_gpu = valid_y * self.x_width + valid_x
                grid_indices_cpu = cp.asnumpy(grid_indices_gpu).astype(int)
                f_cost_cpu = cp.asnumpy(f_cost_gpu)
                valid_costs_cpu = cp.asnumpy(cp.asarray(valid_costs))
                
                # Process valid neighbors
                for idx, valid_idx in enumerate(valid_indices):
                    neighbor_data = all_neighbor_data[valid_idx]
                    batch_idx, new_x, new_y, new_cost, parent_c_id = neighbor_data[0], neighbor_data[1], neighbor_data[2], neighbor_data[3], neighbor_data[4]
                    
                    n_id = int(grid_indices_cpu[idx])
                    
                    if n_id in closed_set:
                        continue
                    
                    node = self.Node(new_x, new_y, valid_costs_cpu[idx], parent_c_id)
                    f_cost = float(f_cost_cpu[idx])
                    
                    if n_id not in open_set_dict:
                        open_set_dict[n_id] = node
                        heapq.heappush(open_heap, (f_cost, valid_costs_cpu[idx], n_id, node))
                    else:
                        existing_node = open_set_dict[n_id]
                        if existing_node.cost > valid_costs_cpu[idx]:
                            open_set_dict[n_id] = node
                            heapq.heappush(open_heap, (f_cost, valid_costs_cpu[idx], n_id, node))
        
        except Exception as e:
            # Fallback to single node processing
            for f_cost, g_cost, c_id, current in batch_nodes:
                self._process_single_node(current, c_id, goal_node, open_heap, open_set_dict, closed_set)


class PathFinder:
    """Pathfinder that uses A* algorithm with Catmull-Rom spline smoothing"""
    
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.original_path = []  # Store original A* path for visualization
        self.explored_nodes = []
        self.open_set_history = []
        self.found_path = False
        
    def _smooth_path_with_spline(self, path, num_points_per_segment=20):
        """
        Apply Catmull-Rom spline smoothing to the A* path
        
        Args:
            path: List of (x, y) tuples representing the A* path
            num_points_per_segment: Number of points to generate per spline segment
        
        Returns:
            List of (x, y) tuples representing the smoothed path
        """
        if len(path) < 2:
            return path
        
        # If path is too short, just return it
        if len(path) == 2:
            # Interpolate between two points
            p0, p1 = path[0], path[1]
            t_vals = np.linspace(0, 1, num_points_per_segment)
            smoothed = []
            for t in t_vals:
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                smoothed.append((x, y))
            return smoothed
        
        # Convert path to numpy array for spline processing
        control_points = np.array(path)
        
        # Calculate total number of points for the smoothed path
        # Use more points for longer paths to maintain smoothness
        total_segments = len(path) - 1
        total_points = total_segments * num_points_per_segment
        
        # Apply Catmull-Rom spline
        # The function returns a transposed array: shape (2, N) where [0] is x and [1] is y
        spline_result = catmull_rom_spline(control_points, num_points_per_segment)
        
        # Extract x and y coordinates
        # spline_result is shape (2, N) after transpose
        if spline_result.shape[0] == 2:
            spline_x, spline_y = spline_result[0], spline_result[1]
        else:
            # Fallback: if shape is (N, 2), transpose it
            spline_result = spline_result.T
            spline_x, spline_y = spline_result[0], spline_result[1]
        
        # Convert back to list of tuples
        smoothed_path = list(zip(spline_x, spline_y))
        
        # Ensure start and end points are exactly as in original path
        if smoothed_path:
            smoothed_path[0] = path[0]
            smoothed_path[-1] = path[-1]
        
        return smoothed_path
    
    def _check_path_collision(self, path):
        """
        Check if smoothed path collides with walls
        Returns True if path is safe, False if it collides
        """
        for x, y in path:
            # Check if point is within maze bounds
            if x < 0 or x >= self.maze.width or y < 0 or y >= self.maze.height:
                return False
            
            # Check if point is in a wall cell
            wall_x, wall_y = int(round(x)), int(round(y))
            if self.maze.is_wall(wall_x, wall_y):
                return False
            
            # Check nearby cells for safety margin
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    check_x, check_y = wall_x + dx, wall_y + dy
                    if self.maze.is_wall(check_x, check_y):
                        # Check distance to wall
                        dist = math.sqrt((x - check_x)**2 + (y - check_y)**2)
                        if dist < ROBOT_RADIUS:
                            return False
        
        return True
        
    def find_path(self, use_distance_cost=None, min_clearance_norm=None, num_points_per_segment=None, smoothing_enabled=None):
        """
        Use A* to find path from start to exit, then smooth with Catmull-Rom spline
        
        Args:
            use_distance_cost: If True, use distance field to prefer safer paths
            min_clearance_norm: Minimum normalized clearance (0-1). Nodes below this are treated as blocked.
            num_points_per_segment: Number of points per segment for spline smoothing
            smoothing_enabled: If False, skip spline smoothing
        """
        cfg = load_planner_config()
        use_distance_cost = cfg["use_distance_cost"] if use_distance_cost is None else use_distance_cost
        num_points_per_segment = cfg.get("num_points_per_segment", 15) if num_points_per_segment is None else num_points_per_segment
        smoothing_enabled = cfg.get("smoothing_enabled", True) if smoothing_enabled is None else smoothing_enabled
        gamma = cfg.get("distance_heat_gamma", 1.0)
        # Derive clearance from gamma if not explicitly provided
        if min_clearance_norm is None:
            min_clearance_norm = max(0.05, min(0.95, gamma))
        # Get obstacles
        ox, oy = self.maze.get_obstacle_list()
        
        # Get grid resolution from config (for node grid)
        grid_resolution = cfg.get("grid_resolution", CELL_SIZE)
        # Get obstacle map resolution from config (for collision detection)
        obstacle_map_resolution = cfg.get("obstacle_map_resolution", 1.0)
        
        print(f"[A*] Node grid resolution: {grid_resolution} pixels per node")
        print(f"[A*] Obstacle map resolution: {obstacle_map_resolution} pixels per cell (for collision detection)")
        
        if grid_resolution > 1.0:
            print(f"[A*] Each node represents a {grid_resolution}x{grid_resolution} pixel area")
            print(f"[A*] This reduces search space from ~{len(ox)} to ~{int(len(ox) / (grid_resolution * grid_resolution))} nodes")
        
        if obstacle_map_resolution == 1.0:
            print(f"[A*] ✓ Using full-resolution obstacle map for maximum collision sensitivity")
        elif obstacle_map_resolution > grid_resolution:
            print(f"[A*] ⚠ Obstacle map resolution ({obstacle_map_resolution}) > grid resolution ({grid_resolution})")
            print(f"[A*] This provides high collision sensitivity with reduced search space")
        
        # Get distance field if available
        distance_field = None
        if use_distance_cost:
            try:
                distance_field = self.maze.compute_distance_field(use_sdf=False)
                # Need to map it to the planner's grid resolution
                # The planner uses a grid based on obstacle bounds, so we need to interpolate
                print("[A*] Distance field available for safety-aware pathfinding")
            except Exception as e:
                print(f"[A*] Could not compute distance field: {e}")
                distance_field = None
                use_distance_cost = False
        
        # Create A* planner with tracking and distance field
        # grid_resolution: controls node grid size (search space)
        # obstacle_map_resolution: controls obstacle map precision (collision sensitivity)
        planner = AStarPlannerWithTracking(ox, oy, grid_resolution, ROBOT_RADIUS,
                                          distance_field=distance_field,
                                          use_distance_cost=use_distance_cost,
                                          min_clearance_norm=min_clearance_norm,
                                          obstacle_map_resolution=obstacle_map_resolution)
        
        # Find path
        sx, sy = self.maze.start
        gx, gy = self.maze.exit
        
        rx, ry = planner.planning(sx, sy, gx, gy)
        
        if rx and ry:
            # Store original A* path
            self.original_path = list(zip(rx, ry))
            self.original_path.reverse()  # Reverse to go from start to goal
            
            # Apply Catmull-Rom spline smoothing
            if smoothing_enabled:
                print("[Spline] Applying Catmull-Rom spline smoothing to A* path...")
                smoothed_path = self._smooth_path_with_spline(self.original_path, num_points_per_segment=num_points_per_segment)
                
                # Check if smoothed path is collision-free
                if self._check_path_collision(smoothed_path):
                    print(f"[Spline] ✓ Smoothed path is collision-free ({len(smoothed_path)} points)")
                    self.current_path = smoothed_path
                else:
                    print("[Spline] ⚠ Smoothed path has collisions, using original A* path")
                    self.current_path = self.original_path
            else:
                self.current_path = self.original_path
            
            self.explored_nodes = planner.explored_nodes
            self.open_set_history = planner.open_set_history
            self.found_path = True
            return True
        return False
    


class MazeGame:
    """Main game class with visualization"""
    
    def __init__(self, maze=None, show_distance_map=None):
        cfg = load_planner_config()
        if show_distance_map is None:
            show_distance_map = cfg.get("show_distance_map", True)
        
        if maze is None:
            self.maze = create_maze()
        else:
            self.maze = maze
        self.pathfinder = PathFinder(self.maze)
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.phase = 'moving'  # Skip exploration, show path immediately
        self.show_distance_map = show_distance_map
        self.distance_field = None
        self._colorbar_added = False
        
        # Compute distance field if enabled
        if self.show_distance_map:
            try:
                print("[Distance Map] Computing distance field...")
                self.distance_field = self.maze.compute_distance_field(use_sdf=False)  # Use UDF for visualization
                print("[Distance Map] ✓ Distance field computed successfully")
            except Exception as e:
                print(f"[Distance Map] ✗ Failed to compute distance field: {e}")
                self.show_distance_map = False
    
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
        
    def draw_maze(self, show_exploration=True):
        """Draw the maze"""
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.maze.width - 0.5)
        self.ax.set_ylim(-0.5, self.maze.height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        
        # Draw distance field as background (if enabled)
        if self.show_distance_map and self.distance_field is not None:
            cfg = load_planner_config()
            gamma = cfg.get("distance_heat_gamma", 1.0)
            df = self.distance_field
            df_norm = (df - df.min()) / (df.max() - df.min() + 1e-8)
            df_vis = np.power(df_norm, gamma)
            # Transpose for correct orientation (distance_field is width x height, but imshow expects height x width)
            distance_plot = self.ax.imshow(
                df_vis.T,
                extent=[-0.5, self.maze.width - 0.5, -0.5, self.maze.height - 0.5],
                origin='lower',
                cmap='viridis',
                alpha=0.3,
                interpolation='bilinear',
                zorder=0
            )
            # Add colorbar
            if not self._colorbar_added:
                plt.colorbar(distance_plot, ax=self.ax, label='Distance to Nearest Obstacle', shrink=0.8)
                self._colorbar_added = True
        
        # Draw walls as continuous rectangles
        self._draw_walls_as_rectangles()
        
        # Draw start position (bold circle)
        sx, sy = self.maze.start
        self.ax.scatter(sx, sy, c='green', s=280, marker='o',
                        edgecolors='black', linewidths=2.5, zorder=5)
        
        # Draw exit position (bold circle)
        ex, ey = self.maze.exit
        self.ax.scatter(ex, ey, c='red', s=320, marker='o',
                        edgecolors='black', linewidths=2.5, zorder=5)
        
        # Draw all explored nodes (always show when path is found)
        if self.pathfinder.found_path and self.pathfinder.explored_nodes:
            explored = self.pathfinder.explored_nodes
            ex_x = [p[0] for p in explored]
            ex_y = [p[1] for p in explored]
            self.ax.scatter(ex_x, ex_y, c='lightblue', s=30, 
                           alpha=0.6, marker='o', label='Explored Nodes', zorder=2)
        
        # Draw all open set nodes from the final state (always show when path is found)
        if self.pathfinder.found_path and self.pathfinder.open_set_history:
            # Get the final open set (last non-empty one)
            final_open_set = None
            for open_set in reversed(self.pathfinder.open_set_history):
                if open_set:
                    final_open_set = open_set
                    break
            
            if final_open_set:
                open_x = [p[0] for p in final_open_set]
                open_y = [p[1] for p in final_open_set]
                self.ax.scatter(open_x, open_y, c='yellow', s=50, 
                               alpha=0.7, marker='*', label='Open Set', zorder=3)
        
        # Draw final path (always show when found)
        if self.pathfinder.found_path and self.pathfinder.current_path:
            # Draw original A* path (if different from smoothed path)
            if hasattr(self.pathfinder, 'original_path') and self.pathfinder.original_path:
                if len(self.pathfinder.original_path) != len(self.pathfinder.current_path):
                    orig_x = [p[0] for p in self.pathfinder.original_path]
                    orig_y = [p[1] for p in self.pathfinder.original_path]
                    self.ax.plot(orig_x, orig_y, 'g--', linewidth=2, alpha=0.4, 
                               label='A* Path (Original)', zorder=3)
            
            # Draw smoothed spline path
            path_x = [p[0] for p in self.pathfinder.current_path]
            path_y = [p[1] for p in self.pathfinder.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, 
                        label='Smooth Path (Catmull-Rom)', zorder=4)
        
        # Update title
        if self.pathfinder.found_path:
            title = f'Maze Game - A* Pathfinding with Catmull-Rom Spline (Path Found: {len(self.pathfinder.current_path)} steps)'
        else:
            title = f'Maze Game - A* Pathfinding with Catmull-Rom Spline'
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
    
    
    def run(self):
        """Run the game"""
        print("Finding path using A* algorithm...")
        success = self.pathfinder.find_path()
        
        if not success:
            print("No path found!")
            self.draw_maze()
            plt.show()
            return
        
        print(f"Path found! Path length: {len(self.pathfinder.current_path)} steps")
        print(f"Explored {len(self.pathfinder.explored_nodes)} nodes")
        print("Displaying visualization...")
        
        # Draw the maze with path (static visualization, no animation)
        self.draw_maze(show_exploration=False)
        
        plt.tight_layout()
        plt.show()


def load_maze_from_maps():
    """Check maps folder for PNG files and load the first one found"""
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    
    if not os.path.exists(maps_dir):
        print(f"Maps directory not found: {maps_dir}")
        return None
    
    # Find all PNG files in maps directory
    png_files = glob.glob(os.path.join(maps_dir, '*.png'))
    png_files.extend(glob.glob(os.path.join(maps_dir, '*.PNG')))
    
    if not png_files:
        print(f"No PNG files found in {maps_dir}")
        return None
    
    # Load config to get detection mode and desired map file
    config = load_planner_config()
    detection_mode = config.get("detection_mode", "canny")
    canny_threshold1 = config.get("canny_threshold1", 100)
    canny_threshold2 = config.get("canny_threshold2", 200)
    sobel_threshold = config.get("sobel_threshold", 50)
    map_file = config.get("map_file")
    
    # Choose image: prefer map_file from config if it exists; otherwise prefer sophisticated_maze.png
    image_path = None
    if map_file:
        candidate = os.path.join(maps_dir, map_file)
        if os.path.exists(candidate):
            image_path = candidate
        else:
            print(f"[Config] map_file '{map_file}' not found in maps/. Falling back to defaults.")
    
    if image_path is None:
        sophisticated_maze = os.path.join(maps_dir, 'sophisticated_maze.png')
        if os.path.exists(sophisticated_maze):
            image_path = sophisticated_maze
        else:
            image_path = png_files[0]
    
    print(f"Loading maze from image: {os.path.basename(image_path)}")
    
    print(f"Using detection mode: {detection_mode}")
    
    try:
        maze = Maze.from_image(image_path, 
                              detection_mode=detection_mode,
                              canny_threshold1=canny_threshold1,
                              canny_threshold2=canny_threshold2,
                              sobel_threshold=sobel_threshold)
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
    print("Maze Game with A* Pathfinding")
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

