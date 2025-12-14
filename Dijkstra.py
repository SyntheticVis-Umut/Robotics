"""
Maze Game with Dijkstra Pathfinding Algorithm and Catmull-Rom Spline Smoothing

This file implements a maze game where a creature uses the Dijkstra algorithm 
to find the shortest path from start to exit, then applies Catmull-Rom spline
smoothing for a smoother, more natural movement trajectory.

The Dijkstra algorithm is used for optimal pathfinding, finding the most 
efficient route through the maze without heuristics. The path is then smoothed
using Catmull-Rom spline interpolation for better aesthetics and more realistic
creature movement.

Features:
- Dijkstra pathfinding with wall edge leak prevention
- Catmull-Rom spline path smoothing
- Collision detection for smoothed paths
- PNG image maze loading
- Real-time visualization of path exploration
- Creature animation following the smoothed optimal path
"""

import sys
import os
import math
import glob
import heapq
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# Add the pythonrobotics path to import Dijkstra algorithm
sys.path.insert(0, '/Users/umutozdemir/Desktop/pythonrobotics')
from PathPlanning.Dijkstra import dijkstra
from PathPlanning.Catmull_RomSplinePath.catmull_rom_spline_path import catmull_rom_spline
from Mapping.DistanceMap.distance_map import compute_udf_scipy, compute_sdf_scipy

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
ROBOT_RADIUS = 0.3  # Increased to prevent edge leaks, but still small enough for paths

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "path_config.json")


def load_planner_config():
    """Load planner configuration from JSON; fall back to defaults if missing/invalid."""
    defaults = {
        "use_distance_cost": True,
        "min_clearance_norm": 0.50,
        "num_points_per_segment": 15,
        "show_distance_map": True,
        "smoothing_enabled": True,
        "distance_heat_gamma": 1.0,
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
    
    def get_valid_positions(self):
        """Get all valid (non-wall) positions in the maze"""
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
        """Convert walls to obstacle lists for Dijkstra planner"""
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
        """
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


class DijkstraPlannerWithTracking(dijkstra.DijkstraPlanner):
    """Optimized Dijkstra planner with GPU support, tracking, wall edge leak prevention, and distance field cost"""
    
    def __init__(self, ox, oy, resolution, rr, distance_field=None, use_distance_cost=True, min_clearance_norm=0.50):
        # Temporarily disable animation in parent class
        original_show = dijkstra.show_animation
        dijkstra.show_animation = False
        super().__init__(ox, oy, resolution, rr)
        dijkstra.show_animation = original_show
        
        self.explored_nodes = []
        self.open_set_history = []
        
        # Pre-compute motion model as numpy array for vectorization
        self.motion_array = np.array(self.motion, dtype=np.float32)
        
        # Distance field for safety-aware pathfinding
        # Store original distance field and maze info for mapping
        self.distance_field = distance_field
        self.use_distance_cost = use_distance_cost and (distance_field is not None)
        self.min_clearance_norm = min_clearance_norm  # Hard cutoff for minimum normalized clearance
        
        if self.use_distance_cost:
            print(f"[Dijkstra] ✓ Distance field cost enabled")
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
        
        # Use GPU for obstacle map if available and usable
        self.use_gpu = False
        if GPU_AVAILABLE and GPU_USABLE:
            try:
                self.obstacle_map_gpu = cp.asarray(self.obstacle_map)
                self.use_gpu = True
                print(f"[GPU] ✓ Using GPU for obstacle map processing")
                print(f"[GPU] Obstacle map size: {self.x_width}x{self.y_width}")
            except Exception as e:
                self.use_gpu = False
                print(f"[GPU] ✗ Failed to initialize GPU obstacle map: {e}")
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
        Uses optimized JIT-compiled version if available.
        """
        parent_x = parent_node.x if parent_node is not None else None
        parent_y = parent_node.y if parent_node is not None else None
        
        if NUMBA_AVAILABLE:
            return self.verify_node_fast(node.x, node.y, self.x_width, self.y_width,
                                        self.obstacle_map_np, parent_x, parent_y)
        else:
            # Fallback to original method
            px = self.calc_position(node.x, self.min_x)
            py = self.calc_position(node.y, self.min_y)

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
        """Optimized Dijkstra path search with exploration tracking and priority queue"""
        print(f"[Dijkstra] Starting pathfinding from ({sx:.1f}, {sy:.1f}) to ({gx:.1f}, {gy:.1f})")
        if self.use_gpu:
            print("[Dijkstra] Using GPU-accelerated pathfinding")
        elif NUMBA_AVAILABLE:
            print("[Dijkstra] Using CPU pathfinding with Numba JIT compilation")
        else:
            print("[Dijkstra] Using optimized CPU pathfinding")
        
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        # Use priority queue (heap) instead of dict + min() for O(log n) operations
        # Format: (cost, node_id, node)
        open_heap = []
        open_set_dict = {}  # For O(1) lookup
        closed_set = {}
        
        start_id = self.calc_index(start_node)
        heapq.heappush(open_heap, (0.0, start_id, start_node))
        open_set_dict[start_id] = start_node
        
        self.explored_nodes = []
        self.open_set_history = []

        while True:
            if len(open_heap) == 0:
                print("[Dijkstra] Open set is empty..")
                break

            # Get node with minimum cost from priority queue (O(log n))
            cost, c_id, current = heapq.heappop(open_heap)
            
            # Skip if this node was already processed with better cost
            if c_id not in open_set_dict or open_set_dict[c_id] != current:
                continue
            
            # Remove from open set
            del open_set_dict[c_id]
            
            # Track explored nodes
            explored_pos = (self.calc_position(current.x, self.min_x),
                           self.calc_position(current.y, self.min_y))
            self.explored_nodes.append(explored_pos)
            
            # Track open set (for visualization)
            open_positions = [(self.calc_position(n.x, self.min_x),
                              self.calc_position(n.y, self.min_y))
                             for n in open_set_dict.values()]
            self.open_set_history.append(open_positions.copy())

            if current.x == goal_node.x and current.y == goal_node.y:
                print(f"[Dijkstra] ✓ Goal found! Explored {len(self.explored_nodes)} nodes")
                if self.use_gpu:
                    print("[Dijkstra] Pathfinding completed using GPU acceleration")
                elif NUMBA_AVAILABLE:
                    print("[Dijkstra] Pathfinding completed using CPU with Numba JIT")
                else:
                    print("[Dijkstra] Pathfinding completed using optimized CPU")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Add it to the closed set
            closed_set[c_id] = current

            # Expand grid search based on motion model
            for move_x, move_y, move_cost in self.motion:
                new_x = current.x + move_x
                new_y = current.y + move_y
                new_cost = current.cost + move_cost
                
                # Add clearance check based on distance field (if enabled)
                if self.use_distance_cost and self.distance_field_norm is not None:
                    # Get position in world coordinates
                    px = self.calc_position(new_x, self.min_x)
                    py = self.calc_position(new_y, self.min_y)
                    
                    # Map world coordinates to distance field grid indices
                    # Distance field is computed on maze grid (0 to width, 0 to height)
                    # We need to map planner's world coordinates to maze grid
                    df_width, df_height = self.distance_field_norm.shape
                    
                    # Convert world coordinates to maze grid indices
                    # Assuming maze grid starts at (0, 0) and matches distance field
                    maze_x = int(np.clip(px, 0, df_width - 1))
                    maze_y = int(np.clip(py, 0, df_height - 1))
                    
                    # Get normalized distance (0 = at wall, 1 = far from wall)
                    if 0 <= maze_x < df_width and 0 <= maze_y < df_height:
                        dist_value = self.distance_field_norm[maze_x, maze_y]
                        # Hard clearance cutoff: skip nodes below threshold
                        if dist_value < self.min_clearance_norm:
                            continue
                
                node = self.Node(new_x, new_y, new_cost, c_id)
                n_id = self.calc_index(node)

                # Fast verification
                if not self.verify_node(node, current):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set_dict:
                    # New node discovered
                    open_set_dict[n_id] = node
                    heapq.heappush(open_heap, (new_cost, n_id, node))
                else:
                    # Update if we found a better path
                    existing_node = open_set_dict[n_id]
                    if existing_node.cost > new_cost:
                        open_set_dict[n_id] = node
                        heapq.heappush(open_heap, (new_cost, n_id, node))

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry


class Creature:
    """Creature that navigates the maze using Dijkstra with Catmull-Rom spline smoothing"""
    
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.original_path = []  # Store original Dijkstra path for visualization
        self.explored_nodes = []
        self.open_set_history = []
        self.path_index = 0
        self.found_path = False
        self.exploration_index = 0
        
    def _smooth_path_with_spline(self, path, num_points_per_segment=20):
        """
        Apply Catmull-Rom spline smoothing to the Dijkstra path
        
        Args:
            path: List of (x, y) tuples representing the Dijkstra path
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
        Use Dijkstra to find path from start to exit, then smooth with Catmull-Rom spline
        
        Args:
            use_distance_cost: If True, use distance field to prefer safer paths
            min_clearance_norm: Minimum normalized clearance (0-1). Nodes below this are treated as blocked.
            num_points_per_segment: Number of points per segment for spline smoothing
            smoothing_enabled: If False, skip spline smoothing
        """
        cfg = load_planner_config()
        use_distance_cost = cfg["use_distance_cost"] if use_distance_cost is None else use_distance_cost
        gamma = cfg.get("distance_heat_gamma", 1.0)
        if min_clearance_norm is None:
            min_clearance_norm = max(0.05, min(0.95, gamma))
        num_points_per_segment = cfg.get("num_points_per_segment", 15) if num_points_per_segment is None else num_points_per_segment
        smoothing_enabled = cfg.get("smoothing_enabled", True) if smoothing_enabled is None else smoothing_enabled
        # Get obstacles
        ox, oy = self.maze.get_obstacle_list()
        
        # Get distance field if available
        distance_field = None
        if use_distance_cost:
            try:
                distance_field = self.maze.compute_distance_field(use_sdf=False)
                # Need to map it to the planner's grid resolution
                # The planner uses a grid based on obstacle bounds, so we need to interpolate
                print("[Dijkstra] Distance field available for safety-aware pathfinding")
            except Exception as e:
                print(f"[Dijkstra] Could not compute distance field: {e}")
                distance_field = None
                use_distance_cost = False
        
        # Create Dijkstra planner with tracking and distance field
        planner = DijkstraPlannerWithTracking(ox, oy, CELL_SIZE, ROBOT_RADIUS,
                                             distance_field=distance_field,
                                             use_distance_cost=use_distance_cost,
                                             min_clearance_norm=min_clearance_norm)
        
        # Find path
        sx, sy = self.maze.start
        gx, gy = self.maze.exit
        
        rx, ry = planner.planning(sx, sy, gx, gy)
        
        if rx and ry:
            # Store original Dijkstra path
            self.original_path = list(zip(rx, ry))
            self.original_path.reverse()  # Reverse to go from start to goal
            
            if smoothing_enabled:
                # Apply Catmull-Rom spline smoothing
                print("[Spline] Applying Catmull-Rom spline smoothing to Dijkstra path...")
                smoothed_path = self._smooth_path_with_spline(self.original_path, num_points_per_segment=num_points_per_segment)
                
                # Check if smoothed path is collision-free
                if self._check_path_collision(smoothed_path):
                    print(f"[Spline] ✓ Smoothed path is collision-free ({len(smoothed_path)} points)")
                    self.current_path = smoothed_path
                else:
                    print("[Spline] ⚠ Smoothed path has collisions, using original Dijkstra path")
                    self.current_path = self.original_path
            else:
                self.current_path = self.original_path
            
            self.explored_nodes = planner.explored_nodes
            self.open_set_history = planner.open_set_history
            self.found_path = True
            return True
        return False
    
    def get_current_position(self):
        """Get current position of creature"""
        if self.current_path and self.path_index < len(self.current_path):
            return self.current_path[self.path_index]
        return self.maze.start
    
    def move_next(self):
        """Move to next position in path"""
        if self.path_index < len(self.current_path) - 1:
            self.path_index += 1
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
        self.creature = Creature(self.maze)
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.animation = None
        self.phase = 'moving'  # Skip exploration, show path immediately
        self.frame_count = 0
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
        if self.creature.found_path and self.creature.explored_nodes:
            explored = self.creature.explored_nodes
            ex_x = [p[0] for p in explored]
            ex_y = [p[1] for p in explored]
            self.ax.scatter(ex_x, ex_y, c='lightblue', s=30, 
                           alpha=0.6, marker='o', label='Explored Nodes', zorder=2)
        
        # Draw all open set nodes from the final state (always show when path is found)
        if self.creature.found_path and self.creature.open_set_history:
            # Get the final open set (last non-empty one)
            final_open_set = None
            for open_set in reversed(self.creature.open_set_history):
                if open_set:
                    final_open_set = open_set
                    break
            
            if final_open_set:
                open_x = [p[0] for p in final_open_set]
                open_y = [p[1] for p in final_open_set]
                self.ax.scatter(open_x, open_y, c='yellow', s=50, 
                               alpha=0.7, marker='*', label='Open Set', zorder=3)
        
        # Draw final path (always show when found)
        if self.creature.found_path and self.creature.current_path:
            # Draw original Dijkstra path (if different from smoothed path)
            if hasattr(self.creature, 'original_path') and self.creature.original_path:
                if len(self.creature.original_path) != len(self.creature.current_path):
                    orig_x = [p[0] for p in self.creature.original_path]
                    orig_y = [p[1] for p in self.creature.original_path]
                    self.ax.plot(orig_x, orig_y, 'g--', linewidth=2, alpha=0.4, 
                               label='Dijkstra Path (Original)', zorder=3)
            
            # Draw smoothed spline path
            path_x = [p[0] for p in self.creature.current_path]
            path_y = [p[1] for p in self.creature.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.8, 
                        label='Smooth Path (Catmull-Rom)', zorder=4)
        
        # Draw creature as simple circle
        cx, cy = self.creature.get_current_position()
        creature_circle = plt.Circle((cx, cy), 0.4, color='blue', zorder=10, ec='black', lw=1.5)
        self.ax.add_patch(creature_circle)
        
        # Update title
        if self.creature.found_path:
            title = f'Maze Game - Dijkstra Pathfinding with Catmull-Rom Spline (Path Found: {len(self.creature.current_path)} steps)'
        else:
            title = f'Maze Game - Dijkstra Pathfinding with Catmull-Rom Spline'
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
    
    def animate(self, frame):
        """Animation function - show creature moving along path"""
        self.frame_count = frame
        
        # Show creature moving along path (path is already drawn)
        self.draw_maze(show_exploration=False)
        if self.creature.path_index < len(self.creature.current_path) - 1:
            if frame % 3 == 0:  # Move every 3 frames
                self.creature.move_next()
        
        return []
    
    def run(self):
        """Run the game"""
        print("Finding path using Dijkstra algorithm...")
        success = self.creature.find_path()
        
        if not success:
            print("[Dijkstra] No path found!")
            self.draw_maze()
            plt.show()
            return
        
        print(f"[Dijkstra] Path found! Path length: {len(self.creature.current_path)} steps")
        print(f"[Dijkstra] Explored {len(self.creature.explored_nodes)} nodes")
        print("Displaying visualization...")
        
        # Draw the maze with path immediately
        self.draw_maze(show_exploration=False)
        
        # Create animation for creature movement only
        movement_frames = len(self.creature.current_path) * 3 + 20
        self.animation = FuncAnimation(self.fig, self.animate, frames=movement_frames,
                                      interval=50, repeat=True, blit=False)
        
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
    
    # Prefer sophisticated_maze.png if it exists, otherwise use the first PNG file found
    sophisticated_maze = os.path.join(maps_dir, 'sophisticated_maze.png')
    if os.path.exists(sophisticated_maze):
        image_path = sophisticated_maze
    else:
        image_path = png_files[0]
    
    print(f"Loading maze from image: {os.path.basename(image_path)}")
    
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
    print("Maze Game with Dijkstra Pathfinding")
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

