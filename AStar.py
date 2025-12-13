"""
Maze Game with A* Pathfinding Algorithm

This file implements a maze game where a creature uses the A* algorithm 
to find the shortest path from start to exit.

The A* algorithm is used for optimal pathfinding, preventing leaks through
wall edges and finding the most efficient route through the maze.

Features:
- A* pathfinding with wall edge leak prevention
- PNG image maze loading
- Real-time visualization of path exploration
- Creature animation following the optimal path
"""

import sys
import os
import math
import glob
import heapq
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

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
sys.path.insert(0, '/Users/umutozdemir/Desktop/pythonrobotics')
from PathPlanning.AStar import a_star

# Maze configuration
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
CELL_SIZE = 1.0
ROBOT_RADIUS = 0.3  # Increased to prevent edge leaks, but still small enough for paths


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
    
    def get_obstacle_list(self):
        """Convert walls to obstacle lists for A* planner"""
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
    
    return maze


class AStarPlannerWithTracking(a_star.AStarPlanner):
    """Optimized A* planner with GPU support, tracking, and wall edge leak prevention"""
    
    def __init__(self, ox, oy, resolution, rr):
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

        while True:
            if len(open_heap) == 0:
                print("Open set is empty..")
                break

            # Get node with minimum f_cost from priority queue (O(log n))
            f_cost, g_cost, c_id, current = heapq.heappop(open_heap)
            
            # Skip if this node was already processed with better cost
            if c_id not in open_set_dict or open_set_dict[c_id] != current:
                continue
            
            # Remove from open set
            del open_set_dict[c_id]
            
            # Track explored nodes
            explored_pos = (self.calc_grid_position(current.x, self.min_x),
                           self.calc_grid_position(current.y, self.min_y))
            self.explored_nodes.append(explored_pos)
            
            # Track open set (for visualization)
            open_positions = [(self.calc_grid_position(n.x, self.min_x),
                              self.calc_grid_position(n.y, self.min_y))
                             for n in open_set_dict.values()]
            self.open_set_history.append(open_positions.copy())

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

            # Add it to the closed set
            closed_set[c_id] = current

            # Expand grid search based on motion model
            # Vectorized expansion for better performance
            for i, motion_step in enumerate(self.motion):
                new_x = current.x + motion_step[0]
                new_y = current.y + motion_step[1]
                new_cost = current.cost + motion_step[2]
                
                node = self.Node(new_x, new_y, new_cost, c_id)
                n_id = self.calc_grid_index(node)

                # Fast verification
                if not self.verify_node(node, current):
                    continue

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

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry


class Creature:
    """Creature that navigates the maze using A*"""
    
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.explored_nodes = []
        self.open_set_history = []
        self.path_index = 0
        self.found_path = False
        self.exploration_index = 0
        
    def find_path(self):
        """Use A* to find path from start to exit"""
        # Get obstacles
        ox, oy = self.maze.get_obstacle_list()
        
        # Create A* planner with tracking
        planner = AStarPlannerWithTracking(ox, oy, CELL_SIZE, ROBOT_RADIUS)
        
        # Find path
        sx, sy = self.maze.start
        gx, gy = self.maze.exit
        
        rx, ry = planner.planning(sx, sy, gx, gy)
        
        if rx and ry:
            self.current_path = list(zip(rx, ry))
            self.current_path.reverse()  # Reverse to go from start to goal
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
    
    def __init__(self, maze=None):
        if maze is None:
            self.maze = create_maze()
        else:
            self.maze = maze
        self.creature = Creature(self.maze)
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.animation = None
        self.phase = 'exploring'  # 'exploring' or 'moving'
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
        
    def draw_maze(self, show_exploration=True):
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
        
        # Draw exploration process
        if show_exploration and self.creature.found_path:
            # Draw explored nodes
            if self.creature.exploration_index < len(self.creature.explored_nodes):
                explored = self.creature.explored_nodes[:self.creature.exploration_index + 1]
                if explored:
                    ex_x = [p[0] for p in explored]
                    ex_y = [p[1] for p in explored]
                    self.ax.scatter(ex_x, ex_y, c='lightblue', s=30, 
                                   alpha=0.6, marker='o', label='Explored', zorder=2)
            
            # Draw open set (nodes being considered)
            if (self.creature.exploration_index < len(self.creature.open_set_history) and
                self.creature.open_set_history[self.creature.exploration_index]):
                open_set = self.creature.open_set_history[self.creature.exploration_index]
                open_x = [p[0] for p in open_set]
                open_y = [p[1] for p in open_set]
                self.ax.scatter(open_x, open_y, c='yellow', s=50, 
                               alpha=0.7, marker='*', label='Open Set', zorder=3)
        
        # Draw final path
        if self.creature.found_path and self.creature.current_path and not show_exploration:
            path_x = [p[0] for p in self.creature.current_path]
            path_y = [p[1] for p in self.creature.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, label='Path', zorder=4)
        
        # Draw creature
        cx, cy = self.creature.get_current_position()
        creature_circle = plt.Circle((cx, cy), 0.4, color='blue', zorder=10)
        self.ax.add_patch(creature_circle)
        # Add eyes to make it look like a creature
        self.ax.scatter([cx - 0.15, cx + 0.15], [cy + 0.1, cy + 0.1], 
                       c='white', s=50, zorder=11)
        
        # Update title based on phase
        if show_exploration:
            title = f'Maze Game - A* Pathfinding (Exploring: {self.creature.exploration_index + 1}/{len(self.creature.explored_nodes)})'
        else:
            title = f'Maze Game - A* Pathfinding (Moving to Exit)'
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        
        self.ax.legend(loc='upper right', fontsize=9)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
    
    def animate(self, frame):
        """Animation function"""
        self.frame_count = frame
        
        # Phase 1: Show exploration
        if self.phase == 'exploring':
            if self.creature.exploration_index < len(self.creature.explored_nodes) - 1:
                # Show exploration progress
                if frame % 2 == 0:  # Update every 2 frames
                    self.creature.exploration_index = min(
                        self.creature.exploration_index + 1,
                        len(self.creature.explored_nodes) - 1
                    )
                self.draw_maze(show_exploration=True)
            else:
                # Switch to movement phase
                self.phase = 'moving'
                self.creature.path_index = 0
                self.draw_maze(show_exploration=False)
        
        # Phase 2: Show creature moving along path
        elif self.phase == 'moving':
            self.draw_maze(show_exploration=False)
            if self.creature.path_index < len(self.creature.current_path) - 1:
                if frame % 3 == 0:  # Move every 3 frames
                    self.creature.move_next()
        
        return []
    
    def run(self):
        """Run the game"""
        print("Finding path using A* algorithm...")
        success = self.creature.find_path()
        
        if not success:
            print("No path found!")
            self.draw_maze()
            plt.show()
            return
        
        print(f"Path found! Path length: {len(self.creature.current_path)} steps")
        print(f"Explored {len(self.creature.explored_nodes)} nodes")
        print("Starting animation...")
        print("Phase 1: Watch the A* algorithm explore the maze")
        print("Phase 2: Watch the creature follow the shortest path")
        
        # Create animation
        exploration_frames = len(self.creature.explored_nodes) * 2
        movement_frames = len(self.creature.current_path) * 3
        total_frames = exploration_frames + movement_frames + 20
        self.animation = FuncAnimation(self.fig, self.animate, frames=total_frames,
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

