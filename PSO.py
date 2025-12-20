"""
Maze Game with Particle Swarm Optimization (PSO) Pathfinding Algorithm

This file implements a maze game where a creature uses Particle Swarm Optimization 
to find a path from start to exit.

PSO is a population-based optimization algorithm inspired by bird flocking behavior.
Multiple particles explore the search space, sharing information to find optimal paths.

Features:
- PSO pathfinding with swarm intelligence
- PNG image maze loading
- Real-time visualization of particle exploration
- Creature animation following the best path found by the swarm
"""

import sys
import os
import math
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree

# Try to import GPU acceleration libraries
try:
    import cupy as cp
    GPU_AVAILABLE = True
    try:
        _ = cp.array([1, 2, 3])
        GPU_USABLE = True
        print("[GPU] ✓ GPU acceleration available and working (CuPy)")
    except Exception as e:
        GPU_USABLE = False
        print(f"[GPU] ✗ GPU available but not usable: {e}")
        print("[GPU] Falling back to CPU")
except ImportError:
    GPU_AVAILABLE = False
    GPU_USABLE = False
    print("[GPU] ✗ GPU acceleration not available")

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("[Numba] ✓ JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range
    print("[Numba] ✗ JIT compilation not available")

# Add the pythonrobotics path
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

# Maze configuration
MAZE_WIDTH = 20
MAZE_HEIGHT = 20
CELL_SIZE = 1.0
ROBOT_RADIUS = 0.3

# PSO parameters
N_PARTICLES = 30  # Number of particles in the swarm
MAX_ITERATIONS = 200  # Maximum iterations
W_START = 0.9  # Initial inertia weight
W_END = 0.4  # Final inertia weight
C1 = 1.5  # Cognitive coefficient (personal best influence)
C2 = 1.5  # Social coefficient (global best influence)


class Particle:
    """Represents a single particle in the PSO swarm"""
    
    def __init__(self, search_bounds, spawn_position):
        self.search_bounds = search_bounds
        self.max_velocity = np.array([(b[1] - b[0]) * 0.05 for b in search_bounds])
        # Start near spawn position with some randomness
        self.position = np.array(spawn_position) + np.random.randn(2) * 0.5
        self.velocity = np.random.randn(2) * 0.1
        self.personal_best_position = self.position.copy()
        self.personal_best_value = np.inf
        self.path = [self.position.copy()]
    
    def update_velocity(self, gbest_pos, w, c1, c2):
        """Update particle velocity using PSO equation"""
        r1 = np.random.rand(2)
        r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.personal_best_position - self.position)
        social = c2 * r2 * (gbest_pos - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
    
    def update_position(self):
        """Update particle position"""
        self.position = self.position + self.velocity
        # Keep in bounds
        for i in range(2):
            self.position[i] = np.clip(
                self.position[i], 
                self.search_bounds[i][0], 
                self.search_bounds[i][1]
            )
        self.path.append(self.position.copy())


class PSOSwarm:
    """Particle Swarm Optimization swarm for maze pathfinding"""
    
    def __init__(self, n_particles, max_iter, target, search_bounds, spawn_position, maze):
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.target = np.array(target)
        self.maze = maze
        self.search_bounds = search_bounds
        
        # PSO parameters
        self.w_start = W_START
        self.w_end = W_END
        self.c1 = C1
        self.c2 = C2
        
        # Initialize particles
        self.particles = [
            Particle(search_bounds, spawn_position) for _ in range(n_particles)
        ]
        
        self.gbest_position = None
        self.gbest_value = np.inf
        self.gbest_path = []
        self.iteration = 0
        
        # Build obstacle map for fast collision checking
        self._build_obstacle_map()
    
    def _build_obstacle_map(self):
        """Build obstacle map from maze walls with border walls"""
        self.obstacle_map = np.zeros((self.maze.width, self.maze.height), dtype=bool)
        
        # Add all internal walls
        for wall_x, wall_y in self.maze.walls:
            if 0 <= wall_x < self.maze.width and 0 <= wall_y < self.maze.height:
                self.obstacle_map[int(wall_x), int(wall_y)] = True
        
        # Add border walls to prevent particles from escaping
        for i in range(self.maze.width):
            self.obstacle_map[i, 0] = True  # Top border
            self.obstacle_map[i, self.maze.height - 1] = True  # Bottom border
        for i in range(self.maze.height):
            self.obstacle_map[0, i] = True  # Left border
            self.obstacle_map[self.maze.width - 1, i] = True  # Right border
        
        # Create KD-tree for obstacle points (for distance queries)
        ox, oy = self.maze.get_obstacle_list()
        # Add border points to obstacle list
        for i in range(self.maze.width):
            ox.append(float(i))
            oy.append(0.0)
            ox.append(float(i))
            oy.append(float(self.maze.height - 1))
        for i in range(self.maze.height):
            ox.append(0.0)
            oy.append(float(i))
            ox.append(float(self.maze.width - 1))
            oy.append(float(i))
        
        if len(ox) > 0:
            self.obstacle_kd_tree = cKDTree(np.vstack((ox, oy)).T)
        else:
            self.obstacle_kd_tree = None
    
    def is_collision(self, pos, radius=ROBOT_RADIUS):
        """Check if position collides with walls (treating walls as filled cells)"""
        x, y = pos
        
        # Check bounds with margin
        margin = radius + 0.1
        if x < margin or x >= self.maze.width - margin:
            return True
        if y < margin or y >= self.maze.height - margin:
            return True
        
        # Check all cells that the particle (with radius) might overlap with
        # Each wall cell is a filled square: [x-0.5, x+0.5] × [y-0.5, y+0.5]
        min_x = int(np.floor(x - radius))
        max_x = int(np.ceil(x + radius))
        min_y = int(np.floor(y - radius))
        max_y = int(np.ceil(y + radius))
        
        # Clamp to valid range
        min_x = max(0, min_x)
        max_x = min(self.maze.width - 1, max_x)
        min_y = max(0, min_y)
        max_y = min(self.maze.height - 1, max_y)
        
        # Check each cell in the overlap region
        for cell_x in range(min_x, max_x + 1):
            for cell_y in range(min_y, max_y + 1):
                if self.obstacle_map[cell_x, cell_y]:
                    # Check if particle circle overlaps with wall cell square
                    # Cell center is at (cell_x + 0.5, cell_y + 0.5)
                    cell_center_x = cell_x + 0.5
                    cell_center_y = cell_y + 0.5
                    
                    # Distance from particle center to cell center
                    dx = abs(x - cell_center_x)
                    dy = abs(y - cell_center_y)
                    
                    # If particle is far enough, no collision
                    if dx > 0.5 + radius or dy > 0.5 + radius:
                        continue
                    
                    # If particle is close to cell center, definitely colliding
                    if dx < 0.5 and dy < 0.5:
                        return True
                    
                    # Check corner cases (particle circle vs cell square)
                    # Distance to nearest point on cell square
                    nearest_x = max(cell_x, min(x, cell_x + 1))
                    nearest_y = max(cell_y, min(y, cell_y + 1))
                    dist_to_cell = np.sqrt((x - nearest_x)**2 + (y - nearest_y)**2)
                    
                    if dist_to_cell < radius:
                        return True
        
        return False
    
    def fitness(self, pos):
        """Calculate fitness - distance to target + obstacle penalty"""
        # Distance to target
        dist = np.linalg.norm(pos - self.target)
        
        # Obstacle penalty
        penalty = 0.0
        if self.is_collision(pos):
            penalty += 10000  # Heavy penalty for being in a wall
        
        # Proximity penalty (encourage staying away from walls)
        if self.obstacle_kd_tree is not None:
            dist_to_wall, _ = self.obstacle_kd_tree.query([pos[0], pos[1]], k=1)
            if dist_to_wall < ROBOT_RADIUS * 2:
                penalty += 1000 / (dist_to_wall + 0.1)
        
        return dist + penalty
    
    def check_path_collision(self, start, end):
        """Check if path from start to end collides with obstacles"""
        # Sample points along the path more densely for better accuracy
        path_length = np.linalg.norm(end - start)
        num_samples = max(20, int(path_length * 20))  # More samples for longer paths
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_pos = start + t * (end - start)
            if self.is_collision(sample_pos, radius=ROBOT_RADIUS):
                return True
        return False
    
    def step(self):
        """Run one PSO iteration"""
        if self.iteration >= self.max_iter:
            return False
        
        # Update inertia weight (linear decay)
        w = self.w_start - (self.w_start - self.w_end) * (self.iteration / self.max_iter)
        
        # Evaluate all particles
        for particle in self.particles:
            value = self.fitness(particle.position)
            
            # Update personal best
            if value < particle.personal_best_value:
                particle.personal_best_value = value
                particle.personal_best_position = particle.position.copy()
            
            # Update global best
            if value < self.gbest_value:
                self.gbest_value = value
                self.gbest_position = particle.position.copy()
        
        if self.gbest_position is not None:
            self.gbest_path.append(self.gbest_position.copy())
        
        # Update particles
        for particle in self.particles:
            if self.gbest_position is not None:
                particle.update_velocity(self.gbest_position, w, self.c1, self.c2)
            
            # Predict next position
            next_pos = particle.position + particle.velocity
            
            # Check collision BEFORE moving
            collision = False
            if self.check_path_collision(particle.position, next_pos) or self.is_collision(next_pos):
                collision = True
                # Strongly reduce velocity
                particle.velocity *= 0.1
                
                # Push away from nearest obstacle
                if self.obstacle_kd_tree is not None:
                    dist, idx = self.obstacle_kd_tree.query([particle.position[0], particle.position[1]], k=1)
                    if dist < ROBOT_RADIUS * 5:
                        obstacle_pos = self.obstacle_kd_tree.data[idx]
                        direction = particle.position - obstacle_pos
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-10:
                            direction = direction / direction_norm
                            # Push away more strongly
                            particle.velocity += direction * 1.0
            
            # Update position
            particle.update_position()
            
            # If still in collision after update, move back toward personal best
            if self.is_collision(particle.position):
                # Move back toward personal best or previous valid position
                if particle.personal_best_value < np.inf:
                    particle.position = particle.personal_best_position.copy()
                elif len(particle.path) > 1:
                    particle.position = np.array(particle.path[-2])
                particle.velocity *= 0.05  # Very strong reduction
        
        self.iteration += 1
        if self.iteration % 20 == 0:
            print(f"[PSO] Iteration {self.iteration}/{self.max_iter}, Best Fitness: {self.gbest_value:.2f}")
        
        return True


class Maze:
    """Simple maze with walls and open spaces"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = set()
        # Start at center by default
        self.start = (width / 2.0, height / 2.0)
        self.exit = (width - 1.5, height - 1.5)
    
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
        """Convert walls to obstacle lists"""
        ox, oy = [], []
        for wall_x, wall_y in self.walls:
            ox.append(float(wall_x))
            oy.append(float(wall_y))
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
        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        
        height, width = img_array.shape
        
        maze = Maze(width, height)
        
        threshold = 128
        
        # Find walls
        for y in range(height):
            for x in range(width):
                pixel_value = img_array[y, x]
                if pixel_value < threshold:
                    maze.add_wall(x, y)
        
        # Auto-detect start and exit if not provided
        if start_pos is None:
            center_x = width // 2
            center_y = height // 2
            
            if center_x < width and center_y < height and img_array[center_y, center_x] >= threshold:
                maze.start = (float(center_x), float(center_y))
            else:
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
    """Create a simple maze with guaranteed path"""
    maze = Maze(MAZE_WIDTH, MAZE_HEIGHT)
    maze.start = (MAZE_WIDTH / 2.0, MAZE_HEIGHT / 2.0)
    
    # Create walls with gaps
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


class Creature:
    """Creature that navigates using PSO"""
    
    def __init__(self, maze):
        self.maze = maze
        self.current_path = []
        self.path_index = 0
        self.found_path = False
        self.swarm = None
    
    def find_path(self):
        """Use PSO to find path from start to exit (called during animation)"""
        # Pathfinding is now done during animation for real-time visualization
        # This method is kept for compatibility but swarm is initialized in animate()
        return True
    
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
        self.frame_count = 0
        self.phase = 'optimizing'  # 'optimizing' or 'moving'
        self.optimization_complete = False
    
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
        
        # Draw PSO particles if available
        if self.creature.swarm is not None:
            # Draw particle paths (trails)
            for particle in self.creature.swarm.particles:
                if len(particle.path) > 1:
                    path = np.array(particle.path)
                    # Show recent path (last 50 points for performance)
                    recent_path = path[-50:] if len(path) > 50 else path
                    self.ax.plot(recent_path[:, 0], recent_path[:, 1], 
                               'b-', linewidth=0.5, alpha=0.2, zorder=2)
            
            # Draw all particles with different colors based on fitness
            particle_positions = np.array([p.position for p in self.creature.swarm.particles])
            particle_fitness = np.array([p.personal_best_value for p in self.creature.swarm.particles])
            
            # Color particles by fitness (better = greener)
            if len(particle_fitness) > 0 and np.max(particle_fitness) > np.min(particle_fitness):
                normalized_fitness = (particle_fitness - np.min(particle_fitness)) / (np.max(particle_fitness) - np.min(particle_fitness) + 1e-10)
                colors = plt.cm.RdYlGn(1 - normalized_fitness)  # Red (bad) to Green (good)
            else:
                colors = 'lightblue'
            
            self.ax.scatter(particle_positions[:, 0], particle_positions[:, 1], 
                          c=colors if isinstance(colors, np.ndarray) else [colors] * len(particle_positions),
                          s=50, alpha=0.7, edgecolors='darkblue', linewidths=0.5,
                          label='Particles', zorder=6)
            
            # Draw personal best positions
            pbest_positions = np.array([p.personal_best_position for p in self.creature.swarm.particles])
            self.ax.scatter(pbest_positions[:, 0], pbest_positions[:, 1], 
                          c='cyan', s=20, alpha=0.4, marker='x', zorder=5)
            
            # Draw global best position
            if self.creature.swarm.gbest_position is not None:
                self.ax.scatter(self.creature.swarm.gbest_position[0], 
                              self.creature.swarm.gbest_position[1],
                              c='yellow', s=200, marker='*', 
                              edgecolors='orange', linewidths=2,
                              label='Global Best', zorder=7)
            
            # Draw global best path (evolving)
            if self.creature.swarm.gbest_path and len(self.creature.swarm.gbest_path) > 1:
                gbest_path = np.array(self.creature.swarm.gbest_path)
                self.ax.plot(gbest_path[:, 0], gbest_path[:, 1], 'y--', 
                           linewidth=3, alpha=0.8, label='Best Path (Evolving)', zorder=4)
        
        # Draw final path (if optimization complete)
        if self.optimization_complete and self.creature.found_path and self.creature.current_path:
            path_x = [p[0] for p in self.creature.current_path]
            path_y = [p[1] for p in self.creature.current_path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Final Path', zorder=4)
        
        # Draw creature (only in moving phase)
        if self.phase == 'moving' and self.creature.found_path:
            cx, cy = self.creature.get_current_position()
            creature_circle = plt.Circle((cx, cy), 0.4, color='blue', zorder=10)
            self.ax.add_patch(creature_circle)
            
            # Add eyes
            self.ax.scatter([cx - 0.1, cx + 0.1], [cy + 0.1, cy + 0.1], 
                           c='white', s=40, zorder=11)
        
        # Title based on phase
        if self.phase == 'optimizing' and self.creature.swarm:
            title = f'PSO Pathfinding - Optimization Phase (Iteration: {self.creature.swarm.iteration}/{MAX_ITERATIONS})'
            if self.creature.swarm.gbest_value < np.inf:
                title += f'\nBest Fitness: {self.creature.swarm.gbest_value:.2f}'
        else:
            title = 'PSO Pathfinding - Movement Phase'
        
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.ax.legend(loc='upper right', fontsize=8, ncol=2)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
    
    def animate(self, frame):
        """Animation function"""
        # Phase 1: Show optimization process
        if self.phase == 'optimizing':
            if self.creature.swarm is None:
                # Initialize swarm
                print("[PSO] Starting pathfinding with Particle Swarm Optimization...")
                search_bounds = [(0, self.maze.width), (0, self.maze.height)]
                self.creature.swarm = PSOSwarm(
                    n_particles=N_PARTICLES,
                    max_iter=MAX_ITERATIONS,
                    target=self.maze.exit,
                    search_bounds=search_bounds,
                    spawn_position=self.maze.start,
                    maze=self.maze
                )
            
            # Run one PSO iteration
            if self.creature.swarm.iteration < MAX_ITERATIONS:
                self.creature.swarm.step()
            else:
                # Optimization complete, switch to movement phase
                self.optimization_complete = True
                if self.creature.swarm.gbest_path:
                    self.creature.current_path = [(p[0], p[1]) for p in self.creature.swarm.gbest_path]
                    self.creature.found_path = True
                    self.creature.path_index = 0
                    print(f"[PSO] ✓ Optimization complete! Best fitness: {self.creature.swarm.gbest_value:.2f}")
                    print(f"[PSO] Path found with {len(self.creature.current_path)} points")
                    print("[PSO] Switching to movement phase...")
                self.phase = 'moving'
        
        # Phase 2: Show creature moving along path
        elif self.phase == 'moving':
            if self.creature.found_path and frame > 0 and frame % 2 == 0:
                self.creature.move_next()
        
        self.draw_maze()
        return []
    
    def run(self):
        """Run the game"""
        print("=" * 50)
        print("Starting PSO Pathfinding Visualization")
        print("=" * 50)
        print("Phase 1: Watch particles explore the maze in real-time")
        print("Phase 2: Watch the creature follow the best path found")
        print("=" * 50)
        
        # Create animation that runs both phases
        # Optimization phase: MAX_ITERATIONS frames
        # Movement phase: path length * 2 frames
        total_frames = MAX_ITERATIONS + (200 if True else 0)  # Will adjust based on path length
        
        self.animation = FuncAnimation(self.fig, self.animate, frames=total_frames,
                                      interval=50, repeat=False, blit=False)
        
        plt.tight_layout()
        plt.show()


def load_maze_from_maps():
    """Check maps folder for PNG files and load the first one found"""
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    
    if not os.path.exists(maps_dir):
        print(f"Maps directory not found: {maps_dir}")
        return None
    
    png_files = glob.glob(os.path.join(maps_dir, '*.png'))
    png_files.extend(glob.glob(os.path.join(maps_dir, '*.PNG')))
    
    if not png_files:
        print(f"No PNG files found in {maps_dir}")
        return None
    
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
    print("Maze Game with Particle Swarm Optimization")
    print("=" * 50)
    
    # Try to load maze from PNG image
    maze = load_maze_from_maps()
    
    if maze is None:
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

