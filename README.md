# Maze Game with Multiple Pathfinding Algorithms

A simple maze game where a creature uses different pathfinding algorithms to find the shortest path from start to exit.

**Available Algorithms:**
- **`AStar.py`** - A* pathfinding algorithm (optimal and efficient)
- **`Dijkstra.py`** - Dijkstra's algorithm (guaranteed shortest path)
- **`HybridAStar.py`** - Hybrid A* algorithm (for vehicles with turning constraints)
- **`PSO.py`** - Particle Swarm Optimization (swarm intelligence-based pathfinding)

## Features

- **Maze Generation**: Pre-defined maze with walls and obstacles OR load from PNG images
- **PNG Image Support**: Load mazes from PNG images (black pixels = walls, white pixels = paths)
- **Multiple Pathfinding Algorithms**:
  - **A***: Optimal pathfinding with heuristic (fast and efficient)
  - **Dijkstra**: Guaranteed shortest path (explores more nodes)
  - **Hybrid A***: Vehicle-like pathfinding with turning constraints (realistic movement)
  - **PSO**: Particle Swarm Optimization (swarm intelligence, multiple particles explore simultaneously)
- **Visualization**: 
  - Watch the algorithm explore the maze (shows explored nodes and open set)
  - Watch the creature follow the shortest path to the exit
  - Hybrid A* shows vehicle orientation and smooth turning

## Requirements

- Python 3.x
- matplotlib
- numpy
- Pillow (PIL) for image processing
- scipy (for Hybrid A*)
- The pythonrobotics library (should be in `/Users/umutozdemir/Desktop/pythonrobotics`)

### Optional (for performance):
- **numba**: JIT compilation for 2-5x speedup (recommended)
- **cupy**: GPU acceleration for 10-100x speedup on large mazes (optional)

## Setup

### Creating Conda Environment

**Step 1: Create a new conda environment**
```bash
conda create -n maze_game python=3.9 -y
```

**Step 2: Activate the environment**
```bash
conda activate maze_game
```

**Step 3: Install required packages**
```bash
# Core dependencies
conda install matplotlib numpy scipy -y
pip install Pillow

# Optional: Performance optimizations (recommended)
pip install numba

# Optional: GPU acceleration (requires CUDA-compatible GPU)
# For CUDA 11.x:
pip install cupy-cuda11x
# For CUDA 12.x:
# pip install cupy-cuda12x
```

**Step 4: Verify installation**
```bash
python -c "import matplotlib, numpy, PIL, scipy; print('All packages installed successfully!')"
```

**Note**: The pythonrobotics library should already be available at `/Users/umutozdemir/Desktop/pythonrobotics`. The scripts will automatically add this path.

### Alternative: Using pip (without conda)

If you prefer not to use conda, you can install packages directly with pip:

```bash
pip install matplotlib numpy scipy Pillow numba
# Optional: pip install cupy-cuda11x  # for GPU support
```

## How to Run

### Using Conda Environment (Recommended)

**A* Algorithm:**
```bash
cd /Users/umutozdemir/Desktop/Robotics
conda activate maze_game
python AStar.py
```

**Dijkstra Algorithm:**
```bash
conda run -n maze_game python Dijkstra.py
```

**Hybrid A* Algorithm:**
```bash
conda run -n maze_game python HybridAStar.py
```

**PSO Algorithm:**
```bash
conda run -n maze_game python PSO.py
```

Or using conda run:
```bash
conda run -n maze_game python AStar.py      # A*
conda run -n maze_game python Dijkstra.py   # Dijkstra
conda run -n maze_game python HybridAStar.py # Hybrid A*
conda run -n maze_game python PSO.py        # PSO
```

### Using Default Python

```bash
cd /Users/umutozdemir/Desktop/Robotics
python3 AStar.py      # A*
python3 Dijkstra.py   # Dijkstra
python3 HybridAStar.py # Hybrid A*
python3 PSO.py        # PSO
```

## Configurable Path Settings (A* & Dijkstra)

Both A* and Dijkstra read runtime parameters from `path_config.json`:
- `use_distance_cost` (bool): Distance-field-based clearance filtering (true/false)
- `min_clearance_norm` (float 0–1): Hard cutoff; nodes below this clearance are blocked
- `num_points_per_segment` (int): Spline smoothing density (higher = smoother, slower)
- `show_distance_map` (bool): Show distance-field heatmap in the background
- `smoothing_enabled` (bool): Enable/disable Catmull-Rom smoothing
- `distance_heat_gamma` (float): Adjusts heatmap contrast ( >1 = hotter near obstacles, <1 = softer )

Current defaults (editable in `path_config.json`):
```json
{
  "use_distance_cost": true,
  "min_clearance_norm": 0.4,
  "num_points_per_segment": 15,
  "show_distance_map": true,
  "smoothing_enabled": true,
  "distance_heat_gamma": 1.0
}
```
Update this file to adjust clearance, smoothing, and visualization without touching code.

## Loading Mazes from PNG Images

1. Place your PNG maze images in the `maps/` folder
2. The game will automatically detect and load the first PNG file found
3. If no PNG files are found, it will use the default generated maze

### Image Format Requirements

- **Format**: PNG
- **Color Scheme**:
  - **Black pixels** (or dark pixels < 128 grayscale) = Walls
  - **White pixels** (or light pixels >= 128 grayscale) = Open paths

### Auto-Detection

- **Start Position**: Center of the maze (or nearest valid position)
- **Exit Position**: Last white pixel found from bottom-right (scanning right to left, bottom to top)

See `maps/README.md` for more details.

## What You'll See

1. **Exploration Phase**:
   - **A* and Dijkstra**: Shows explored nodes (light blue dots) and open set (yellow stars)
   - **PSO**: Shows all particles (light blue dots) exploring simultaneously, with global best path (yellow dashed line)
   - **Hybrid A***: Shows vehicle exploration with orientation
   - Green circle: Start position
   - Red square: Exit position

2. **Movement Phase**: The creature follows the path found:
   - **A* and Dijkstra**: Blue circle with eyes moving along the path
   - **Hybrid A***: Blue circle with arrow showing orientation, demonstrating vehicle-like turning
   - **PSO**: Blue circle with eyes following the best path found by the swarm
   - Blue line: The optimal/found path

## Maze Configuration

- Maze size: 20x20 cells
- Start position: (1, 1)
- Exit position: (18, 18)
- The maze contains various walls and obstacles that create a challenging path

## Algorithm Comparison

| Algorithm | Best For | Characteristics |
|-----------|----------|----------------|
| **A*** | General pathfinding | Fast, optimal, uses heuristic |
| **Dijkstra** | Guaranteed shortest path | Explores more nodes, no heuristic |
| **Hybrid A*** | Vehicle navigation | Realistic turning, continuous paths |
| **PSO** | Optimization problems, dynamic environments | Swarm intelligence, multiple particles explore simultaneously |

**Note**: Hybrid A* is designed for vehicles with turning constraints. It scales the maze coordinates to accommodate the vehicle size (which needs ~2.4 units of clearance).

## Customization

### Option 1: Edit the Code
You can modify the maze by editing the `create_maze()` function in any of the algorithm files to add or remove walls.

### Option 2: Use PNG Images
1. Create a black and white PNG image of your maze
2. Place it in the `maps/` folder
3. Run the game - it will automatically load your maze!

### Creating Maze Images

You can use any image editor to create maze images:
- Use black (#000000) for walls
- Use white (#FFFFFF) for paths
- Make sure there's a clear path from start to exit
- The image will be automatically converted to grayscale

A sample maze (`sample_maze.png`) is included in the `maps/` folder as an example.

