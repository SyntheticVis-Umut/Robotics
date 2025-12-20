# Comparison: Distance Maps and Obstacle Maps in Dijkstra.py vs AStar.py

## Overview
Both files use distance maps and obstacle maps, but **AStar.py** has more advanced resolution control and optimization features.

---

## 1. DISTANCE MAP (Distance Field)

### **Similarities:**
Both implementations:
- Use the same `compute_distance_field()` method in the `Maze` class
- Support both SDF (Signed Distance Field) and UDF (Unsigned Distance Field)
- Normalize distance fields to [0, 1] range for cost calculation
- Use `min_clearance_norm` parameter to filter out nodes too close to obstacles
- Apply distance field as a cost modifier during pathfinding

### **Implementation Details:**
```python
# Both use the same method:
def compute_distance_field(self, use_sdf=True):
    obstacle_map = self.get_obstacle_map_2d()
    obstacles_bool = obstacle_map.astype(int)
    if use_sdf:
        return compute_sdf_scipy(obstacles_bool)
    else:
        return compute_udf_scipy(obstacles_bool)
```

### **Usage in Pathfinding:**
- **Dijkstra**: Uses distance field to skip nodes below `min_clearance_norm` threshold
- **A***: Uses distance field similarly, but with GPU acceleration support

---

## 2. OBSTACLE MAP

### **Key Differences:**

#### **A. Resolution Control**

**Dijkstra.py:**
- ❌ **NO separate obstacle map resolution control**
- Uses the same resolution for both:
  - Node grid (search space)
  - Obstacle map (collision detection)
- Resolution is determined by `CELL_SIZE` constant (1.0) or planner's `resolution` parameter
- Obstacle map is created by the parent `DijkstraPlanner` class (no customization)

**AStar.py:**
- ✅ **DUAL resolution system:**
  - `grid_resolution`: Controls node grid size (search space)
  - `obstacle_map_resolution`: Controls obstacle map precision (collision detection)
- Configurable via `path_config.json`:
  ```json
  {
    "grid_resolution": 2.0,           // Node grid: 2x2 pixels per node
    "obstacle_map_resolution": 1.0    // Obstacle map: 1 pixel per cell (max precision)
  }
  ```
- Allows **high collision sensitivity** with **reduced search space**

#### **B. Obstacle Map Creation**

**Dijkstra.py:**
```python
# Uses parent class's obstacle map (no customization)
super().__init__(ox, oy, resolution, rr)
# obstacle_map is created by parent DijkstraPlanner class
self.obstacle_map_np = np.array(self.obstacle_map, dtype=bool)
```

**AStar.py:**
```python
# Custom calc_obstacle_map() method with dual resolution
def calc_obstacle_map(self, ox, oy):
    # 1. Create FULL-RESOLUTION obstacle map (at obstacle_map_resolution)
    obstacle_x_width = round((self.max_x - self.min_x) / self.obstacle_map_resolution)
    obstacle_y_width = round((self.max_y - self.min_y) / self.obstacle_map_resolution)
    
    # 2. Create node grid (at grid_resolution)
    self.x_width = round((self.max_x - self.min_x) / self.resolution)
    self.y_width = round((self.max_y - self.min_y) / self.resolution)
    
    # 3. Sample full-resolution obstacle map at node grid positions
    obstacle_map = obstacle_map_full[node_x_indices, node_y_indices]
```

#### **C. GPU Acceleration**

**Dijkstra.py:**
- ✅ GPU support for obstacle map lookups
- ❌ NO GPU-accelerated obstacle map **creation**
- Uses parent class's obstacle map (CPU-based)

**AStar.py:**
- ✅ GPU-accelerated obstacle map **creation** (batched processing)
- ✅ GPU support for obstacle map lookups
- ✅ Batch processing of obstacles (2000 obstacles per batch)
- ✅ Full-resolution obstacle map computed on GPU, then sampled to node grid

#### **D. Vectorization**

**Dijkstra.py:**
- Uses parent class's obstacle map (likely less optimized)
- Converts to numpy array for faster access: `self.obstacle_map_np`

**AStar.py:**
- ✅ Custom vectorized obstacle map creation
- ✅ Broadcasting operations for distance calculations
- ✅ Separate CPU and GPU vectorized implementations

---

## 3. CONFIGURATION PARAMETERS

### **Dijkstra.py:**
```python
# path_config.json parameters:
{
  "use_distance_cost": True,
  "min_clearance_norm": 0.50,
  "num_points_per_segment": 15,
  "show_distance_map": True,
  "smoothing_enabled": True,
  "distance_heat_gamma": 1.0,
  "detection_mode": "canny",
  "canny_threshold1": 100,
  "canny_threshold2": 200,
  "map_file": None
}
```

### **AStar.py:**
```python
# path_config.json parameters (ADDITIONAL):
{
  "use_distance_cost": True,
  "min_clearance_norm": 0.50,
  "num_points_per_segment": 15,
  "show_distance_map": True,
  "smoothing_enabled": True,
  "distance_heat_gamma": 1.0,
  "detection_mode": "canny",  // Also supports "sobel"
  "canny_threshold1": 100,
  "canny_threshold2": 200,
  "sobel_threshold": 50,      // NEW: Sobel edge detection
  "map_file": None,
  "grid_resolution": 1.0,              // NEW: Node grid resolution
  "obstacle_map_resolution": 1.0        // NEW: Obstacle map resolution
}
```

---

## 4. PERFORMANCE IMPLICATIONS

### **Dijkstra.py:**
- **Search Space**: Fixed to obstacle map resolution
- **Collision Precision**: Same as search space resolution
- **Memory**: Lower (single resolution)
- **Speed**: Good for small maps, may be slower for large maps

### **AStar.py:**
- **Search Space**: Can be reduced via `grid_resolution` (e.g., 2.0 = 4x fewer nodes)
- **Collision Precision**: Can be maximized via `obstacle_map_resolution` (e.g., 1.0 = pixel-perfect)
- **Memory**: Higher (stores full-resolution obstacle map + node grid)
- **Speed**: 
  - Faster for large maps (reduced search space)
  - GPU acceleration for obstacle map creation
  - Better scalability

---

## 5. CODE STRUCTURE DIFFERENCES

### **Dijkstra.py:**
```python
class DijkstraPlannerWithTracking(dijkstra.DijkstraPlanner):
    def __init__(self, ox, oy, resolution, rr, distance_field=None, ...):
        super().__init__(ox, oy, resolution, rr)  # Parent creates obstacle map
        # Uses parent's obstacle_map directly
        self.obstacle_map_np = np.array(self.obstacle_map, dtype=bool)
```

### **AStar.py:**
```python
class AStarPlannerWithTracking(a_star.AStarPlanner):
    def __init__(self, ox, oy, resolution, rr, distance_field=None, 
                 obstacle_map_resolution=None, ...):
        self.obstacle_map_resolution = obstacle_map_resolution or resolution
        super().__init__(ox, oy, resolution, rr)  # Calls calc_obstacle_map()
    
    def calc_obstacle_map(self, ox, oy):  # OVERRIDDEN METHOD
        # Custom dual-resolution obstacle map creation
        # Full-resolution map + node grid sampling
```

---

## 6. SUMMARY TABLE

| Feature | Dijkstra.py | AStar.py |
|---------|-------------|----------|
| **Distance Map** | ✅ Same implementation | ✅ Same implementation |
| **Obstacle Map Resolution** | ❌ Single resolution | ✅ Dual resolution system |
| **Grid Resolution Control** | ❌ No | ✅ Yes (`grid_resolution`) |
| **Obstacle Map Resolution Control** | ❌ No | ✅ Yes (`obstacle_map_resolution`) |
| **GPU Obstacle Map Creation** | ❌ No | ✅ Yes (batched) |
| **GPU Obstacle Map Lookups** | ✅ Yes | ✅ Yes |
| **Vectorized Operations** | ⚠️ Limited | ✅ Full vectorization |
| **Sobel Edge Detection** | ❌ No | ✅ Yes |
| **Config Complexity** | Simple | Advanced |

---

## 7. RECOMMENDATIONS

### **Use Dijkstra.py when:**
- Maps are small to medium size (< 1000x1000 pixels)
- Simple configuration is preferred
- Single resolution is sufficient
- CPU-only environment

### **Use AStar.py when:**
- Maps are large (> 1000x1000 pixels)
- Need fine collision detection with reduced search space
- GPU acceleration is available
- Want maximum performance and flexibility
- Need Sobel edge detection option

---

## 8. EXAMPLE CONFIGURATION SCENARIOS

### **Scenario 1: High Precision, Small Search Space**
```json
{
  "grid_resolution": 5.0,           // 5x5 pixels per node (25x fewer nodes)
  "obstacle_map_resolution": 1.0    // Pixel-perfect collision detection
}
```
**Result**: Fast pathfinding with maximum collision sensitivity

### **Scenario 2: Balanced Performance**
```json
{
  "grid_resolution": 2.0,           // 2x2 pixels per node (4x fewer nodes)
  "obstacle_map_resolution": 2.0    // 2x2 pixels per obstacle cell
}
```
**Result**: Good balance between speed and precision

### **Scenario 3: Maximum Speed (Dijkstra-like)**
```json
{
  "grid_resolution": 1.0,           // 1 pixel per node
  "obstacle_map_resolution": 1.0    // 1 pixel per obstacle cell
}
```
**Result**: Same as Dijkstra.py behavior, but with GPU acceleration

---

## Conclusion

**AStar.py** provides significantly more advanced obstacle map handling with:
- Dual resolution system for independent control of search space and collision precision
- GPU-accelerated obstacle map creation
- Better scalability for large maps
- More configuration options

**Dijkstra.py** is simpler and sufficient for smaller maps, but lacks the advanced resolution control and GPU optimizations found in AStar.py.


