# Workflow Explanation: A* and Dijkstra Pathfinding

## Your Understanding vs. Actual Implementation

### Your Understanding:
1. ✅ Apply Sobel or Canny on `map.png`
2. ⚠️ Calculate distance map on `sobel.png` or `canny.png` (saved files)
3. ⚠️ Calculate A* or Dijkstra path based on distance map

### Actual Implementation:

## Correct Workflow:

### Step 1: Edge Detection on Original Map ✅
**What happens:**
- Load `map.png` (original image)
- Apply **Canny** or **Sobel** edge detection
- Store result in `maze.canny_edge_map` (numpy array in memory)
- **Optionally** save to `canny.png` or `sobel.png` for visualization (but not used for computation)

**Code location:**
```python
# In Maze.from_image()
if detection_mode == 'canny':
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    maze.canny_edge_map = edges  # Stored in memory
    cv2.imwrite('canny.png', edges)  # Saved for reference only
```

### Step 2: Create Obstacle Map from Edge Map ✅
**What happens:**
- Convert edge map (`canny_edge_map`) to 2D boolean obstacle map
- Edge pixels (white/255) → `True` (obstacles)
- Free space pixels (black/0) → `False` (free space)
- Add border walls

**Code location:**
```python
# In Maze.get_obstacle_map_2d()
if self.canny_edge_map is not None:
    obstacle_map = (self.canny_edge_map > 0).T.astype(bool)
    # Add border walls...
```

### Step 3: Calculate Distance Field from Obstacle Map ✅
**What happens:**
- Use the **obstacle map** (from Step 2) to compute distance field
- NOT from the saved `canny.png` or `sobel.png` files
- Distance field shows distance from each point to nearest obstacle
- Used for safety-aware pathfinding (prefer paths away from walls)

**Code location:**
```python
# In Maze.compute_distance_field()
obstacle_map = self.get_obstacle_map_2d()  # Uses canny_edge_map internally
obstacles_bool = obstacle_map.astype(int)
distance_field = compute_udf_scipy(obstacles_bool)  # or compute_sdf_scipy
```

### Step 4: Pathfinding with Both Obstacle List AND Distance Field ✅
**What happens:**
- **Obstacle list** (`ox`, `oy`): Direct list of obstacle coordinates from edge map
  - Used for collision checking during pathfinding
  - Determines which nodes are blocked
  
- **Distance field**: Used for cost calculation
  - Nodes closer to obstacles get higher cost
  - Nodes with `distance < min_clearance_norm` are treated as blocked
  - Encourages paths that stay away from walls

**Code location:**
```python
# In PathFinder.find_path()
ox, oy = self.maze.get_obstacle_list()  # From edge map
distance_field = self.maze.compute_distance_field()  # From obstacle map

planner = AStarPlannerWithTracking(
    ox, oy,  # Obstacle list for collision checking
    grid_resolution, ROBOT_RADIUS,
    distance_field=distance_field,  # For cost calculation
    use_distance_cost=use_distance_cost,
    ...
)
```

---

## Key Corrections:

### ❌ **Incorrect:** "Calculate distance map on sobel.png or canny.png"
**Why:** The saved PNG files (`canny.png`, `sobel.png`) are **only for visualization/reference**. The actual computation uses the in-memory `canny_edge_map` numpy array.

### ✅ **Correct:** "Calculate distance map from obstacle map (derived from edge map)"
**Why:** The distance field is computed from the obstacle map, which is created from the edge map stored in memory.

### ❌ **Incorrect:** "Calculate path based on distance map"
**Why:** The pathfinding uses **BOTH**:
- **Obstacle list** (from edge map) → for collision checking
- **Distance field** (from obstacle map) → for cost calculation (optional)

---

## Complete Data Flow:

```
map.png (original image)
    ↓
[Edge Detection: Canny/Sobel]
    ↓
canny_edge_map (numpy array in memory)
    ├─→ canny.png / sobel.png (saved for visualization only)
    ├─→ get_obstacle_list() → ox, oy (obstacle coordinates)
    └─→ get_obstacle_map_2d() → obstacle_map (2D boolean)
            ↓
        compute_distance_field() → distance_field (2D float array)
            ↓
        [Pathfinding Algorithm]
            ├─ Uses: ox, oy (for collision checking)
            └─ Uses: distance_field (for cost calculation, if enabled)
```

---

## Summary:

1. ✅ **Edge detection on map.png** → Creates `canny_edge_map` (in memory)
2. ✅ **Obstacle map from edge map** → Creates 2D boolean obstacle map
3. ✅ **Distance field from obstacle map** → Creates distance field
4. ✅ **Pathfinding uses both:**
   - Obstacle list (`ox`, `oy`) for collision checking
   - Distance field for cost calculation (if `use_distance_cost=True`)

**Important:** The saved PNG files (`canny.png`, `sobel.png`) are **NOT used in computation** - they're only saved for you to visualize what the edge detection found. All computation uses the in-memory numpy arrays.

