# Performance Optimization Guide for Large Images

## Why 640x640 Images Are Slow

### The Problem:

With `grid_resolution: 1.0` on a 640x640 image:
- **Search Space**: 640 × 640 = **409,600 nodes** to explore
- **Obstacle Map**: 640 × 640 = **409,600 cells** to compute
- **Distance Field**: 640 × 640 = **409,600 cells** to process
- **Pathfinding**: Must check millions of node combinations

**Result**: Extremely slow pathfinding (minutes or hours)

---

## Solution: Increase Grid Resolution

### Recommended Settings for 640x640 Images:

```json
{
  "grid_resolution": 4.0,           // 4x4 pixels per node
  "obstacle_map_resolution": 2.0    // 2x2 pixels per obstacle cell
}
```

**Benefits:**
- **Search Space**: 160 × 160 = **25,600 nodes** (16x reduction!)
- **Obstacle Map**: 320 × 320 = **102,400 cells** (still precise)
- **Speed**: **~16x faster** pathfinding
- **Quality**: Still maintains good path quality

---

## Resolution Settings Guide

### For 640x640 Images:

| grid_resolution | obstacle_map_resolution | Search Nodes | Speed | Quality |
|----------------|------------------------|--------------|-------|---------|
| 1.0 | 1.0 | 409,600 | Very Slow | Perfect |
| 2.0 | 2.0 | 102,400 | Slow | Excellent |
| **4.0** | **2.0** | **25,600** | **Fast** | **Very Good** ⭐ |
| 8.0 | 4.0 | 6,400 | Very Fast | Good |
| 10.0 | 5.0 | 4,096 | Extremely Fast | Acceptable |

### For 128x128 Images:

| grid_resolution | obstacle_map_resolution | Search Nodes | Speed | Quality |
|----------------|------------------------|--------------|-------|---------|
| 1.0 | 1.0 | 16,384 | Fast | Perfect |
| 2.0 | 1.0 | 4,096 | Very Fast | Excellent |
| 4.0 | 2.0 | 1,024 | Extremely Fast | Good |

### For 64x64 Images:

| grid_resolution | obstacle_map_resolution | Search Nodes | Speed | Quality |
|----------------|------------------------|--------------|-------|---------|
| 1.0 | 1.0 | 4,096 | Very Fast | Perfect |
| 2.0 | 1.0 | 1,024 | Extremely Fast | Excellent |

---

## Understanding the Two Resolutions

### `grid_resolution` (Node Grid)
- **Controls**: Search space size (how many nodes A*/Dijkstra explores)
- **Lower = More nodes = Slower but more precise paths**
- **Higher = Fewer nodes = Faster but coarser paths**
- **Recommendation**: 4.0-8.0 for 640x640 images

### `obstacle_map_resolution` (Collision Detection)
- **Controls**: Collision detection precision
- **Lower = More precise collision checking**
- **Higher = Faster collision checking but less precise**
- **Recommendation**: 1.0-2.0 for good collision sensitivity

### Strategy: **High grid_resolution + Low obstacle_map_resolution**
- Reduces search space (faster pathfinding)
- Maintains collision precision (safe paths)

---

## Performance Calculation

### Search Space Size:
```
nodes = (image_width / grid_resolution) × (image_height / grid_resolution)
```

### Example for 640x640:
- `grid_resolution: 1.0` → 640×640 = **409,600 nodes** ❌ Too slow
- `grid_resolution: 4.0` → 160×160 = **25,600 nodes** ✅ Good balance
- `grid_resolution: 8.0` → 80×80 = **6,400 nodes** ✅ Very fast

### Speed Improvement:
- **4.0**: ~16x faster than 1.0
- **8.0**: ~64x faster than 1.0
- **10.0**: ~100x faster than 1.0

---

## Quick Optimization Guide

### For Maximum Speed (640x640):
```json
{
  "grid_resolution": 8.0,
  "obstacle_map_resolution": 4.0
}
```
- **6,400 nodes** to search
- **Very fast** pathfinding
- **Good** path quality

### For Balanced Performance (640x640):
```json
{
  "grid_resolution": 4.0,
  "obstacle_map_resolution": 2.0
}
```
- **25,600 nodes** to search
- **Fast** pathfinding
- **Very good** path quality ⭐ **RECOMMENDED**

### For Maximum Quality (640x640):
```json
{
  "grid_resolution": 2.0,
  "obstacle_map_resolution": 1.0
}
```
- **102,400 nodes** to search
- **Moderate** speed
- **Excellent** path quality

---

## Additional Performance Tips

### 1. Disable Distance Field (if not needed):
```json
{
  "use_distance_cost": false
}
```
- Saves time computing distance field
- Still finds valid paths, just not safety-optimized

### 2. Disable Distance Map Visualization:
```json
{
  "show_distance_map": false
}
```
- Saves computation time
- Only affects visualization

### 3. Use Direct Mode (if applicable):
```json
{
  "detection_mode": "direct"
}
```
- Faster than Canny/Sobel edge detection
- Use when map is already binary

---

## Expected Performance

### 640x640 Image with `grid_resolution: 4.0`:

| Phase | Time (approx) |
|-------|---------------|
| Image Loading | < 1 second |
| Obstacle Map Creation | 2-5 seconds |
| Distance Field Computation | 3-10 seconds |
| Pathfinding (A*/Dijkstra) | 5-30 seconds |
| **Total** | **10-45 seconds** |

### 640x640 Image with `grid_resolution: 1.0`:

| Phase | Time (approx) |
|-------|---------------|
| Image Loading | < 1 second |
| Obstacle Map Creation | 10-30 seconds |
| Distance Field Computation | 30-120 seconds |
| Pathfinding (A*/Dijkstra) | **Minutes to hours** |
| **Total** | **Very slow** ❌ |

---

## Updated Configuration

I've updated your `path_config.json` with optimized settings:

```json
{
  "grid_resolution": 4.0,           // 16x fewer nodes (25,600 vs 409,600)
  "obstacle_map_resolution": 2.0    // Still precise collision detection
}
```

This should give you **~16x faster** pathfinding while maintaining good path quality!

