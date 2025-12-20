# Resolution Relationship: grid_resolution vs obstacle_map_resolution

## Answer: **No, but it's recommended**

`obstacle_map_resolution` should be **≤ grid_resolution** (equal or lower), not always lower.

---

## How They Work Together

### The Process:
1. **Create obstacle map** at `obstacle_map_resolution` (fine or coarse)
2. **Create node grid** at `grid_resolution` (for pathfinding)
3. **Sample obstacle map** to node grid positions

### The Sampling:
```python
# Each node samples from the obstacle map
node_x_indices = ((X - self.min_x) / self.obstacle_map_resolution).astype(int)
```

---

## Three Scenarios

### ✅ **Scenario 1: obstacle_map_resolution < grid_resolution** (RECOMMENDED)
**Example:** `grid_resolution: 4.0`, `obstacle_map_resolution: 2.0`

**For 640x640 image:**
- Obstacle map: 320×320 cells (fine collision detection)
- Node grid: 160×160 nodes (coarse search space)
- **Sampling**: Each node samples from a 2×2 area of obstacle map

**Result:**
- ✅ Fast pathfinding (fewer nodes)
- ✅ Precise collision detection (finer obstacle map)
- ✅ **Best balance of speed and precision**

---

### ✅ **Scenario 2: obstacle_map_resolution = grid_resolution** (GOOD)
**Example:** `grid_resolution: 4.0`, `obstacle_map_resolution: 4.0`

**For 640x640 image:**
- Obstacle map: 160×160 cells
- Node grid: 160×160 nodes
- **Sampling**: Each node samples from 1 cell of obstacle map

**Result:**
- ✅ Fast pathfinding
- ✅ Good collision detection
- ✅ Simpler (same resolution for both)

---

### ⚠️ **Scenario 3: obstacle_map_resolution > grid_resolution** (NOT RECOMMENDED)
**Example:** `grid_resolution: 4.0`, `obstacle_map_resolution: 8.0`

**For 640x640 image:**
- Obstacle map: 80×80 cells (coarse collision detection)
- Node grid: 160×160 nodes (finer search space)
- **Sampling**: Multiple nodes share the same obstacle cell

**Result:**
- ✅ Fast pathfinding
- ⚠️ Less precise collision detection (coarse obstacle map)
- ⚠️ **Wasteful**: Fine search grid but coarse collision data

**Why it's wasteful:**
- You're exploring many nodes (fine grid)
- But collision checking is imprecise (coarse obstacle map)
- Better to use coarser grid_resolution instead

---

## Best Practices

### ✅ **Recommended: obstacle_map_resolution ≤ grid_resolution**

| Use Case | grid_resolution | obstacle_map_resolution | Why |
|----------|----------------|------------------------|-----|
| **Maximum Speed** | 8.0 | 4.0 | Fast search, acceptable collision |
| **Balanced** | 4.0 | 2.0 | Fast search, precise collision ⭐ |
| **Quality Focus** | 2.0 | 1.0 | Slower search, very precise collision |
| **Equal Resolution** | 4.0 | 4.0 | Simple, good balance |

### ❌ **Not Recommended: obstacle_map_resolution > grid_resolution**

| grid_resolution | obstacle_map_resolution | Problem |
|----------------|------------------------|---------|
| 4.0 | 8.0 | Coarse collision with fine search (wasteful) |
| 2.0 | 4.0 | Coarse collision with fine search (wasteful) |

---

## Visual Example (640x640 Image)

### ✅ Good Configuration:
```
grid_resolution: 4.0
obstacle_map_resolution: 2.0

Obstacle Map: 320×320 cells (fine)
    ↓ (sampling)
Node Grid: 160×160 nodes (coarse)
    ↓
Each node checks 2×2 obstacle cells
```

### ⚠️ Not Ideal Configuration:
```
grid_resolution: 4.0
obstacle_map_resolution: 8.0

Obstacle Map: 80×80 cells (coarse)
    ↓ (sampling)
Node Grid: 160×160 nodes (fine)
    ↓
Multiple nodes share same obstacle cell
```

---

## Code Warning

The code already warns you if `obstacle_map_resolution > grid_resolution`:

```python
elif obstacle_map_resolution > grid_resolution:
    print(f"[A*] ⚠ Obstacle map resolution ({obstacle_map_resolution}) > grid resolution ({grid_resolution})")
    print(f"[A*] This provides high collision sensitivity with reduced search space")
```

**Note:** The warning message is a bit misleading - it actually provides **lower** collision sensitivity, not higher.

---

## Summary

### Rule of Thumb:
**`obstacle_map_resolution ≤ grid_resolution`**

### Why:
- **Finer obstacle map** = Better collision detection
- **Coarser node grid** = Faster pathfinding
- **Best combination**: Fine obstacle map + Coarse node grid

### Your Current Settings:
```json
{
  "grid_resolution": 4.0,
  "obstacle_map_resolution": 2.0
}
```

✅ **This is PERFECT!** 
- obstacle_map_resolution (2.0) < grid_resolution (4.0)
- Fast pathfinding with precise collision detection

---

## Quick Reference

| Image Size | Recommended grid_resolution | Recommended obstacle_map_resolution |
|------------|---------------------------|-----------------------------------|
| 64×64 | 1.0-2.0 | 1.0 |
| 128×128 | 2.0-4.0 | 1.0-2.0 |
| 320×320 | 4.0-8.0 | 2.0-4.0 |
| **640×640** | **4.0-8.0** | **2.0-4.0** ⭐ |
| 1280×1280 | 8.0-16.0 | 4.0-8.0 |

**Always ensure:** `obstacle_map_resolution ≤ grid_resolution`

