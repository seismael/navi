# Observer Palette — Dashboard Rendering Reference

This document describes the unified colour system used across all Navi auditor
dashboard visualisation surfaces.  Every panel, snapshot, recorder, and
occupancy view shares one palette pipeline so distance always looks the same
regardless of the viewing mode.

---

## 1. Design Goals

| Goal | How It Is Achieved |
|------|--------------------|
| **Threat-graded readability** | Collision-red near field transitions through warm → cool → gray so operators read distance intuitively. |
| **Unbounded range** | Logarithmic scale maps `[0, ∞)` → `[0, 1)` — no hard far-clipping distance. |
| **Near-field priority** | 50 % of the colour spectrum covers 0–10 m (configurable via `focus_m`). |
| **Far-field differentiation** | Six distinct gray shades between 25 m and ∞ keep far structure visible. |
| **Missing-data clarity** | Invalid/unsampled bins get a purple checkerboard tile pattern — impossible to confuse with any distance colour. |
| **Single source of truth** | One palette definition in `renderers.py` feeds all 10+ consumer surfaces. |

---

## 2. Colour Scale Mathematics

### 2.1  Unbounded Logarithmic Mapping

Every metric distance `d` (in metres) maps to a normalised position `t ∈ [0, 1)`:

$$
t = \frac{\ln(1 + d)}{\ln(1 + d) + C}, \quad C = \ln(1 + \text{focus\_m})
$$

The `focus_m` parameter (default **10.0**) controls where 50 % of the
colour spectrum is allocated.  With the default: $C = \ln(11) \approx 2.398$.

Properties:
- `t(0) = 0` — contact / collision.
- `t(focus_m) = 0.5` — half the spectrum covers `[0, focus_m]`.
- `t -> 1` as `d -> inf` — spectrum never clips, just compresses.

Tuning guide:
- `focus_m = 5`  — maximises near-field colour detail (indoor micro-nav).
- `focus_m = 10` — default; balanced near-field and room-scale visibility.
- `focus_m = 30` — wider view, more colour allocated to mid-range.

### 2.2  Environment Wire Format

The SDF/DAG environment sphere-traces each ray and normalises depth before
transmission as:

$$
\text{depth\_wire} = \frac{\ln(1 + d_{\text{clamped}})}{\ln(1 + d_{\max})}
$$

where `d_max = 100 m` by default.  The dashboard denormalises back to metric
metres via `expm1(depth_wire × ln(1 + d_max))` and then applies the unbounded
log mapping above.

---

## 3. Palette Anchor Table

Each anchor is a fixed metric distance with a hand-tuned BGR colour.
Between anchors, colours are pre-interpolated into a 1024-bin continuous
LUT at module load time.  All `t`-values below use the default
`focus_m = 10`.

| # | Distance | Colour Name | BGR | `t` | Spectrum % | LUT Index |
|--:|:---------|:------------|:----|----:|----------:|----------:|
| 0 | **0.0 m** | Bright Red | `(30, 30, 255)` | 0.000 | 0.0 % | 0 |
| 1 | **0.1 m** | Red-Orange | `(25, 90, 255)` | 0.038 | 3.8 % | 39 |
| 2 | **0.3 m** | Orange | `(30, 155, 255)` | 0.099 | 9.9 % | 101 |
| 3 | **0.7 m** | Gold / Amber | `(40, 210, 245)` | 0.181 | 18.1 % | 185 |
| 4 | **1.5 m** | Yellow-Green | `(50, 230, 190)` | 0.276 | 27.6 % | 283 |
| 5 | **3.0 m** | Fresh Green | `(65, 215, 90)` | 0.366 | 36.6 % | 375 |
| 6 | **6.0 m** | Teal | `(150, 200, 45)` | 0.448 | 44.8 % | 458 |
| 7 | **12.0 m** | Steel Blue | `(210, 155, 50)` | 0.517 | 51.7 % | 529 |
| 8 | **25.0 m** | Medium Blue | `(195, 105, 55)` | 0.576 | 57.6 % | 589 |
| 9 | **35.0 m** | Dark Steel Blue | `(170, 88, 55)` | 0.599 | 59.9 % | 613 |
| 10 | **50.0 m** | Dark Blue | `(145, 72, 55)` | 0.621 | 62.1 % | 635 |
| 11 | **75.0 m** | Blue-Gray | `(118, 64, 56)` | 0.644 | 64.4 % | 658 |
| 12 | **100.0 m** | Medium Gray | `(95, 58, 56)` | 0.658 | 65.8 % | 673 |
| 13 | **150.0 m** | Dark Gray | `(72, 52, 50)` | 0.677 | 67.7 % | 692 |
| 14 | **300.0 m** | Dim Gray | `(56, 48, 48)` | 0.704 | 70.4 % | 720 |
| 15 | **inf** | Faint Dark Gray | `(42, 40, 40)` | 1.000 | 100.0 % | 1023 |

### 3.1  Spectrum Allocation per Segment

| Range | Spectrum Span | Description |
|:------|:-------------|:------------|
| 0 -- 0.1 m | 3.8 % | Collision / contact zone (bright red) |
| 0.1 -- 0.3 m | 6.0 % | Danger proximity (red-orange to orange) |
| 0.3 -- 0.7 m | 8.3 % | Close warning (orange to gold) |
| 0.7 -- 1.5 m | 9.5 % | Near-field (gold to yellow-green) |
| 1.5 -- 3.0 m | 9.0 % | Transition (yellow-green to green) |
| 3.0 -- 6.0 m | 8.2 % | Comfortable navigation (green to teal) |
| 6.0 -- 12.0 m | 6.9 % | Mid-range structure (teal to steel blue) |
| 12.0 -- 25.0 m | 5.9 % | Away structure (steel blue to medium blue) |
| 25.0 -- 35.0 m | 2.3 % | Far transition (medium blue to dark steel blue) |
| 35.0 -- 50.0 m | 2.2 % | Deep far (dark steel blue to dark blue) |
| 50.0 -- 75.0 m | 2.2 % | Very far (dark blue to blue-gray) |
| 75.0 -- 100.0 m | 1.4 % | Distant (blue-gray to medium gray) |
| 100.0 -- 150.0 m | 1.9 % | Receding (medium gray to dark gray) |
| 150.0 -- 300.0 m | 2.8 % | Fading horizon (dark gray to dim gray) |
| 300.0 m -- inf | 29.6 % | Void (dim gray to faint dark gray) |

Key insight: with the default `focus_m = 10`, **~58 % of the spectrum covers
0--25 m** (the active navigation zone) while the remaining **~42 % stretches
smoothly from 25 m to infinity** with six distinct gray shades preventing the
far field from collapsing into a single flat colour.  Compared to the previous
`focus_m = 30` setting, near-field colour diversity is significantly higher.

---

## 4. Missing-Data Representation

When a ray produces no valid distance (sensor gap, escape without hitting
geometry, non-finite result), that bin is marked `valid = False` in the
`DistanceMatrix` wire format.

### Visual Treatment

Invalid bins are stamped with a **4×4 pixel dark-purple checkerboard**:

| Tile | BGR | Hex |
|------|-----|-----|
| Colour A (dark purple) | `(50, 28, 58)` | `#3A1C32` |
| Colour B (lighter purple) | `(70, 42, 78)` | `#462A4E` |

The pattern alternates between the two colours on a 4-pixel grid:

```
A A A A B B B B A A A A B B B B
A A A A B B B B A A A A B B B B
A A A A B B B B A A A A B B B B
A A A A B B B B A A A A B B B B
B B B B A A A A B B B B A A A A
B B B B A A A A B B B B A A A A
B B B B A A A A B B B B A A A A
B B B B A A A A B B B B A A A A
```

### Why Purple?

The entire distance palette runs through red → orange → yellow → green → blue →
gray.  Purple/magenta occupies a completely different hue axis — it is
**impossible** to confuse with any valid distance at any range.

### When Does It Appear?

| Situation | `valid` mask | Visual |
|-----------|:------------|:-------|
| Ray hits geometry | `True` | Distance colour from palette |
| Ray escapes bounding box (finite) | `True` | Far gray (clamped to `max_distance`) |
| Ray result is non-finite (NaN/Inf) | `False` | **Purple checkerboard** |
| Sensor not yet initialised | `False` | **Purple checkerboard** |

Note: Rays that escape the scene bounding box but return a finite (max)
distance are treated as **valid far readings** (faint gray), not missing data.
Only truly non-finite ray results produce the checkerboard.

---

## 5. Rendering Pipeline

```
Environment (SDF/DAG sphere trace)
    │
    │  metric distance per ray → log1p normalisation → [0, 1] float32
    │  valid_mask: bool per ray (isfinite check)
    │
    ▼
Wire Format (DistanceMatrix via ZMQ/msgpack)
    │
    │  depth[1, Az, El]   float32 [0, 1]
    │  valid_mask[1, Az, El]  bool
    │
    ▼
Dashboard Deserialization
    │
    │  depth_to_observer_palette(depth, valid)
    │    1. Denormalize: metres = expm1(depth x ln(1 + 100))
    │    2. Log scale:   t = log1p(m) / (log1p(m) + ln(1 + focus_m))
    │    3. LUT index:   idx = clamp(t x 1023, 0, 1023)
    │    4. Colour:      bgr = lut[idx, 0, :]     (O(1) array lookup)
    │    5. Fog-of-war:  stamp purple checkerboard on ~valid
    │
    ▼
BGR uint8 image → panel rendering
```

---

## 6. Consumer Surfaces

Every surface below uses the same palette pipeline, ensuring colours mean the
same thing everywhere in the dashboard.

### 6.1  Primary Dashboard Panels (app.py → renderers.py)

| Panel | Renderer Function | Palette Entry Point | Missing Data |
|:------|:------------------|:-------------------|:-------------|
| **Half-Sphere (Actor 0)** | `render_first_person()` | `depth_to_observer_palette()` | Purple checkerboard via `_apply_fog_of_war()` |
| **Front Depth Grid** | `render_front_depth_grid()` | `depth_to_observer_palette()` | Confidence-based alpha fade + checkerboard |
| **Hemisphere Heatmap** | `render_front_hemisphere_heatmap()` | `depth_to_observer_palette()` | Purple checkerboard on invalid projections |
| **Forward Polar Scan** | `render_forward_polar()` | `_observer_lut()` direct LUT | N/A (scan-boundary rendering, filled interior only) |
| **Lidar Colorbar** | `_draw_lidar_colorbar()` | `_observer_lut()` gradient strip | N/A (legend, not data) |
| **Distance HUD Text** | `distance_color()` | O(1) LUT lookup | N/A (scalar colour for text/labels) |

### 6.2  Recorder (recorder.py)

| Surface | Original | Now |
|:--------|:---------|:----|
| `_depth_to_bgr()` — depth matrix image | `cv2.COLORMAP_TURBO` | `depth_to_observer_palette()` |
| `_distance_color()` — polar/BEV labels | Hard-coded step function | `distance_color()` from renderers |

### 6.3  Occupancy View (occupancy_view.py)

| Surface | Original | Now |
|:--------|:---------|:----|
| Occupied cells colouring | `cv2.COLORMAP_TURBO` | `_observer_lut()` + `_log_color_t_array()` |

### 6.4  CLI Snapshot (cli.py)

| Surface | Original | Now |
|:--------|:---------|:----|
| Raw distance matrix PNG | `depth_to_viridis()` | `depth_to_observer_palette()` |

### 6.5  Dashboard Snapshot (app.py)

| Surface | Original | Now |
|:--------|:---------|:----|
| Full 360° panoramic PNG | `depth_to_viridis()` | `depth_to_observer_palette()` |

---

## 7. LUT Implementation Details

All surfaces share a single **1024-bin continuous lookup table**
(`_observer_lut()`) that is lazily built once at first access:

1. Compute 16 anchor `t`-positions from the metric palette anchors using the
   default `focus_m`.
2. Sample 1024 uniform positions in `t` in `[0, 1]`.
3. Interpolate each of the B, G, R channels from the 16 control points
   via `np.interp` (one-time build cost).
4. Store as `(1024, 1, 3)` BGR `uint8` array.

At render time, pixel colouring is O(1) array indexing with no per-frame
interpolation:
```python
idx = np.clip(t * 1023, 0, 1023).astype(np.int32)
colour = lut[idx, 0, :]
```

Both `depth_to_observer_palette()` (array path) and `distance_color()`
(scalar path) share this same LUT and O(1) lookup strategy.  The previous
256-bin LUT and per-frame `np.interp` across three channels have been
retired.  With 1024 bins the quantisation step is under 0.1 % of the
spectrum width, yielding sub-pixel-smooth gradients indistinguishable
from continuous interpolation.

---

## 8. Colour Perception Guide

### Reading the dashboard at a glance:

| What You See | What It Means |
|:------------|:-------------|
| **Bright red / red-orange** | Imminent collision (< 0.3 m) — actor is touching or about to touch geometry |
| **Orange / gold** | Close obstacle (0.3 – 1.5 m) — danger zone, actor should manoeuvre |
| **Yellow-green** | Near-field transition (1.5 – 3.0 m) — approaching safe clearance |
| **Fresh green** | Comfortable navigation (3.0 – 6.0 m) — good clearance |
| **Teal** | Mid-range structure (6.0 – 12.0 m) — walls/objects within room scale |
| **Steel / medium blue** | Away structure (12.0 – 35.0 m) — far walls, corridors |
| **Dark blue shades** | Deep far (35.0 – 75.0 m) — building-scale or open exteriors |
| **Gray shades** | Distant / horizon (75.0 m – ∞) — six distinct gray levels fade to near-black |
| **Purple checkerboard** | **Missing data** — sensor gap, no valid ray result for this direction |

### Quick distance estimation from colour:
- If it's **warm** (red/orange/yellow family) → **under 3 m** — watch out
- If it's **green** → **3–6 m** — safe zone
- If it's **blue** → **6–75 m** — structural features at some distance
- If it's **gray** → **75 m+** — very far, approaching infinity
- If it's **purple tiles** → **no data** — sensor blind spot

---

## 9. Source Files

| File | Role |
|:-----|:-----|
| `projects/auditor/src/navi_auditor/dashboard/renderers.py` | Palette definition, LUT, all rendering functions |
| `projects/auditor/src/navi_auditor/dashboard/occupancy_view.py` | 2D occupancy map (imports LUT) |
| `projects/auditor/src/navi_auditor/dashboard/app.py` | Dashboard layout (calls renderers) |
| `projects/auditor/src/navi_auditor/recorder.py` | Legacy OpenCV dashboard (imports palette) |
| `projects/auditor/src/navi_auditor/cli.py` | CLI snapshot tool (imports palette) |
| `projects/auditor/tests/unit/test_renderers.py` | 39 unit tests covering all palette surfaces |

---

*Last updated: April 2026*
