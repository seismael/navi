"""Pure NumPy rendering functions for Ghost-Matrix dashboard panels.

All functions accept NumPy arrays and return BGR ``uint8`` images —
no Qt, no OpenCV GUI, no GPU dependency.  They can be tested and
profiled independently of the dashboard UI layer.

The single observer palette — a threat-graded logarithmic colour scale
from red (collision) through green/blue (navigation) to gray (far/void)
— is pre-computed as a 1024-bin continuous LUT and shared across all
dashboard, recorder, CLI, and occupancy surfaces.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

__all__: list[str] = [
    "SEMANTIC_COLORS",
    "VIEW_RANGE_M",
    "add_orientation_guides",
    "center_forward_azimuth",
    "compute_nav_metrics",
    "depth_to_observer_palette",
    "distance_color",
    "draw_semantic_legend",
    "extract_forward_fov",
    "overlay_overhead_annotations",
    "render_bev_occupancy",
    "render_front_depth_grid",
    "render_front_hemisphere_heatmap",
    "render_first_person",
    "render_forward_polar",
    "zoom_overhead",
]

# ── visual constants ─────────────────────────────────────────────────
VIEW_RANGE_M: float = 30.0
FWD_FOV_DEG: float = 180.0

# ── Adjustable logarithmic colour focus ──────────────────────────────
# Maps [0, ∞) → [0, 1) via  t = log1p(d) / (log1p(d) + C).
# ``focus_m`` controls where 50 % of the colour spectrum is allocated:
#   focus_m = 20  ->  half the spectrum covers 0-20 m (navigation detail)
#   focus_m = 30  ->  half the spectrum covers 0-30 m (wider view)
# Lower values allocate more colour diversity to close-range structure.
_DEFAULT_FOCUS_M: float = 20.0

# ── High-resolution continuous LUT ───────────────────────────────────
# 1024-bin pre-computed palette eliminates per-frame np.interp overhead.
# Colours are looked up via O(1) array indexing from the normalised
# log-scale position ``t``.
_LUT_SIZE: int = 1024
_LUT_MAX_IDX: int = _LUT_SIZE - 1

# Palette control anchors — fixed metric distances and their BGR colours.
# 4-band atmospheric perspective: ORANGE → GREEN → BLUE → GRAY.
# Orange alert zone (0-1 m) transitions smoothly into green near-field,
# then each band desaturates (washes out) as distance grows.
_PALETTE_METER_ANCHORS: tuple[float, ...] = (
    0.0, 0.15, 0.5, 1.0, 2.5, 5.0, 10.0, 18.0, 30.0, 45.0,
    65.0, 90.0, 130.0, 250.0, 500.0,
)
_PALETTE_BGR: tuple[tuple[float, float, float], ...] = (
    # -- ORANGE band (0-1 m): vivid orange -> yellow-green transition --
    (0.0, 120.0, 255.0),    # vivid orange         (contact / extreme near)
    (5.0, 158.0, 230.0),    # warm amber           (close proximity)
    (12.0, 198.0, 170.0),   # yellow               (transitioning to green)
    # -- GREEN band (1-5 m): fresh green -> pale teal-green --
    (30.0, 220.0, 90.0),    # fresh green          (near-field entry)
    (72.0, 200.0, 60.0),    # medium green         (comfortable, fading)
    (110.0, 178.0, 55.0),   # pale teal-green      (washed out)
    # -- BLUE band (5-45 m): teal-blue -> pale washed blue --
    (152.0, 155.0, 48.0),   # teal                 (green-to-blue transition)
    (185.0, 128.0, 42.0),   # medium blue          (mid-range peak)
    (180.0, 120.0, 65.0),   # soft blue            (fading)
    (165.0, 125.0, 90.0),   # pale blue            (washed out)
    (148.0, 128.0, 108.0),  # very pale blue       (almost gray)
    # -- GRAY band (65 m - inf): blue-gray -> dim -> near-black --
    (128.0, 120.0, 112.0),  # blue-gray            (far transition)
    (105.0, 100.0, 96.0),   # medium gray          (receding)
    (85.0, 82.0, 80.0),     # dark gray            (distant)
    (62.0, 60.0, 58.0),     # dim gray             (fading horizon)
    (35.0, 34.0, 33.0),     # near-black           (void / infinity)
)

# ── Missing-data tile pattern ────────────────────────────────────────
# Distinctive checkerboard in dark magenta / purple tones — clearly
# distinguishable from any valid distance color in the palette.
_MISSING_TILE_SIZE: int = 4
_MISSING_COLOR_A: np.ndarray = np.array([50, 28, 58], dtype=np.uint8)  # dark purple
_MISSING_COLOR_B: np.ndarray = np.array([70, 42, 78], dtype=np.uint8)  # lighter purple


def _log_color_t(distance_m: float, focus_m: float = _DEFAULT_FOCUS_M) -> float:
    """Map metric distance [0, ∞) → [0, 1) via unbounded log scale."""
    v = math.log1p(max(0.0, distance_m))
    return v / (v + math.log1p(max(0.1, focus_m)))


def _log_color_t_array(
    distance_m: np.ndarray, focus_m: float = _DEFAULT_FOCUS_M,
) -> np.ndarray:
    """Vectorised unbounded log colour scale for NumPy arrays."""
    v = np.log1p(np.maximum(distance_m, 0.0))
    return v / (v + math.log1p(max(0.1, focus_m)))


# ── 1024-bin continuous observer LUT ─────────────────────────────────
# Built once at first access by interpolating the 16 palette anchors to
# 1024 bins.  At render time, colour lookup is a single O(1) array index
# instead of per-pixel np.interp across three channels.
_OBSERVER_LUT: np.ndarray | None = None


def _observer_lut() -> np.ndarray:
    """Return (or lazily build) a 1024x1x3 BGR uint8 continuous LUT.

    Index 0 = near (red), index 1023 = far (gray).  Built by sampling
    the 16 palette anchor colours at 1024 uniform positions in the
    ``[0, 1]`` colour-scale range for sub-pixel-smooth gradients.
    """
    global _OBSERVER_LUT
    if _OBSERVER_LUT is not None:
        return _OBSERVER_LUT
    # Compute anchor positions in t-space using the default focus.
    ctrl_x = np.array(
        [_log_color_t(m) for m in _PALETTE_METER_ANCHORS] + [1.0],
        dtype=np.float32,
    )
    ctrl_bgr = np.array(_PALETTE_BGR, dtype=np.float32)
    x_full = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32)
    b = np.interp(x_full, ctrl_x, ctrl_bgr[:, 0])
    g = np.interp(x_full, ctrl_x, ctrl_bgr[:, 1])
    r = np.interp(x_full, ctrl_x, ctrl_bgr[:, 2])
    bgr = np.stack([b, g, r], axis=-1)
    _OBSERVER_LUT = np.clip(bgr, 0, 255).astype(np.uint8).reshape(_LUT_SIZE, 1, 3)
    return _OBSERVER_LUT

# First-person gradient colours (BGR)
_FP_SKY_TOP: tuple[int, int, int] = (80, 40, 10)
_FP_SKY_BOT: tuple[int, int, int] = (160, 100, 50)
_FP_FLOOR_TOP: tuple[int, int, int] = (50, 70, 90)
_FP_FLOOR_BOT: tuple[int, int, int] = (25, 35, 45)

# Semantic class colours (BGR)
SEMANTIC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (50, 50, 50),  # AIR
    1: (140, 140, 140),  # WALL
    2: (110, 160, 100),  # FLOOR
    3: (140, 120, 100),  # CEILING
    4: (80, 200, 240),  # PILLAR — bright cyan
    5: (60, 150, 240),  # RAMP — orange
    6: (100, 100, 240),  # OBSTACLE — red
    7: (180, 180, 180),  # ROAD — silver
    8: (140, 100, 70),  # BUILDING
    9: (240, 230, 140),  # WINDOW — light cyan
    10: (255, 0, 255),  # TARGET — bright magenta
}

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TITLE_TEXT_COLOR = (235, 235, 235)
_GUIDE_COLOR = (90, 90, 90)
_TRAIL_COLOR = (60, 220, 220)
_HEADING_COLOR = (0, 255, 255)




def _apply_fog_of_war(img: np.ndarray, invalid_mask: np.ndarray) -> None:
    """Stamp a checkerboard tile pattern over missing-data regions.

    Uses a small dark-purple checkerboard that is immediately
    distinguishable from any valid distance colour in the palette.
    """
    if not np.any(invalid_mask):
        return
    ys, xs = np.nonzero(invalid_mask)
    checker = (
        (xs.astype(np.int32) // _MISSING_TILE_SIZE)
        + (ys.astype(np.int32) // _MISSING_TILE_SIZE)
    ) % 2
    img[ys, xs] = np.where(
        checker[:, np.newaxis] == 0,
        _MISSING_COLOR_A,
        _MISSING_COLOR_B,
    )


# ── public renderers ─────────────────────────────────────────────────


def depth_to_observer_palette(
    depth: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    fog_of_war: bool = True,
    env_max_distance: float = 100.0,
    focus_m: float = _DEFAULT_FOCUS_M,
) -> np.ndarray:
    """Convert log-normalized depth to a continuous observer palette.

    Input depth is already log-normalized by the environment as
    ``log1p(d) / log1p(max_distance)``.  This function denormalises back
    to metric metres and maps through a smooth, unbounded logarithmic
    colour scale via O(1) LUT indexing.

    Parameters
    ----------
    env_max_distance:
        The environment's normalisation constant (for denormalisation
        only — the colour mapping itself is parameter-free).
    focus_m:
        The metric distance that consumes 50 % of the colour spectrum.
        Lower values (e.g. 5-10) maximise near-field colour diversity.
    """
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)

    # Denormalise to metric metres, then apply unbounded log scale.
    metric = np.expm1(depth * math.log1p(env_max_distance))
    t = _log_color_t_array(metric, focus_m=focus_m)

    # O(1) LUT indexing — replaces per-pixel np.interp.
    lut = _observer_lut()
    indices = np.clip(t * _LUT_MAX_IDX, 0, _LUT_MAX_IDX).astype(np.int32)
    coloured = lut[indices, 0, :]

    if fog_of_war and valid is not None:
        _apply_fog_of_war(coloured, ~valid)

    return coloured


def render_front_depth_grid(
    depth: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the exact front-half `(azimuth, elevation)` bins as a dense heatmap.

    This keeps the observer view faithful to the actual front-half spherical
    observation: for the canonical `256x48` contract and `180` degrees FOV this
    is an exact `128x48` depth grid, enlarged with nearest-neighbour scaling.
    """
    panel = np.full((height, width, 3), (14, 10, 8), dtype=np.uint8)
    az_bins, el_bins = depth.shape
    if az_bins == 0 or el_bins == 0:
        return panel

    pad_x = 16
    pad_top = 24
    pad_bottom = 28
    target_w = max(1, width - 2 * pad_x)
    target_h = max(1, height - pad_top - pad_bottom)

    depth_src = depth.T.astype(np.float32)
    valid_src = valid.T.astype(np.float32)
    weighted_src = depth_src * valid_src

    weighted_up = cv2.resize(weighted_src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    valid_up = cv2.resize(valid_src, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Smooth sparse observations into a readable observer view while preserving
    # the original front-half geometry trend and keeping low-confidence regions dark.
    weighted_blur = cv2.GaussianBlur(weighted_up, (0, 0), sigmaX=7.0, sigmaY=5.0)
    valid_blur = cv2.GaussianBlur(valid_up, (0, 0), sigmaX=7.0, sigmaY=5.0)
    dense_depth = np.divide(weighted_blur, np.maximum(valid_blur, 1e-4))
    confidence = np.clip(valid_blur, 0.0, 1.0)

    heatmap = depth_to_observer_palette(dense_depth, confidence > 0.02).astype(np.float32)
    background = np.full((target_h, target_w, 3), (18, 14, 10), dtype=np.float32)
    alpha = np.clip(confidence[..., None] ** 0.85, 0.0, 1.0)
    blended = (heatmap * alpha) + (background * (1.0 - alpha))
    panel[pad_top : pad_top + target_h, pad_x : pad_x + target_w] = blended.astype(np.uint8)

    cx = pad_x + target_w // 2
    cy = pad_top + target_h // 2
    cv2.line(panel, (cx, pad_top), (cx, pad_top + target_h - 1), (58, 58, 58), 1, cv2.LINE_AA)
    cv2.line(panel, (pad_x, cy), (pad_x + target_w - 1, cy), (58, 58, 58), 1, cv2.LINE_AA)
    cv2.rectangle(
        panel, (pad_x - 1, pad_top - 1), (pad_x + target_w, pad_top + target_h), (70, 70, 70), 1
    )

    cv2.putText(panel, "UP", (cx - 10, 18), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        panel, "DOWN", (cx - 24, height - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )
    cv2.putText(
        panel, "LEFT", (pad_x, height - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )
    right_size = cv2.getTextSize("RIGHT", _HUD_FONT, 0.45, 1)[0]
    cv2.putText(
        panel,
        "RIGHT",
        (width - right_size[0] - pad_x, height - 8),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return panel


def render_front_hemisphere_heatmap(
    depth: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Render a 180-degree front hemisphere as a filled semi-ellipse heatmap."""
    panel = np.full((height, width, 3), 12, dtype=np.uint8)
    az_bins, el_bins = depth.shape
    if az_bins == 0 or el_bins == 0:
        return panel

    pad_x = 12
    pad_top = 12
    pad_bottom = 18
    cx = width * 0.5
    cy = height - pad_bottom
    rx = max((width - 2 * pad_x) * 0.5, 1.0)
    ry = max(height - pad_top - pad_bottom, 1.0)

    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)

    x_norm = (x_grid - cx) / rx
    inside_x = np.abs(x_norm) <= 1.0
    dome = np.sqrt(np.clip(1.0 - np.square(x_norm), 0.0, 1.0))
    top_y = cy - ry * dome
    inside_y = (y_grid >= top_y) & (y_grid <= cy)
    inside = inside_x & inside_y

    denom = np.maximum(cy - top_y, 1.0)
    elevation_t = np.clip((y_grid - top_y) / denom, 0.0, 1.0)
    azimuth_t = np.clip((x_norm + 1.0) * 0.5, 0.0, 1.0)

    depth_src = depth.T.astype(np.float32)
    valid_src = valid.T.astype(np.float32)
    map_x = (azimuth_t * (az_bins - 1)).astype(np.float32)
    map_y = (elevation_t * (el_bins - 1)).astype(np.float32)

    depth_proj = cv2.remap(
        depth_src,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    valid_proj = (
        cv2.remap(
            valid_src,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        > 0.5
    )

    heatmap = depth_to_observer_palette(depth_proj, valid_proj & inside)
    panel[inside] = heatmap[inside]

    boundary_pts: list[tuple[int, int]] = []
    for x in range(int(pad_x), int(width - pad_x)):
        xn = (x - cx) / rx
        if abs(xn) > 1.0:
            continue
        y = round(cy - ry * np.sqrt(max(0.0, 1.0 - xn * xn)))
        boundary_pts.append((x, y))
    if len(boundary_pts) >= 2:
        pts = np.array(boundary_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, (110, 110, 110), 1, cv2.LINE_AA)

    base_y = round(cy)
    cv2.line(
        panel,
        (int(cx), int(top_y[:, width // 2].min())),
        (int(cx), base_y),
        _GUIDE_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.line(panel, (int(cx - rx), base_y), (int(cx + rx), base_y), (60, 60, 60), 1, cv2.LINE_AA)

    cv2.putText(
        panel,
        "LEFT",
        (pad_x, max(20, height - 6)),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    right_size = cv2.getTextSize("RIGHT", _HUD_FONT, 0.45, 1)[0]
    cv2.putText(
        panel,
        "RIGHT",
        (max(pad_x, width - right_size[0] - pad_x), max(20, height - 6)),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return panel


_SEMANTIC_LUT: np.ndarray | None = None

_MAX_SEMANTIC_ID: int = 11


def center_forward_azimuth(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Roll aligned `(azimuth, elevation)` arrays so forward lands at the center.

    The canonical environment convention emits forward-facing rays at azimuth bin
    `0`. Dashboard panels that take a center-cropped forward view must first roll
    the panorama so the forward seam is centered before slicing.
    """
    if not arrays:
        return ()
    az_bins = int(arrays[0].shape[0])
    shift = az_bins // 2
    centered: list[np.ndarray] = []
    for array in arrays:
        if array.shape[0] != az_bins:
            msg = "All arrays must share the same azimuth dimension"
            raise ValueError(msg)
        centered.append(np.roll(array, shift=shift, axis=0))
    return tuple(centered)


def extract_forward_fov(
    *arrays: np.ndarray, fov_degrees: float = FWD_FOV_DEG
) -> tuple[np.ndarray, ...]:
    """Return aligned forward-FOV slices from canonical `(azimuth, elevation)` arrays."""
    if not arrays:
        return ()
    centered = center_forward_azimuth(*arrays)
    az_bins = int(centered[0].shape[0])
    fov_fraction = float(fov_degrees) / 360.0
    fov_bins = max(1, int(az_bins * fov_fraction))
    center_bin = az_bins // 2
    fov_lo = center_bin - fov_bins // 2
    fov_hi = fov_lo + fov_bins
    return tuple(array[fov_lo:fov_hi, :] for array in centered)


def _semantic_color_lut() -> np.ndarray:
    """Return a (MAX_ID, 3) uint8 BGR lookup table for semantic classes."""
    global _SEMANTIC_LUT
    if _SEMANTIC_LUT is not None:
        return _SEMANTIC_LUT
    lut = np.full((_MAX_SEMANTIC_ID, 3), 150, dtype=np.uint8)
    for sid, bgr in SEMANTIC_COLORS.items():
        if 0 <= sid < _MAX_SEMANTIC_ID:
            lut[sid] = bgr
    _SEMANTIC_LUT = lut
    return _SEMANTIC_LUT


def render_first_person(
    depth: np.ndarray,
    semantic: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    pitch: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Render the exact centered 180-degree half-sphere as a dense heatmap."""
    centered_depth, centered_semantic, centered_valid = extract_forward_fov(
        depth,
        semantic,
        valid,
        fov_degrees=FWD_FOV_DEG,
    )
    del centered_semantic

    az_bins, el_bins = centered_depth.shape
    if az_bins == 0 or el_bins == 0:
        return np.full((height, width, 3), 15, dtype=np.uint8), 0.0

    pad_x = 4
    pad_top = 4
    pad_bottom = 4
    avail_w = max(1, width - 2 * pad_x)
    avail_h = max(1, height - pad_top - pad_bottom)

    depth_src = centered_depth.T.astype(np.float32)
    valid_src = centered_valid.T.astype(np.uint8)
    heatmap_src = depth_to_observer_palette(depth_src, centered_valid.T, fog_of_war=False)

    # Preserve the observation's native aspect ratio (e.g. 128:48 = 2.67:1)
    # to prevent distortion when panel dimensions change with actor count.
    src_h, src_w = heatmap_src.shape[:2]  # (el_bins, az_bins)
    src_aspect = src_w / max(src_h, 1)
    avail_aspect = avail_w / max(avail_h, 1)
    if avail_aspect > src_aspect:
        target_h = avail_h
        target_w = max(1, int(avail_h * src_aspect))
    else:
        target_w = avail_w
        target_h = max(1, int(avail_w / src_aspect))

    blended = cv2.resize(heatmap_src, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    valid_up = cv2.resize(valid_src, (target_w, target_h), interpolation=cv2.INTER_NEAREST) > 0
    _apply_fog_of_war(blended, ~valid_up)

    canvas = np.full((height, width, 3), (14, 10, 8), dtype=np.uint8)
    # Center the aspect-correct heatmap within the available area
    off_x = pad_x + (avail_w - target_w) // 2
    off_y = pad_top + (avail_h - target_h) // 2
    canvas[off_y : off_y + target_h, off_x : off_x + target_w] = blended.astype(np.uint8)

    cx = off_x + target_w // 2
    cy = off_y + target_h // 2
    cv2.line(canvas, (cx, off_y), (cx, off_y + target_h - 1), (58, 58, 58), 1, cv2.LINE_AA)
    pitch_offset = int(np.clip(float(pitch), -1.0, 1.0) * (target_h * 0.35))
    horizon_y = int(np.clip(cy + pitch_offset, off_y, off_y + target_h - 1))
    cv2.line(
        canvas, (off_x, horizon_y), (off_x + target_w - 1, horizon_y), (58, 58, 58), 1, cv2.LINE_AA
    )
    cv2.rectangle(
        canvas, (off_x - 1, off_y - 1), (off_x + target_w, off_y + target_h), (70, 70, 70), 1
    )

    cv2.putText(canvas, "UP", (cx - 10, off_y + 16), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA)
    down_size = cv2.getTextSize("DOWN", _HUD_FONT, 0.45, 1)[0]
    cv2.putText(
        canvas, "DOWN", (cx - down_size[0] // 2, off_y + target_h - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )
    left_y = min(off_y + target_h - 10, max(off_y + 16, cy + 5))
    cv2.putText(
        canvas, "LEFT", (off_x, left_y), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )
    right_size = cv2.getTextSize("RIGHT", _HUD_FONT, 0.45, 1)[0]
    cv2.putText(
        canvas,
        "RIGHT",
        (max(off_x, off_x + target_w - right_size[0]), left_y),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    pitch_deg = float(np.rad2deg(pitch))
    if abs(pitch_deg) > 0.5:
        cv2.putText(
            canvas,
            f"pitch {pitch_deg:+.1f}\xb0",
            (cx - 40, off_y + 30),
            _HUD_FONT,
            0.45,
            _HEADING_COLOR,
            1,
            cv2.LINE_AA,
        )

    mid_az = az_bins // 2
    mid_el = el_bins // 2
    center_value = centered_depth[mid_az, mid_el] if centered_valid[mid_az, mid_el] else 0.0
    return canvas, float(center_value)


def _draw_lidar_colorbar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a vertical observer-palette colorbar with distance labels."""
    lut = _observer_lut()
    bar_margin = 4
    bar_x = x + bar_margin
    bar_w = w - bar_margin * 2
    bar_top = y + 30
    bar_bot = y + h - 30
    bar_h = max(1, bar_bot - bar_top)

    # Draw the gradient bar
    for row in range(bar_h):
        t = row / max(bar_h - 1, 1)
        # Top of bar = near (red, idx 0), bottom = far (gray, idx 1023)
        idx = int(min(t * _LUT_MAX_IDX, _LUT_MAX_IDX))
        color = (int(lut[idx, 0, 0]), int(lut[idx, 0, 1]), int(lut[idx, 0, 2]))
        cv2.line(
            img,
            (bar_x, bar_top + row),
            (bar_x + bar_w, bar_top + row),
            color,
            1,
        )

    # Border
    cv2.rectangle(
        img,
        (bar_x, bar_top),
        (bar_x + bar_w, bar_bot),
        (80, 80, 80),
        1,
    )

    # Distance labels placed at their log-scale bar positions
    label_specs: list[tuple[float, str]] = [
        (0.0, "0m"), (1.0, "1m"), (5.0, "5m"),
        (20.0, "20m"), (60.0, "60m"),
    ]
    for dist_m, label in label_specs:
        frac = _log_color_t(dist_m)
        row_y = int(bar_top + frac * (bar_h - 1))
        # Tick mark
        cv2.line(img, (bar_x - 2, row_y), (bar_x, row_y), (140, 140, 140), 1)
        # Label text
        cv2.putText(
            img,
            label,
            (bar_x - 2, row_y - 3) if dist_m == 0.0 else (bar_x - 2, row_y + 4),
            _HUD_FONT,
            0.30,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # Title
    cv2.putText(
        img,
        "DIST",
        (bar_x, bar_top - 8),
        _HUD_FONT,
        0.35,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )


def _draw_lidar_crosshair(
    img: np.ndarray,
    cx: int,
    cy: int,
) -> None:
    """Draw a bright green targeting crosshair at the viewport centre."""
    gap = 6
    arm = 16
    color = (0, 255, 0)
    # Horizontal arms with gap
    cv2.line(img, (cx - arm, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + gap, cy), (cx + arm, cy), color, 1, cv2.LINE_AA)
    # Vertical arms with gap
    cv2.line(img, (cx, cy - arm), (cx, cy - gap), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + gap), (cx, cy + arm), color, 1, cv2.LINE_AA)
    # Centre dot
    cv2.circle(img, (cx, cy), 2, color, -1, cv2.LINE_AA)


def distance_color(distance_m: float) -> tuple[int, int, int]:
    """Map distance in metres to a BGR HUD colour via O(1) LUT lookup.

    Uses the same continuous ``log1p(d) / (log1p(d) + C)`` mapping and
    1024-bin observer LUT as all other dashboard surfaces.
    """
    t = _log_color_t(distance_m)
    lut = _observer_lut()
    idx = int(min(t * _LUT_MAX_IDX, _LUT_MAX_IDX))
    row = lut[idx, 0]
    return (int(row[0]), int(row[1]), int(row[2]))


def render_forward_polar(
    depth: np.ndarray,
    valid: np.ndarray,
    w: int,
    h: int,
    scan_history: list[np.ndarray] | None = None,
    max_distance_m: float = VIEW_RANGE_M,
    env_max_distance: float = 100.0,
) -> np.ndarray:
    """Render forward 180° scan as a continuous observer-palette polar plot.

    Radial pixel positions use the unbounded log scale so near-field
    structure gets proportionally more screen space and there is no
    hard far-clipping boundary.
    """
    panel = np.full((h, w, 3), 18, dtype=np.uint8)
    az_bins = depth.shape[0]
    if az_bins == 0:
        return panel

    _env_log = math.log1p(env_max_distance)

    # ── per-azimuth minimum metric distance (with smoothing) ─────
    min_d = np.full((az_bins,), float(max_distance_m), dtype=np.float32)
    for i in range(az_bins):
        v = valid[i]
        if np.any(v):
            d_norm = float(np.min(depth[i][v]))
            min_d[i] = float(math.expm1(d_norm * _env_log))

    min_d = np.convolve(
        min_d,
        np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0,
        mode="same",
    )
    if scan_history is not None:
        scan_history.append(min_d.copy())
        if len(scan_history) > 3:
            scan_history[:] = scan_history[-3:]
        hist = np.stack(scan_history, axis=0)
        min_d = np.mean(hist, axis=0)

    cx = w // 2
    cy = h - 12
    max_r = int(min(w, h) * 0.45)

    # ── Radial guide lines (every 30°) ────────────────────────────
    for deg in range(-90, 91, 30):
        theta = np.deg2rad(deg)
        x2 = int(cx + max_r * np.sin(theta))
        y2 = int(cy - max_r * np.cos(theta))
        cv2.line(panel, (cx, cy), (x2, y2), (35, 35, 35), 1, cv2.LINE_AA)

    # ── Log-scaled scan radii for pixel space ─────────────────────
    min_d_log = np.array([_log_color_t(float(d)) for d in min_d], dtype=np.float32)

    # ── Pixel-space polar rasterisation ───────────────────────────
    ys, xs = np.mgrid[0:h, 0:w]
    dx = (xs - cx).astype(np.float32)
    dy = (cy - ys).astype(np.float32)  # +Y = up (toward forward)
    pixel_dist = np.sqrt(dx * dx + dy * dy)
    pixel_angle = np.arctan2(dx, dy)  # [-pi, pi], 0=forward

    # Map pixel angle → azimuth bin index (continuous float)
    az_lo = -np.pi / 2.0
    az_hi = np.pi / 2.0
    bin_idx_f = (pixel_angle - az_lo) / (az_hi - az_lo) * (az_bins - 1)
    bin_idx = np.clip(bin_idx_f, 0, az_bins - 1)

    # Interpolate log-scaled scan radius at each pixel's azimuth
    bin_lo = np.floor(bin_idx).astype(np.int32)
    bin_hi = np.clip(bin_lo + 1, 0, az_bins - 1)
    frac = bin_idx - bin_lo.astype(np.float32)
    scan_t = min_d_log[bin_lo] * (1.0 - frac) + min_d_log[bin_hi] * frac
    scan_r_px = scan_t * max_r

    # Also interpolate metric distances for Turbo colouring
    scan_dist = min_d[bin_lo] * (1.0 - frac) + min_d[bin_hi] * frac

    # Fill mask: inside scan boundary AND within 180° forward arc
    in_arc = (pixel_angle >= az_lo) & (pixel_angle <= az_hi) & (dy >= -4)
    fill_mask = in_arc & (pixel_dist <= scan_r_px) & (pixel_dist > 6)

    # Depth at each filled pixel (fraction within scan boundary)
    filled_depth = np.where(
        fill_mask,
        np.clip(pixel_dist / np.maximum(scan_r_px, 1.0), 0.0, 1.0),
        0.0,
    )
    # Map fraction to metric distance then to unbounded log colour scale
    filled_dist_m = filled_depth * scan_dist
    depth_norm = _log_color_t_array(filled_dist_m)

    # Observer-palette LUT colouring (unified with all dashboard surfaces)
    lut = _observer_lut()
    obs_idx = np.clip(depth_norm * _LUT_MAX_IDX, 0, _LUT_MAX_IDX).astype(np.int32)
    colored = lut[obs_idx, 0, :]  # (h, w, 3)

    # Subtle radial darkening for depth perception
    darken = np.clip(
        1.0 - 0.25 * (pixel_dist / max(max_r, 1)),
        0.4,
        1.0,
    ).astype(np.float32)
    colored_dark = (colored.astype(np.float32) * darken[:, :, np.newaxis]).astype(
        np.uint8,
    )
    panel[fill_mask] = colored_dark[fill_mask]

    # ── Scan boundary polyline (white, anti-aliased) ───────────────
    boundary_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, az_bins)
    boundary_pts: list[tuple[int, int]] = []
    for k in range(az_bins):
        rd = float(min_d_log[k]) * max_r
        bx = int(cx + rd * np.sin(boundary_angles[k]))
        by = int(cy - rd * np.cos(boundary_angles[k]))
        boundary_pts.append((bx, by))
    if len(boundary_pts) >= 2:
        pts_arr = np.array(boundary_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts_arr], False, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Range rings with metric labels (unbounded log scale) ──────
    ring_specs = [
        (1.0, "1m"),
        (5.0, "5m"),
        (20.0, "20m"),
        (60.0, "60m"),
    ]
    for ring_m, ring_label in ring_specs:
        rr = int(max_r * _log_color_t(ring_m))
        cv2.ellipse(
            panel,
            (cx, cy),
            (rr, rr),
            0,
            180,
            360,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        # Label on right side of ring
        lx = cx + rr + 3
        ly = cy + 4
        if lx < w - 20:
            cv2.putText(
                panel,
                ring_label,
                (lx, ly),
                _HUD_FONT,
                0.32,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )

    # ── Ego marker ───────────────────────────────────────────────
    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(
        panel,
        (cx, cy),
        (cx, cy - 22),
        (0, 255, 255),
        2,
        tipLength=0.4,
    )

    # ── Angle labels ─────────────────────────────────────────────
    cv2.putText(
        panel, "FWD", (cx - 12, cy - max_r - 6), _HUD_FONT, 0.38, _HEADING_COLOR, 1, cv2.LINE_AA
    )
    cv2.putText(
        panel, "L", (cx - max_r - 12, cy + 4), _HUD_FONT, 0.38, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )
    cv2.putText(
        panel, "R", (cx + max_r + 4, cy + 4), _HUD_FONT, 0.38, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA
    )

    return panel


def render_bev_occupancy(
    depth: np.ndarray,
    valid: np.ndarray,
    w: int,
    h: int,
    pose_history: list[tuple[float, float, float]] | None = None,
    max_distance_m: float = VIEW_RANGE_M,
    env_max_distance: float = 100.0,
) -> np.ndarray:
    """Render 360-degree observation as top-down occupancy map.

    Dot positions use the unbounded log colour scale so near-field
    structure is expanded and far-field compresses gracefully.
    """
    panel = np.full((h, w, 3), 20, dtype=np.uint8)
    az_bins = depth.shape[0]
    if az_bins == 0:
        return panel

    cx = w // 2
    cy = h // 2
    _max_pixel_r = min(w, h) * 0.45
    _env_log = math.log1p(env_max_distance)
    angles = np.linspace(-np.pi, np.pi, az_bins, endpoint=False)

    # Distance rings (unbounded log scale — no max_distance)
    for ring_m in (1.0, 3.0, 10.0, 30.0):
        rr = int(_log_color_t(ring_m) * _max_pixel_r)
        if rr < max(w, h):
            cv2.circle(panel, (cx, cy), rr, _GUIDE_COLOR, 1)

    for i in range(az_bins):
        v = valid[i]
        if not np.any(v):
            continue
        d_norm = float(np.min(depth[i][v]))
        actual_m = math.expm1(d_norm * _env_log)
        r_frac = _log_color_t(actual_m)
        x = int(cx + r_frac * np.sin(angles[i]) * _max_pixel_r)
        y = int(cy - r_frac * np.cos(angles[i]) * _max_pixel_r)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel, (x, y), 3, distance_color(actual_m), thickness=-1)

    if pose_history and len(pose_history) >= 2:
        cur_x, cur_z, _ = pose_history[-1]
        trail: list[tuple[int, int]] = []
        for hx, hz, _hyaw in pose_history:
            dx_m = hx - cur_x
            dz_m = hz - cur_z
            dist_m = math.sqrt(dx_m * dx_m + dz_m * dz_m)
            if dist_m < 1e-6:
                trail.append((cx, cy))
                continue
            r_frac = _log_color_t(dist_m)
            px = int(cx + (dx_m / dist_m) * r_frac * _max_pixel_r)
            py = int(cy + (dz_m / dist_m) * r_frac * _max_pixel_r)
            if 0 <= px < w and 0 <= py < h:
                trail.append((px, py))
        for i in range(1, len(trail)):
            cv2.line(panel, trail[i - 1], trail[i], _TRAIL_COLOR, 2)

    cv2.arrowedLine(
        panel,
        (cx, cy),
        (cx, cy - 20),
        (0, 255, 255),
        2,
        tipLength=0.4,
    )
    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)

    # Compass labels
    cv2.putText(panel, "N", (cx - 5, 16), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
    cv2.putText(
        panel,
        "E",
        (w - 16, cy + 4),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
    )
    cv2.putText(
        panel,
        "S",
        (cx - 5, h - 8),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
    )
    cv2.putText(panel, "W", (6, cy + 4), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)

    # Forward direction marker
    cv2.putText(
        panel,
        "FWD",
        (cx - 12, 32),
        _HUD_FONT,
        0.35,
        _HEADING_COLOR,
        1,
    )
    return panel


def zoom_overhead(overhead: np.ndarray, zoom: float) -> np.ndarray:
    """Crop and scale overhead minimap by the given zoom factor."""
    h, w = overhead.shape[:2]
    zoom = max(1.0, min(zoom, 6.0))
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = overhead[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay_overhead_annotations(
    panel: np.ndarray,
    zoom: float,
    pose_history: list[tuple[float, float, float]],
) -> None:
    """Draw range guides, ego-rotated trajectory trail, and heading on overhead."""
    h, w = panel.shape[:2]
    cx = w // 2
    cy = h // 2

    # Crosshair at centre
    cv2.line(panel, (cx - 12, cy), (cx + 12, cy), _HEADING_COLOR, 1)
    cv2.line(panel, (cx, cy - 12), (cx, cy + 12), _HEADING_COLOR, 1)

    # Range rings with distance labels
    ring_specs = [
        (int(min(h, w) * 0.14), "5m"),
        (int(min(h, w) * 0.28), "10m"),
        (int(min(h, w) * 0.42), "15m"),
    ]
    for r, rlabel in ring_specs:
        cv2.circle(panel, (cx, cy), r, _GUIDE_COLOR, 1)
        cv2.putText(
            panel,
            rlabel,
            (cx + r + 3, cy - 2),
            _HUD_FONT,
            0.32,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )

    if len(pose_history) < 2:
        return

    cur_x, cur_z, cur_yaw = pose_history[-1]
    world_radius = 18.0 / max(zoom, 1.0)
    scale = (min(w, h) * 0.5) / max(world_radius, 1e-3)

    trail_points: list[tuple[int, int]] = []
    for hx, hz, _ in pose_history:
        dx = hx - cur_x
        dz = hz - cur_z
        rx = dx * np.cos(-cur_yaw) - dz * np.sin(-cur_yaw)
        rz = dx * np.sin(-cur_yaw) + dz * np.cos(-cur_yaw)
        px = int(cx + rx * scale)
        pz = int(cy + rz * scale)
        if 0 <= px < w and 0 <= pz < h:
            trail_points.append((px, pz))

    for i in range(1, len(trail_points)):
        cv2.line(panel, trail_points[i - 1], trail_points[i], _TRAIL_COLOR, 3)

    if trail_points:
        cv2.circle(panel, trail_points[-1], 6, (255, 0, 255), thickness=-1)

    heading_len = 18
    hx_px = int(cx + heading_len)
    cv2.arrowedLine(
        panel,
        (cx, cy),
        (hx_px, cy),
        _HEADING_COLOR,
        2,
        tipLength=0.3,
    )


def add_orientation_guides(
    panel: np.ndarray,
    left_label: str = "LEFT",
    right_label: str = "RIGHT",
) -> None:
    """Draw centre crosshair lines and LEFT/RIGHT labels on a panel."""
    h, w = panel.shape[:2]
    cx = w // 2
    cy = h // 2
    cv2.line(panel, (cx, 0), (cx, h - 1), _GUIDE_COLOR, 1)
    cv2.line(panel, (0, cy), (w - 1, cy), _GUIDE_COLOR, 1)
    cv2.putText(
        panel,
        left_label,
        (8, h - 8),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
    )
    txt_size = cv2.getTextSize(right_label, _HUD_FONT, 0.45, 1)[0]
    cv2.putText(
        panel,
        right_label,
        (max(8, w - txt_size[0] - 8), h - 8),
        _HUD_FONT,
        0.45,
        _TITLE_TEXT_COLOR,
        1,
    )


def compute_nav_metrics(
    depth: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float]:
    """Compute nearest normalized depth in forward/left/right sectors."""
    az_bins = depth.shape[0]
    if az_bins == 0:
        return 1.0, 1.0, 1.0

    def nearest(lo: int, hi: int) -> float:
        lo_clamped = max(0, lo)
        hi_clamped = min(az_bins, hi)
        if hi_clamped <= lo_clamped:
            return 1.0
        sector_depth = depth[lo_clamped:hi_clamped, :]
        sector_valid = valid[lo_clamped:hi_clamped, :]
        if not np.any(sector_valid):
            return 1.0
        return float(np.min(sector_depth[sector_valid]))

    center = az_bins // 2
    span = max(4, az_bins // 8)
    side_span = max(4, az_bins // 6)

    forward = nearest(center - span // 2, center + span // 2)
    left = nearest(center - span // 2 - side_span, center - span // 2)
    right = nearest(center + span // 2, center + span // 2 + side_span)
    return forward, left, right


def draw_semantic_legend(
    frame: np.ndarray,
    x: int,
    y: int,
) -> None:
    """Draw compact semantic legend for human interpretation."""
    legend_items: list[tuple[str, tuple[int, int, int]]] = [
        ("Wall", SEMANTIC_COLORS[1]),
        ("Obstacle", SEMANTIC_COLORS[6]),
        ("Pillar", SEMANTIC_COLORS[4]),
        ("Road", SEMANTIC_COLORS[7]),
        ("Building", SEMANTIC_COLORS[8]),
    ]
    row_h = 18
    box_w = 190
    box_h = 10 + row_h * len(legend_items)
    cv2.rectangle(
        frame,
        (x, y),
        (x + box_w, y + box_h),
        (18, 18, 18),
        thickness=-1,
    )
    cv2.rectangle(
        frame,
        (x, y),
        (x + box_w, y + box_h),
        (60, 60, 60),
        thickness=1,
    )
    for legend_idx, (label, color) in enumerate(legend_items):
        yy = y + 6 + legend_idx * row_h
        cv2.rectangle(
            frame,
            (x + 8, yy),
            (x + 20, yy + 12),
            color,
            thickness=-1,
        )
        cv2.putText(
            frame,
            label,
            (x + 28, yy + 11),
            _HUD_FONT,
            0.42,
            _TITLE_TEXT_COLOR,
            1,
        )
