"""Pure NumPy rendering functions for Ghost-Matrix dashboard panels.

All functions accept NumPy arrays and return BGR ``uint8`` images —
no Qt, no OpenCV GUI, no GPU dependency.  They can be tested and
profiled independently of the dashboard UI layer.

This module extracts the static renderers previously embedded in
``LiveDashboard`` and adds perceptually-uniform Viridis/Magma
colormaps with fog-of-war hatching for masked regions.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__: list[str] = [
    "SEMANTIC_COLORS",
    "VIEW_RANGE_M",
    "add_orientation_guides",
    "center_forward_azimuth",
    "compute_nav_metrics",
    "depth_to_viridis",
    "distance_color",
    "draw_semantic_legend",
    "extract_forward_fov",
    "overlay_overhead_annotations",
    "render_bev_occupancy",
    "render_first_person",
    "render_forward_polar",
    "turbo_color_bgr",
    "zoom_overhead",
]

# ── visual constants ─────────────────────────────────────────────────
VIEW_RANGE_M: float = 10.0
FWD_FOV_DEG: float = 120.0
MAIN_FOV_DEG: float = 95.0

# First-person gradient colours (BGR)
_FP_SKY_TOP: tuple[int, int, int] = (80, 40, 10)
_FP_SKY_BOT: tuple[int, int, int] = (160, 100, 50)
_FP_FLOOR_TOP: tuple[int, int, int] = (50, 70, 90)
_FP_FLOOR_BOT: tuple[int, int, int] = (25, 35, 45)

# Semantic class colours (BGR)
SEMANTIC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (50, 50, 50),        # AIR
    1: (140, 140, 140),     # WALL
    2: (110, 160, 100),     # FLOOR
    3: (140, 120, 100),     # CEILING
    4: (80, 200, 240),      # PILLAR — bright cyan
    5: (60, 150, 240),      # RAMP — orange
    6: (100, 100, 240),     # OBSTACLE — red
    7: (180, 180, 180),     # ROAD — silver
    8: (140, 100, 70),      # BUILDING
    9: (240, 230, 140),     # WINDOW — light cyan
    10: (255, 0, 255),      # TARGET — bright magenta
}

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TITLE_TEXT_COLOR = (235, 235, 235)
_GUIDE_COLOR = (90, 90, 90)
_TRAIL_COLOR = (60, 220, 220)
_HEADING_COLOR = (0, 255, 255)

# ── Turbo LUT (256 entries, BGR) ──────────────────────────────────────
_TURBO_LUT: np.ndarray | None = None


def _turbo_lut() -> np.ndarray:
    """Return (or lazily build) a 256x1x3 BGR uint8 Turbo lookup table.

    Turbo is Google's perceptually-uniform rainbow colormap designed
    specifically for depth/distance data.  Near = warm red/yellow,
    far = cool blue/indigo.
    """
    global _TURBO_LUT
    if _TURBO_LUT is not None:
        return _TURBO_LUT

    # 12-point Turbo control curve (RGB 0-1)
    control_rgb = np.array([
        [0.190, 0.072, 0.232],  # 0 — dark indigo (far)
        [0.120, 0.240, 0.640],  # 1 — deep blue
        [0.140, 0.460, 0.860],  # 2 — vivid blue
        [0.130, 0.660, 0.900],  # 3 — cyan
        [0.180, 0.820, 0.720],  # 4 — teal-green
        [0.360, 0.910, 0.440],  # 5 — lime-green
        [0.580, 0.950, 0.240],  # 6 — yellow-green
        [0.790, 0.930, 0.150],  # 7 — yellow
        [0.950, 0.820, 0.120],  # 8 — amber
        [0.990, 0.620, 0.100],  # 9 — orange
        [0.950, 0.370, 0.080],  # 10 — red-orange
        [0.840, 0.160, 0.080],  # 11 — deep red (near)
    ], dtype=np.float32)
    x_ctrl = np.linspace(0.0, 255.0, len(control_rgb))
    x_full = np.arange(256, dtype=np.float32)
    r = np.interp(x_full, x_ctrl, control_rgb[:, 0])
    g = np.interp(x_full, x_ctrl, control_rgb[:, 1])
    b = np.interp(x_full, x_ctrl, control_rgb[:, 2])
    bgr = np.stack([b, g, r], axis=-1)
    _TURBO_LUT = (bgr * 255.0).astype(np.uint8).reshape(256, 1, 3)
    return _TURBO_LUT


def turbo_color_bgr(depth_norm: float) -> tuple[int, int, int]:
    """Map normalised depth [0, 1] to a BGR Turbo colour.

    0 = near (warm red), 1 = far (cool blue).
    """
    lut = _turbo_lut()
    # Invert so 0 (near) → high index (red), 1 (far) → low index (blue)
    idx = int(np.clip((1.0 - depth_norm) * 255.0, 0, 255))
    b, g, r = int(lut[idx, 0, 0]), int(lut[idx, 0, 1]), int(lut[idx, 0, 2])
    return (b, g, r)


# ── Viridis LUT (256 entries, BGR) ───────────────────────────────────
_VIRIDIS_LUT: np.ndarray | None = None


def _viridis_lut() -> np.ndarray:
    """Return (or lazily build) a 256x1x3 BGR uint8 Viridis lookup table."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is not None:
        return _VIRIDIS_LUT

    control_rgb = np.array([
        [0.267, 0.004, 0.329],
        [0.282, 0.140, 0.458],
        [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.134, 0.658, 0.517],
        [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150],
        [0.993, 0.906, 0.144],
    ], dtype=np.float32)
    x_ctrl = np.linspace(0.0, 255.0, len(control_rgb))
    x_full = np.arange(256, dtype=np.float32)
    r = np.interp(x_full, x_ctrl, control_rgb[:, 0])
    g = np.interp(x_full, x_ctrl, control_rgb[:, 1])
    b = np.interp(x_full, x_ctrl, control_rgb[:, 2])
    bgr = np.stack([b, g, r], axis=-1)
    _VIRIDIS_LUT = (bgr * 255.0).astype(np.uint8).reshape(256, 1, 3)
    return _VIRIDIS_LUT


def _apply_fog_of_war(img: np.ndarray, invalid_mask: np.ndarray) -> None:
    """Draw semi-transparent diagonal hatching over invalid regions."""
    if not np.any(invalid_mask):
        return
    ys, xs = np.nonzero(invalid_mask)
    stripe = ((xs.astype(np.int32) + ys.astype(np.int32)) % 8) < 3
    fog_pixels = np.where(stripe)
    img[ys[fog_pixels], xs[fog_pixels]] = (
        img[ys[fog_pixels], xs[fog_pixels]] * 0.3
        + np.array([45, 25, 55], dtype=np.float32) * 0.7
    ).astype(np.uint8)


# ── public renderers ─────────────────────────────────────────────────


def depth_to_viridis(
    depth: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Convert 2-D float32 depth to Viridis-coloured BGR with fog-of-war.

    Uses percentile-based contrast stretch so close-range indoor scenes
    still show clear depth variation instead of a single-colour heatmap.
    """
    if valid is not None and np.any(valid):
        valid_vals = depth[valid]
        lo = float(np.percentile(valid_vals, 1))
        hi = float(np.percentile(valid_vals, 99))
    else:
        lo, hi = 0.0, 1.0

    span = max(hi - lo, 1e-4)
    normalised = 1.0 - np.clip((depth - lo) / span, 0.0, 1.0)
    indices = (normalised * 255.0).astype(np.uint8)
    lut = _viridis_lut()
    coloured = lut[indices, 0, :]

    if valid is not None:
        _apply_fog_of_war(coloured, ~valid)

    return coloured


# ── Remap table cache for dense first-person projection ──────────────
_REMAP_CACHE: dict[tuple[int, int, int, int, int], tuple[np.ndarray, np.ndarray]] = {}
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


def extract_forward_fov(*arrays: np.ndarray, fov_degrees: float = FWD_FOV_DEG) -> tuple[np.ndarray, ...]:
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


def _build_remap_tables(
    width: int,
    height: int,
    az_bins: int,
    el_bins: int,
    pitch_q: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (or return cached) inverse-projection tables for cv2.remap.

    Maps every (screen_x, screen_y) → (az_bin, el_bin) as float32,
    so cv2.remap can bilinearly sample the semantic panorama.
    """
    key = (width, height, az_bins, el_bins, pitch_q)
    cached = _REMAP_CACHE.get(key)
    if cached is not None:
        return cached

    # Pitch in radians from quantised centidegrees
    pitch_rad = float(pitch_q) * 0.01 * (np.pi / 180.0)

    # Focal length for MAIN_FOV_DEG
    f = width / (2.0 * np.tan(np.deg2rad(MAIN_FOV_DEG) / 2.0))

    # Horizon shifts with pitch
    horizon_y = height * 0.54 + pitch_rad * f

    # Screen pixel grids (every output pixel)
    sx = np.arange(width, dtype=np.float32)
    sy = np.arange(height, dtype=np.float32)
    sx_grid, sy_grid = np.meshgrid(sx, sy)  # (H, W) each

    # Inverse pinhole → view angles
    view_az = np.arctan2(sx_grid - width * 0.5, f)  # horizontal angle
    view_el = np.arctan2(horizon_y - sy_grid, f)     # vertical angle

    # Map view angles → source bin coordinates (float)
    az_lo = -np.deg2rad(FWD_FOV_DEG / 2.0)
    az_hi = np.deg2rad(FWD_FOV_DEG / 2.0)
    el_lo = -np.deg2rad(35.0)
    el_hi = np.deg2rad(35.0)

    # Normalise to [0, bins-1]
    map_x = ((view_az - az_lo) / (az_hi - az_lo) * (az_bins - 1)).astype(np.float32)
    map_y = ((el_hi - view_el) / (el_hi - el_lo) * (el_bins - 1)).astype(np.float32)

    # Keep cache bounded
    if len(_REMAP_CACHE) > 8:
        _REMAP_CACHE.clear()
    _REMAP_CACHE[key] = (map_x, map_y)
    return map_x, map_y


def _make_bg_gradient(width: int, height: int, horizon_y: int) -> np.ndarray:
    """Build sky/floor gradient background (vectorised, no Python loop)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    sky_h = max(1, horizon_y)
    floor_h = max(1, height - horizon_y)

    # Sky gradient
    sky_t = np.linspace(0.0, 1.0, sky_h, dtype=np.float32).reshape(-1, 1, 1)
    top = np.array(_FP_SKY_TOP, dtype=np.float32)
    bot = np.array(_FP_SKY_BOT, dtype=np.float32)
    img[:sky_h] = (top * (1.0 - sky_t) + bot * sky_t).astype(np.uint8)

    # Floor gradient
    floor_t = np.linspace(0.0, 1.0, floor_h, dtype=np.float32).reshape(-1, 1, 1)
    f_top = np.array(_FP_FLOOR_TOP, dtype=np.float32)
    f_bot = np.array(_FP_FLOOR_BOT, dtype=np.float32)
    end = min(horizon_y + floor_h, height)
    img[horizon_y:end] = (f_top * (1.0 - floor_t[: end - horizon_y])
                          + f_bot * floor_t[: end - horizon_y]).astype(np.uint8)
    return img


def render_first_person(
    depth: np.ndarray,
    semantic: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    pitch: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Dense depth-fill first-person view via equirect-to-rectilinear remap.

    Uses ``cv2.remap`` to bilinearly sample the equirectangular depth
    data for every output pixel, producing a fully-filled perspective
    projection with Turbo depth colouring.  Invalid pixels fall through
    to a sky/floor gradient background.

    Parameters
    ----------
    depth
        ``(az_bins, el_bins)`` normalised [0, 1] depth.
    semantic
        ``(az_bins, el_bins)`` integer semantic class IDs (reserved).
    valid
        ``(az_bins, el_bins)`` boolean validity mask.
    width, height
        Output image dimensions in pixels.
    pitch
        Robot pitch in radians (positive = look up).

    Returns
    -------
    ``(bgr_image, center_distance_m)``
    """
    _ = semantic  # reserved for future semantic overlay
    az_bins, el_bins = depth.shape
    colorbar_w = 28
    view_w = width - colorbar_w

    if az_bins == 0 or el_bins == 0:
        img = np.full((height, width, 3), 8, dtype=np.uint8)
        return img, VIEW_RANGE_M

    # Pitch quantisation for remap cache (centidegrees)
    pitch_q = round(float(np.rad2deg(pitch)) * 100.0)
    flen = view_w / (2.0 * np.tan(np.deg2rad(MAIN_FOV_DEG) / 2.0))
    horizon_y = int(np.clip(height * 0.54 + pitch * flen, 0, height - 1))

    # ── Background: sky / floor gradient ─────────────────────────────
    img = _make_bg_gradient(width, height, horizon_y)

    # ── Inverse-projection remap tables ──────────────────────────────
    map_x, map_y = _build_remap_tables(view_w, height, az_bins, el_bins, pitch_q)

    # Source images transposed to (el, az) so remap X→az col, Y→el row
    depth_src = depth.T.astype(np.float32)
    valid_src = valid.T.astype(np.float32)

    depth_proj = cv2.remap(
        depth_src, map_x, map_y, cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=1.0,
    )
    valid_proj = cv2.remap(
        valid_src, map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    vis_mask = valid_proj > 0.5

    # ── Turbo colormap with local contrast stretch ─────────────────
    lut = _turbo_lut()
    depth_clamped = np.clip(depth_proj, 0.0, 1.0)

    # Auto-range: stretch actual valid depth range across full LUT
    # so close-range indoor scenes still show clear depth variation
    if np.any(vis_mask):
        valid_depths = depth_clamped[vis_mask]
        d_lo = float(np.percentile(valid_depths, 1))
        d_hi = float(np.percentile(valid_depths, 99))
        d_span = max(d_hi - d_lo, 0.01)
        depth_stretched = np.clip((depth_clamped - d_lo) / d_span, 0.0, 1.0)
    else:
        depth_stretched = depth_clamped

    turbo_idx = np.clip(
        (1.0 - depth_stretched) * 255.0, 0, 255,
    ).astype(np.uint8)
    depth_colored = lut[turbo_idx, 0, :]  # (height, view_w, 3)

    # Composite depth onto gradient (view area only)
    view_region = img[:, :view_w]
    view_region[vis_mask] = depth_colored[vis_mask]

    # ── Subtle grid overlay ──────────────────────────────────────────
    cv2.line(img, (0, horizon_y), (view_w - 1, horizon_y), (55, 55, 55), 1)
    cv2.line(img, (view_w // 2, 0), (view_w // 2, height - 1), (40, 40, 40), 1)

    for i in range(1, 8):
        y_up = int(horizon_y - (horizon_y * i / 8.0))
        y_dn = int(horizon_y + ((height - horizon_y) * i / 8.0))
        if 0 <= y_up < height:
            cv2.line(img, (0, y_up), (view_w - 1, y_up), (28, 28, 28), 1)
        if 0 <= y_dn < height:
            cv2.line(img, (0, y_dn), (view_w - 1, y_dn), (28, 28, 28), 1)

    for i in range(1, 12):
        x_line = int(view_w * i / 12.0)
        cv2.line(img, (x_line, 0), (x_line, height - 1), (28, 28, 28), 1)

    # ── Colorbar + crosshair ─────────────────────────────────────────
    _draw_lidar_colorbar(img, view_w, 0, colorbar_w, height)
    _draw_lidar_crosshair(img, view_w // 2, horizon_y)

    # Centre distance readout
    center_col = az_bins // 2
    center_band = valid[max(0, center_col - 2): min(az_bins, center_col + 3), :]
    center_depth = depth[max(0, center_col - 2): min(az_bins, center_col + 3), :]
    if np.any(center_band):
        center_m = float(np.min(center_depth[center_band]) * VIEW_RANGE_M)
    else:
        center_m = VIEW_RANGE_M

    dist_label = f"{center_m:.1f}m"
    cv2.putText(
        img, dist_label,
        (view_w // 2 + 16, horizon_y - 6),
        _HUD_FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
    )

    pitch_deg = float(np.rad2deg(pitch))
    if abs(pitch_deg) > 0.5:
        label = f"pitch {pitch_deg:+.1f}\xb0"
        cv2.putText(
            img, label, (view_w // 2 - 40, 24),
            _HUD_FONT, 0.45, _HEADING_COLOR, 1, cv2.LINE_AA,
        )

    return img, center_m


def _draw_lidar_colorbar(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw a vertical Turbo colorbar with distance labels on the right edge."""
    lut = _turbo_lut()
    bar_margin = 4
    bar_x = x + bar_margin
    bar_w = w - bar_margin * 2
    bar_top = y + 30
    bar_bot = y + h - 30
    bar_h = max(1, bar_bot - bar_top)

    # Draw the gradient bar
    for row in range(bar_h):
        t = row / max(bar_h - 1, 1)
        # Top of bar = near (red), bottom = far (blue)
        idx = int(np.clip((1.0 - t) * 255.0, 0, 255))
        color = (int(lut[idx, 0, 0]), int(lut[idx, 0, 1]), int(lut[idx, 0, 2]))
        cv2.line(
            img,
            (bar_x, bar_top + row),
            (bar_x + bar_w, bar_top + row),
            color, 1,
        )

    # Border
    cv2.rectangle(
        img, (bar_x, bar_top), (bar_x + bar_w, bar_bot),
        (80, 80, 80), 1,
    )

    # Distance labels at key positions
    labels = [(0.0, "0m"), (0.2, "2m"), (0.5, "5m"), (1.0, "10m")]
    for frac, label in labels:
        row_y = int(bar_top + frac * (bar_h - 1))
        # Tick mark
        cv2.line(img, (bar_x - 2, row_y), (bar_x, row_y), (140, 140, 140), 1)
        # Label text
        cv2.putText(
            img, label,
            (bar_x - 2, row_y - 3) if frac == 0.0 else (bar_x - 2, row_y + 4),
            _HUD_FONT, 0.30, (160, 160, 160), 1, cv2.LINE_AA,
        )

    # Title
    cv2.putText(
        img, "DIST",
        (bar_x, bar_top - 8),
        _HUD_FONT, 0.35, (160, 160, 160), 1, cv2.LINE_AA,
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
    """Map distance in meters to a BGR HUD colour."""
    d = float(max(0.0, distance_m))
    if d <= 0.5:
        return (0, 0, 255)
    if d <= 1.5:
        return (0, 100, 255)
    if d <= 3.0:
        return (0, 255, 200)
    if d <= 6.0:
        return (100, 255, 0)
    if d <= 10.0:
        return (200, 200, 100)
    return (50, 50, 50)


def render_forward_polar(
    depth: np.ndarray,
    valid: np.ndarray,
    w: int,
    h: int,
    scan_history: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Render forward 180° scan as a continuous Turbo-filled polar plot.

    Each pixel is rasterised through polar-coordinate transform and
    coloured by depth using the Turbo colormap, producing a smooth,
    professional-grade radar-style display.
    """
    panel = np.full((h, w, 3), 18, dtype=np.uint8)
    az_bins = depth.shape[0]
    if az_bins == 0:
        return panel

    # ── per-azimuth minimum distance (with smoothing) ───────────────
    min_d = np.full((az_bins,), VIEW_RANGE_M, dtype=np.float32)
    for i in range(az_bins):
        v = valid[i]
        if np.any(v):
            min_d[i] = float(np.min(depth[i][v]) * VIEW_RANGE_M)

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

    # Interpolate scan distance at each pixel's azimuth
    bin_lo = np.floor(bin_idx).astype(np.int32)
    bin_hi = np.clip(bin_lo + 1, 0, az_bins - 1)
    frac = bin_idx - bin_lo.astype(np.float32)
    scan_dist = min_d[bin_lo] * (1.0 - frac) + min_d[bin_hi] * frac

    # Pixel radius corresponding to scan distance
    scan_r_px = (scan_dist / VIEW_RANGE_M) * max_r

    # Fill mask: inside scan boundary AND within 180° forward arc
    in_arc = (pixel_angle >= az_lo) & (pixel_angle <= az_hi) & (dy >= -4)
    fill_mask = in_arc & (pixel_dist <= scan_r_px) & (pixel_dist > 6)

    # Depth at each filled pixel (normalised 0-1)
    filled_depth = np.where(
        fill_mask,
        np.clip(pixel_dist / np.maximum(scan_r_px, 1.0), 0.0, 1.0),
        0.0,
    )
    # Map to actual distance for Turbo colouring
    filled_dist_m = filled_depth * scan_dist
    depth_norm = np.clip(filled_dist_m / VIEW_RANGE_M, 0.0, 1.0)

    # Turbo LUT colouring
    lut = _turbo_lut()
    turbo_idx = np.clip((1.0 - depth_norm) * 255.0, 0, 255).astype(np.uint8)
    colored = lut[turbo_idx, 0, :]  # (h, w, 3)

    # Subtle radial darkening for depth perception
    darken = np.clip(
        1.0 - 0.25 * (pixel_dist / max(max_r, 1)), 0.4, 1.0,
    ).astype(np.float32)
    colored_dark = (colored.astype(np.float32) * darken[:, :, np.newaxis]).astype(
        np.uint8,
    )
    panel[fill_mask] = colored_dark[fill_mask]

    # ── Scan boundary polyline (white, anti-aliased) ───────────────
    boundary_angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, az_bins)
    boundary_pts: list[tuple[int, int]] = []
    for k in range(az_bins):
        rd = float(min_d[k] / VIEW_RANGE_M) * max_r
        bx = int(cx + rd * np.sin(boundary_angles[k]))
        by = int(cy - rd * np.cos(boundary_angles[k]))
        boundary_pts.append((bx, by))
    if len(boundary_pts) >= 2:
        pts_arr = np.array(boundary_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts_arr], False, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Range rings with metric labels ────────────────────────────
    ring_specs = [
        (1.0, "1m"), (2.0, "2m"), (5.0, "5m"), (10.0, "10m"),
    ]
    for ring_m, ring_label in ring_specs:
        rr = int(max_r * min(1.0, ring_m / VIEW_RANGE_M))
        cv2.ellipse(
            panel, (cx, cy), (rr, rr), 0, 180, 360,
            (80, 80, 80), 1, cv2.LINE_AA,
        )
        # Label on right side of ring
        lx = cx + rr + 3
        ly = cy + 4
        if lx < w - 20:
            cv2.putText(
                panel, ring_label, (lx, ly),
                _HUD_FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA,
            )

    # ── Ego marker ───────────────────────────────────────────────
    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.arrowedLine(
        panel, (cx, cy), (cx, cy - 22), (0, 255, 255), 2, tipLength=0.4,
    )

    # ── Angle labels ─────────────────────────────────────────────
    cv2.putText(panel, "FWD", (cx - 12, cy - max_r - 6),
                _HUD_FONT, 0.38, _HEADING_COLOR, 1, cv2.LINE_AA)
    cv2.putText(panel, "L", (cx - max_r - 12, cy + 4),
                _HUD_FONT, 0.38, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(panel, "R", (cx + max_r + 4, cy + 4),
                _HUD_FONT, 0.38, _TITLE_TEXT_COLOR, 1, cv2.LINE_AA)

    return panel


def render_bev_occupancy(
    depth: np.ndarray,
    valid: np.ndarray,
    w: int,
    h: int,
    pose_history: list[tuple[float, float, float]] | None = None,
) -> np.ndarray:
    """Render 360-degree observation as top-down occupancy map."""
    panel = np.full((h, w, 3), 20, dtype=np.uint8)
    az_bins = depth.shape[0]
    if az_bins == 0:
        return panel

    cx = w // 2
    cy = h // 2
    scale = (min(w, h) * 0.45) / VIEW_RANGE_M
    angles = np.linspace(-np.pi, np.pi, az_bins, endpoint=False)

    # Distance rings for reference
    for ring_m in (1.0, 2.0, 5.0, 10.0):
        rr = int(ring_m * scale)
        if rr < max(w, h):
            cv2.circle(panel, (cx, cy), rr, _GUIDE_COLOR, 1)

    for i in range(az_bins):
        v = valid[i]
        if not np.any(v):
            continue
        d = float(np.min(depth[i][v]) * VIEW_RANGE_M)
        d_clamped = min(VIEW_RANGE_M, d)
        x = int(cx + d_clamped * np.sin(angles[i]) * scale)
        y = int(cy - d_clamped * np.cos(angles[i]) * scale)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel, (x, y), 3, distance_color(d_clamped), thickness=-1)

    if pose_history and len(pose_history) >= 2:
        cur_x, cur_z, _ = pose_history[-1]
        trail: list[tuple[int, int]] = []
        for hx, hz, _hyaw in pose_history:
            dx = hx - cur_x
            dz = hz - cur_z
            px = int(cx + dx * scale)
            py = int(cy + dz * scale)
            if 0 <= px < w and 0 <= py < h:
                trail.append((px, py))
        for i in range(1, len(trail)):
            cv2.line(panel, trail[i - 1], trail[i], _TRAIL_COLOR, 2)

    cv2.arrowedLine(
        panel, (cx, cy), (cx, cy - 20), (0, 255, 255), 2, tipLength=0.4,
    )
    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)

    # Compass labels
    cv2.putText(panel, "N", (cx - 5, 16), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
    cv2.putText(
        panel, "E", (w - 16, cy + 4), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1,
    )
    cv2.putText(
        panel, "S", (cx - 5, h - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1,
    )
    cv2.putText(panel, "W", (6, cy + 4), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)

    # Forward direction marker
    cv2.putText(
        panel, "FWD", (cx - 12, 32), _HUD_FONT, 0.35, _HEADING_COLOR, 1,
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
            panel, rlabel,
            (cx + r + 3, cy - 2),
            _HUD_FONT, 0.32, (140, 140, 140), 1, cv2.LINE_AA,
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
        panel, (cx, cy), (hx_px, cy), _HEADING_COLOR, 2, tipLength=0.3,
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
        panel, left_label, (8, h - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1,
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
        frame, (x, y), (x + box_w, y + box_h), (18, 18, 18), thickness=-1,
    )
    cv2.rectangle(
        frame, (x, y), (x + box_w, y + box_h), (60, 60, 60), thickness=1,
    )
    for legend_idx, (label, color) in enumerate(legend_items):
        yy = y + 6 + legend_idx * row_h
        cv2.rectangle(
            frame, (x + 8, yy), (x + 20, yy + 12), color, thickness=-1,
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
