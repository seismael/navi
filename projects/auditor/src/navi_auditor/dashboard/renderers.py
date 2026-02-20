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
    "compute_nav_metrics",
    "depth_to_viridis",
    "distance_color",
    "draw_semantic_legend",
    "overlay_overhead_annotations",
    "render_bev_occupancy",
    "render_first_person",
    "render_forward_polar",
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
    """Convert 2-D float32 depth to Viridis-coloured BGR with fog-of-war."""
    if valid is not None and np.any(valid):
        valid_vals = depth[valid]
        lo = float(np.min(valid_vals))
        hi = float(np.max(valid_vals))
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
    """Dense first-person projection via inverse remap.

    Uses ``cv2.remap()`` to sample a semantic-coloured + depth-shaded
    panorama at every output pixel — producing a fully filled, humanly
    readable environment view instead of sparse dots.

    Parameters
    ----------
    depth
        ``(az_bins, el_bins)`` normalised [0, 1] depth.
    semantic
        ``(az_bins, el_bins)`` integer semantic class IDs.
    valid
        ``(az_bins, el_bins)`` boolean validity mask.
    width, height
        Output image dimensions in pixels.
    pitch
        Robot pitch in radians (positive = look up).  Shifts the
        horizon line for a tilt-aware view.

    Returns
    -------
    ``(bgr_image, center_distance_m)``
    """
    az_bins, el_bins = depth.shape

    # Quantise pitch to centidegrees for cache key stability
    pitch_q = int(round(np.rad2deg(pitch) * 100.0))

    # Focal length & horizon (also used for HUD elements)
    f = width / (2.0 * np.tan(np.deg2rad(MAIN_FOV_DEG) / 2.0))
    horizon_y = int(np.clip(height * 0.54 + pitch * f, 0, height - 1))

    # ── background gradient (visible only where sensor has no data) ──
    img = _make_bg_gradient(width, height, horizon_y)

    if az_bins == 0 or el_bins == 0:
        return img, VIEW_RANGE_M

    # ── build semantic-coloured + depth-shaded panorama ──────────────
    # panorama shape: (el_bins, az_bins, 3) — note transposed for remap
    sem_lut = _semantic_color_lut()
    sem_clamped = np.clip(semantic, 0, _MAX_SEMANTIC_ID - 1)
    pano_base = sem_lut[sem_clamped].astype(np.float32)  # (az, el, 3)

    # Depth shading: closer = brighter, farther = darker
    d_norm = np.clip(depth, 0.0, 1.0)
    shade = np.maximum(0.25, 1.0 - d_norm * 0.75)  # [0.25 .. 1.0]
    pano_shaded = (pano_base * shade[:, :, np.newaxis]).astype(np.uint8)

    # Transpose to (el, az, 3) — cv2.remap treats dim-0 as rows (el)
    pano_img = np.ascontiguousarray(pano_shaded.transpose(1, 0, 2))
    valid_f32 = valid.astype(np.float32).T  # (el, az) for remap

    # ── inverse-project every screen pixel to panorama coords ────────
    map_x, map_y = _build_remap_tables(width, height, az_bins, el_bins, pitch_q)

    remapped = cv2.remap(
        pano_img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid_remapped = cv2.remap(
        valid_f32, map_x, map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    # ── composite: remapped data over gradient background ────────────
    hit_mask = valid_remapped > 0.5
    img[hit_mask] = remapped[hit_mask]

    # Fog-of-war on pixels that mapped to valid panorama coords but
    # had no sensor hit (in-FOV but occluded/out-of-range)
    in_fov = (
        (map_x >= 0) & (map_x < az_bins)
        & (map_y >= 0) & (map_y < el_bins)
    )
    fog_mask = in_fov & (~hit_mask)
    _apply_fog_of_war(img, fog_mask)

    # Light blur for anti-aliasing at bin boundaries
    img = cv2.GaussianBlur(img, (3, 3), 0.6)

    # ── HUD overlays ────────────────────────────────────────────────
    # Horizon line
    cv2.line(img, (0, horizon_y), (width - 1, horizon_y), (60, 60, 60), 1)

    # Floor grid perspective lines
    for i in range(1, 7):
        y_line = int(horizon_y + (height - horizon_y) * (i / 7.0))
        cv2.line(img, (0, y_line), (width - 1, y_line), (35, 45, 55), 1)

    # Centre distance
    center_col = az_bins // 2
    center_band = valid[max(0, center_col - 2): min(az_bins, center_col + 3), :]
    center_depth = depth[max(0, center_col - 2): min(az_bins, center_col + 3), :]
    if np.any(center_band):
        center_m = float(np.min(center_depth[center_band]) * VIEW_RANGE_M)
    else:
        center_m = VIEW_RANGE_M

    # Green crosshair at viewport centre
    cx_fp = width // 2
    cy_fp = height // 2
    cv2.line(img, (cx_fp - 12, cy_fp), (cx_fp + 12, cy_fp), (0, 255, 0), 1)
    cv2.line(img, (cx_fp, cy_fp - 12), (cx_fp, cy_fp + 12), (0, 255, 0), 1)

    # Pitch indicator text
    pitch_deg = np.rad2deg(pitch)
    if abs(pitch_deg) > 0.5:
        label = f"pitch {pitch_deg:+.1f}\xb0"
        cv2.putText(
            img, label, (cx_fp - 40, 24),
            _HUD_FONT, 0.45, _HEADING_COLOR, 1,
        )

    return img, center_m


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
    """Render forward 180-degree scan as a smoothed polar plot."""
    panel = np.full((h, w, 3), 18, dtype=np.uint8)
    az_bins = depth.shape[0]
    if az_bins == 0:
        return panel

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
    radius = int(min(w, h) * 0.45)

    for ring_m in (1.0, 2.0, 5.0, 10.0):
        rr = int(radius * min(1.0, ring_m / VIEW_RANGE_M))
        cv2.ellipse(panel, (cx, cy), (rr, rr), 0, 180, 360, (90, 90, 90), 1)

    for deg in range(-90, 91, 15):
        theta = np.deg2rad(deg)
        x2 = int(cx + radius * np.sin(theta))
        y2 = int(cy - radius * np.cos(theta))
        cv2.line(panel, (cx, cy), (x2, y2), (60, 60, 60), 1)

    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, az_bins)
    for i in range(1, az_bins):
        for scan_idx in (i - 1, i):
            dist = float(min_d[scan_idx])
            rr = int(radius * min(1.0, dist / VIEW_RANGE_M))
            th = angles[scan_idx]
            x = int(cx + rr * np.sin(th))
            y = int(cy - rr * np.cos(th))
            cv2.line(panel, (cx, cy), (x, y), distance_color(dist), 2)

    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)
    cv2.arrowedLine(
        panel, (cx, cy), (cx, cy - 24), (0, 255, 255), 2, tipLength=0.4,
    )
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

    # Range rings
    for r in (
        int(min(h, w) * 0.14),
        int(min(h, w) * 0.28),
        int(min(h, w) * 0.42),
    ):
        cv2.circle(panel, (cx, cy), r, _GUIDE_COLOR, 1)

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
        cv2.line(panel, trail_points[i - 1], trail_points[i], _TRAIL_COLOR, 2)

    if trail_points:
        cv2.circle(panel, trail_points[-1], 4, (255, 0, 255), thickness=-1)

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
