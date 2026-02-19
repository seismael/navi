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
    "compute_nav_metrics",
    "depth_to_viridis",
    "distance_color",
    "draw_semantic_legend",
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
}

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TITLE_TEXT_COLOR = (235, 235, 235)
_GUIDE_COLOR = (90, 90, 90)
_TRAIL_COLOR = (60, 220, 220)
_HEADING_COLOR = (0, 255, 255)

# ── Viridis LUT (256 entries, BGR) ───────────────────────────────────
# Pre-baked from matplotlib Viridis — close=warm/bright, far=cool/dark.
_VIRIDIS_LUT: np.ndarray | None = None


def _viridis_lut() -> np.ndarray:
    """Return (or lazily build) a 256x1x3 BGR uint8 Viridis lookup table."""
    global _VIRIDIS_LUT
    if _VIRIDIS_LUT is not None:
        return _VIRIDIS_LUT

    # Key control points — RGB floats from the Viridis specification.
    # Interpolated linearly for 256 entries.
    control_rgb = np.array([
        [0.267, 0.004, 0.329],   #   0 — dark purple  (far / faint)
        [0.282, 0.140, 0.458],   #  36
        [0.254, 0.265, 0.530],   #  72
        [0.207, 0.372, 0.553],   # 108
        [0.164, 0.471, 0.558],   # 144
        [0.128, 0.567, 0.551],   # 180
        [0.134, 0.658, 0.517],   # 200
        [0.478, 0.821, 0.318],   # 220
        [0.741, 0.873, 0.150],   # 240
        [0.993, 0.906, 0.144],   # 255 — bright yellow (close / big)
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
        img[ys[fog_pixels], xs[fog_pixels]] * 0.3 + np.array([45, 25, 55], dtype=np.float32) * 0.7
    ).astype(np.uint8)


# ── public renderers ─────────────────────────────────────────────────

def depth_to_viridis(
    depth: np.ndarray,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Convert 2-D float32 depth to Viridis-coloured BGR with fog-of-war.

    Close objects → bright yellow/white, far → dark purple.
    Invalid (masked) regions → semi-transparent diagonal hatching.
    """
    if valid is not None and np.any(valid):
        valid_vals = depth[valid]
        lo = float(np.min(valid_vals))
        hi = float(np.max(valid_vals))
    else:
        lo, hi = 0.0, 1.0

    span = max(hi - lo, 1e-4)
    # Invert: close → high index (bright), far → low index (dark)
    normalised = 1.0 - np.clip((depth - lo) / span, 0.0, 1.0)
    indices = (normalised * 255.0).astype(np.uint8)
    lut = _viridis_lut()
    coloured = lut[indices, 0, :]  # (H, W, 3) BGR

    if valid is not None:
        _apply_fog_of_war(coloured, ~valid)

    return coloured


def render_first_person(
    depth: np.ndarray,
    semantic: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, float]:
    """Project depth samples into first-person screen space with semantic colouring.

    Returns ``(bgr_image, center_distance_m)``.
    """
    az_bins, el_bins = depth.shape
    horizon_y = int(height * 0.54)

    img = np.zeros((height, width, 3), dtype=np.uint8)
    top = np.array(_FP_SKY_TOP, dtype=np.float32)
    bottom = np.array(_FP_SKY_BOT, dtype=np.float32)
    sky_h = max(1, horizon_y)
    for row in range(sky_h):
        t = row / max(1, sky_h - 1)
        img[row, :] = (top * (1.0 - t) + bottom * t).astype(np.uint8)

    floor_top = np.array(_FP_FLOOR_TOP, dtype=np.float32)
    floor_bot = np.array(_FP_FLOOR_BOT, dtype=np.float32)
    for row in range(horizon_y, height):
        t = (row - horizon_y) / max(1, height - horizon_y - 1)
        img[row, :] = (floor_top * (1.0 - t) + floor_bot * t).astype(np.uint8)

    az = np.linspace(-np.deg2rad(FWD_FOV_DEG / 2.0), np.deg2rad(FWD_FOV_DEG / 2.0), az_bins)
    el = np.linspace(np.deg2rad(35.0), -np.deg2rad(35.0), el_bins)
    az_grid, el_grid = np.meshgrid(az, el, indexing="ij")

    dist_m = np.clip(depth, 0.0, 1.0) * VIEW_RANGE_M
    valid_pts = valid & (dist_m > 1e-4)
    if not np.any(valid_pts):
        return img, VIEW_RANGE_M

    x = dist_m * np.cos(el_grid) * np.cos(az_grid)
    y = dist_m * np.sin(el_grid)
    z = dist_m * np.cos(el_grid) * np.sin(az_grid)

    f = width / (2.0 * np.tan(np.deg2rad(MAIN_FOV_DEG) / 2.0))
    px = (width * 0.5 + (z / np.maximum(x, 1e-4)) * f).astype(np.int32)
    py = (horizon_y - (y / np.maximum(x, 1e-4)) * f).astype(np.int32)

    in_view = valid_pts & (x > 0.1)
    in_view &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(in_view):
        return img, VIEW_RANGE_M

    xx = x[in_view]
    pp_x = px[in_view]
    pp_y = py[in_view]
    sem = semantic[in_view]
    d_norm = np.clip(depth[in_view], 0.0, 1.0)

    order = np.argsort(xx)[::-1]
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    for idx in order:
        u = int(pp_x[idx])
        v = int(pp_y[idx])
        z_depth = float(xx[idx])
        if z_depth >= float(zbuf[v, u]):
            continue
        zbuf[v, u] = z_depth

        base = np.array(SEMANTIC_COLORS.get(int(sem[idx]), (150, 150, 150)), dtype=np.float32)
        shade = max(0.2, 1.0 - float(d_norm[idx]) * 0.8)
        color = (base * shade).astype(np.uint8)
        img[v, u] = color
        if v + 1 < height:
            img[v + 1, u] = color

    img = cv2.GaussianBlur(img, (3, 3), 0.0)

    for i in range(1, 7):
        y_line = int(horizon_y + (height - horizon_y) * (i / 7.0))
        cv2.line(img, (0, y_line), (width - 1, y_line), (35, 45, 55), 1)

    center_col = az_bins // 2
    center_band = valid[max(0, center_col - 2) : min(az_bins, center_col + 3), :]
    center_depth = depth[max(0, center_col - 2) : min(az_bins, center_col + 3), :]
    if np.any(center_band):
        center_m = float(np.min(center_depth[center_band]) * VIEW_RANGE_M)
    else:
        center_m = VIEW_RANGE_M

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
        for idx in (i - 1, i):
            dist = float(min_d[idx])
            rr = int(radius * min(1.0, dist / VIEW_RANGE_M))
            th = angles[idx]
            x = int(cx + rr * np.sin(th))
            y = int(cy - rr * np.cos(th))
            cv2.line(panel, (cx, cy), (x, y), distance_color(dist), 2)

    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)
    cv2.arrowedLine(panel, (cx, cy), (cx, cy - 24), (0, 255, 255), 2, tipLength=0.4)
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

    for i in range(az_bins):
        v = valid[i]
        if not np.any(v):
            continue
        d = float(np.min(depth[i][v]) * VIEW_RANGE_M)
        d_clamped = min(VIEW_RANGE_M, d)
        x = int(cx + d_clamped * np.sin(angles[i]) * scale)
        y = int(cy - d_clamped * np.cos(angles[i]) * scale)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(panel, (x, y), 2, distance_color(d_clamped), thickness=-1)

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
            cv2.line(panel, trail[i - 1], trail[i], _TRAIL_COLOR, 1)

    cv2.arrowedLine(panel, (cx, cy), (cx, cy - 20), (0, 255, 255), 2, tipLength=0.4)
    cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)
    cv2.putText(panel, "N", (cx - 5, 16), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
    cv2.putText(panel, "E", (w - 16, cy + 4), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
    cv2.putText(panel, "S", (cx - 5, h - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
    cv2.putText(panel, "W", (6, cy + 4), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
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
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (18, 18, 18), thickness=-1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (60, 60, 60), thickness=1)
    for idx, (label, color) in enumerate(legend_items):
        yy = y + 6 + idx * row_h
        cv2.rectangle(frame, (x + 8, yy), (x + 20, yy + 12), color, thickness=-1)
        cv2.putText(
            frame,
            label,
            (x + 28, yy + 11),
            _HUD_FONT,
            0.42,
            _TITLE_TEXT_COLOR,
            1,
        )
