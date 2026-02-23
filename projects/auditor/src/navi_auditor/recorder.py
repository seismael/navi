"""Recorder and OpenCV-based live matrix dashboard for Ghost-Matrix v2."""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np
import zmq

from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    DistanceMatrix,
    StepRequest,
    deserialize,
    serialize,
)

if TYPE_CHECKING:
    from navi_auditor.config import AuditorConfig
    from navi_auditor.storage.base import AbstractStorageBackend

__all__: list[str] = ["LiveDashboard", "Recorder"]

# ── visual constants ─────────────────────────────────────────────────
# Main-first layout at 1920x1080:
# Left 2/3 = first-person, Right 1/3 = two stacked columns
# (interpreted + raw), each with forward/overhead/panorama rows.
_DISPLAY_W = 1920
_DISPLAY_H = 1080
_MAIN_W = (_DISPLAY_W * 2) // 3
_SIDE_W = _DISPLAY_W - _MAIN_W
_SIDE_COL_W = _SIDE_W // 2
_STACK_H = _DISPLAY_H // 3
_HUD_COLOR = (220, 220, 220)
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HUD_SCALE = 0.50
_HUD_THICK = 1
_PANEL_BORDER_COLOR = (75, 75, 75)
_TITLE_BG_COLOR = (20, 20, 20)
_TITLE_TEXT_COLOR = (235, 235, 235)
_GUIDE_COLOR = (90, 90, 90)
_TRAIL_COLOR = (60, 220, 220)
_HEADING_COLOR = (0, 255, 255)
_DANGER_COLOR = (30, 30, 240)
_VIEW_RANGE_M = 10.0
_FWD_FOV_DEG = 120.0
_MAIN_FOV_DEG = 95.0

# ── first-person renderer constants ──────────────────────────────────
_FP_SKY_TOP: tuple[int, int, int] = (80, 40, 10)      # dark blue sky (BGR)
_FP_SKY_BOT: tuple[int, int, int] = (160, 100, 50)    # lighter horizon sky
_FP_FLOOR_TOP: tuple[int, int, int] = (50, 70, 90)    # near floor (tan/brown)
_FP_FLOOR_BOT: tuple[int, int, int] = (25, 35, 45)    # far floor (darker)

# Brighter semantic colours for first-person walls (BGR)
_FP_SEMANTIC_COLORS: dict[int, tuple[int, int, int]] = {
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


class Recorder:
    """Passive ZMQ subscriber that records v2 stream messages to storage."""

    def __init__(self, config: AuditorConfig, backend: AbstractStorageBackend) -> None:
        self._config = config
        self._backend = backend
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sockets: list[zmq.Socket[bytes]] = []

    def start(self) -> None:
        """Open storage and connect to publishers."""
        self._backend.open(self._config.output_path, mode="w")

        for addr in self._config.sub_addresses:
            sock = self._context.socket(zmq.SUB)
            sock.connect(addr)
            sock.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
            sock.setsockopt(zmq.SUBSCRIBE, TOPIC_ACTION.encode("utf-8"))
            sock.setsockopt(zmq.SUBSCRIBE, TOPIC_TELEMETRY_EVENT.encode("utf-8"))
            self._sockets.append(sock)

    def record_one(self, timeout: int = 1000) -> bool:
        """Try to receive and record one message."""
        poller = zmq.Poller()
        for sock in self._sockets:
            poller.register(sock, zmq.POLLIN)

        events = dict(poller.poll(timeout))
        for sock in self._sockets:
            if sock in events:
                topic_bytes, data = sock.recv_multipart()
                topic = topic_bytes.decode("utf-8")
                self._backend.write(topic, data, time.time())
                return True
        return False

    def run(self) -> None:
        """Run the recording loop until interrupted."""
        self.start()
        try:
            while True:
                self.record_one()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Close storage and sockets."""
        self._backend.close()
        for sock in self._sockets:
            sock.close()
        self._context.term()


class LiveDashboard:
    """OpenCV-based live distance-matrix visualizer (legacy).

    .. deprecated::
        Replaced by ``navi_auditor.dashboard.GhostMatrixDashboard`` which
        uses PyQtGraph for GPU-accelerated rendering, multi-stream ZMQ
        ingestion, and real-time RL training curves.  This class is
        retained for fallback in environments without PyQt6.

    Renders at true 30-60 fps using ``cv2.imshow`` with deterministic
    ``waitKey`` polling — no matplotlib overhead.

    Supports **Tab-toggle teleop**: pressing Tab switches between
    OBSERVER mode (Actor drives) and MANUAL mode (WASD/arrow keys
    send ``StepRequest`` directly to the Environment REP socket).
    """

    def __init__(
        self,
        matrix_sub: str,
        step_endpoint: str | None = None,
        tick_hz: float = 30.0,
        linear_speed: float = 1.5,
        yaw_rate: float = 1.5,
    ) -> None:
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._matrix_sub = self._context.socket(zmq.SUB)
        self._matrix_sub.connect(matrix_sub)
        self._matrix_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))

        # REQ socket for direct step-mode teleop
        self._step_socket: zmq.Socket[bytes] | None = None
        if step_endpoint is not None:
            self._step_socket = self._context.socket(zmq.REQ)
            self._step_socket.connect(step_endpoint)

        self._tick_hz = max(1.0, tick_hz)
        self._linear_speed = linear_speed
        self._yaw_rate = yaw_rate

        self._latest_matrix: DistanceMatrix | None = None
        self._manual_mode: bool = False  # Tab-toggle state
        self._last_rx_time: float = 0.0
        self._pose_history: deque[tuple[float, float, float]] = deque(maxlen=90)
        self._forward_scan_history: deque[np.ndarray] = deque(maxlen=3)

        # key state
        self._forward = 0.0
        self._yaw = 0.0

        # overhead zoom state (+/- keys)
        self._overhead_zoom: float = 1.0

    # ── ZMQ helpers ──────────────────────────────────────────────────

    def _drain_messages(self) -> None:
        """Non-blocking drain of all queued distance matrix messages."""
        while True:
            try:
                _topic, data = self._matrix_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            msg = deserialize(data)
            if isinstance(msg, DistanceMatrix):
                self._latest_matrix = msg
                self._last_rx_time = time.time()
                pose = msg.robot_pose
                self._pose_history.append((pose.x, pose.z, pose.yaw))

    # ── rendering helpers ────────────────────────────────────────────

    @staticmethod
    def _depth_to_bgr(
        depth: np.ndarray,
        valid: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert a 2-D float32 depth plane to a coloured BGR image.

        Uses auto-contrast: the valid depth range is stretched to the
        full 0-255 colour range so nearby objects appear bright and
        distant objects dark, regardless of ``max_distance``.
        """
        if valid is not None and np.any(valid):
            valid_vals = depth[valid]
            lo = float(np.min(valid_vals))
            hi = float(np.max(valid_vals))
        else:
            lo, hi = 0.0, 1.0

        span = max(hi - lo, 1e-4)
        # Invert: close → high value (bright), far → low value (dark)
        normalised = 1.0 - np.clip((depth - lo) / span, 0.0, 1.0)
        grey = (normalised * 255.0).astype(np.uint8)
        coloured: np.ndarray = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)

        if valid is not None:
            invalid = ~valid
            coloured[invalid] = (30, 30, 30)

        return coloured

    @staticmethod
    def _delta_to_bgr(delta: np.ndarray) -> np.ndarray:
        """Convert delta-depth to a diverging colour map."""
        clamped = np.clip(delta, -0.5, 0.5)
        norm = ((clamped + 0.5) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_COOL)  # type: ignore[return-value]

    @staticmethod
    def _render_first_person(
        depth: np.ndarray,
        semantic: np.ndarray,
        valid: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, float]:
        """Render first-person view by projecting depth samples into screen space."""
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

        az = np.linspace(-np.deg2rad(_FWD_FOV_DEG / 2.0), np.deg2rad(_FWD_FOV_DEG / 2.0), az_bins)
        el = np.linspace(np.deg2rad(35.0), -np.deg2rad(35.0), el_bins)
        az_grid, el_grid = np.meshgrid(az, el, indexing="ij")

        dist_m = np.clip(depth, 0.0, 1.0) * _VIEW_RANGE_M
        valid_pts = valid & (dist_m > 1e-4)
        if not np.any(valid_pts):
            return img, _VIEW_RANGE_M

        x = dist_m * np.cos(el_grid) * np.cos(az_grid)
        y = dist_m * np.sin(el_grid)
        z = dist_m * np.cos(el_grid) * np.sin(az_grid)

        f = width / (2.0 * np.tan(np.deg2rad(_MAIN_FOV_DEG) / 2.0))
        px = (width * 0.5 + (z / np.maximum(x, 1e-4)) * f).astype(np.int32)
        py = (horizon_y - (y / np.maximum(x, 1e-4)) * f).astype(np.int32)

        in_view = valid_pts & (x > 0.1)
        in_view &= (px >= 0) & (px < width) & (py >= 0) & (py < height)
        if not np.any(in_view):
            return img, _VIEW_RANGE_M

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

            base = np.array(_FP_SEMANTIC_COLORS.get(int(sem[idx]), (150, 150, 150)), dtype=np.float32)
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
            center_m = float(np.min(center_depth[center_band]) * _VIEW_RANGE_M)
        else:
            center_m = _VIEW_RANGE_M

        return img, center_m

    @staticmethod
    def _distance_color(distance_m: float) -> tuple[int, int, int]:
        """Map distance in meters to unified dashboard BGR color."""
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

    def _render_forward_polar(self, depth: np.ndarray, valid: np.ndarray, w: int, h: int) -> np.ndarray:
        """Render forward 180° scan as a smoothed polar plot."""
        panel = np.full((h, w, 3), 18, dtype=np.uint8)
        az_bins = depth.shape[0]
        if az_bins == 0:
            return panel

        min_d = np.full((az_bins,), _VIEW_RANGE_M, dtype=np.float32)
        for i in range(az_bins):
            v = valid[i]
            if np.any(v):
                min_d[i] = float(np.min(depth[i][v]) * _VIEW_RANGE_M)

        min_d = np.convolve(min_d, np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0, mode="same")
        self._forward_scan_history.append(min_d.copy())
        hist = np.stack(tuple(self._forward_scan_history), axis=0)
        smoothed = np.mean(hist, axis=0)

        cx = w // 2
        cy = h - 12
        radius = int(min(w, h) * 0.45)

        for ring_m in (1.0, 2.0, 5.0, 10.0):
            rr = int(radius * min(1.0, ring_m / _VIEW_RANGE_M))
            cv2.ellipse(panel, (cx, cy), (rr, rr), 0, 180, 360, (90, 90, 90), 1)

        for deg in range(-90, 91, 15):
            theta = np.deg2rad(deg)
            x2 = int(cx + radius * np.sin(theta))
            y2 = int(cy - radius * np.cos(theta))
            cv2.line(panel, (cx, cy), (x2, y2), (60, 60, 60), 1)

        angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, az_bins)
        for i in range(1, az_bins):
            for idx in (i - 1, i):
                dist = float(smoothed[idx])
                rr = int(radius * min(1.0, dist / _VIEW_RANGE_M))
                th = angles[idx]
                x = int(cx + rr * np.sin(th))
                y = int(cy - rr * np.cos(th))
                cv2.line(panel, (cx, cy), (x, y), self._distance_color(dist), 2)

        cv2.circle(panel, (cx, cy), 5, (0, 255, 255), thickness=-1)
        cv2.arrowedLine(panel, (cx, cy), (cx, cy - 24), (0, 255, 255), 2, tipLength=0.4)
        return panel

    def _render_bev_occupancy(self, depth: np.ndarray, valid: np.ndarray, w: int, h: int) -> np.ndarray:
        """Render 360° observation as top-down occupancy map."""
        panel = np.full((h, w, 3), 20, dtype=np.uint8)
        az_bins = depth.shape[0]
        if az_bins == 0:
            return panel

        cx = w // 2
        cy = h // 2
        scale = (min(w, h) * 0.45) / _VIEW_RANGE_M
        angles = np.linspace(-np.pi, np.pi, az_bins, endpoint=False)

        for i in range(az_bins):
            v = valid[i]
            if not np.any(v):
                continue
            d = float(np.min(depth[i][v]) * _VIEW_RANGE_M)
            d_clamped = min(_VIEW_RANGE_M, d)
            x = int(cx + d_clamped * np.sin(angles[i]) * scale)
            y = int(cy - d_clamped * np.cos(angles[i]) * scale)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(panel, (x, y), 2, self._distance_color(d_clamped), thickness=-1)

        if len(self._pose_history) >= 2:
            cur_x, cur_z, _ = self._pose_history[-1]
            trail: list[tuple[int, int]] = []
            for hx, hz, _hyaw in self._pose_history:
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

    @staticmethod
    def _zoom_overhead(
        overhead: np.ndarray,
        zoom: float,
    ) -> np.ndarray:
        """Crop and scale the overhead minimap by the given zoom factor."""
        h, w = overhead.shape[:2]
        zoom = max(1.0, min(zoom, 6.0))
        crop_w = int(w / zoom)
        crop_h = int(h / zoom)
        x0 = (w - crop_w) // 2
        y0 = (h - crop_h) // 2
        cropped = overhead[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _add_orientation_guides(panel: np.ndarray, left_label: str, right_label: str) -> None:
        """Draw center guides and orientation labels on a side panel."""
        h, w = panel.shape[:2]
        cx = w // 2
        cy = h // 2
        cv2.line(panel, (cx, 0), (cx, h - 1), _GUIDE_COLOR, 1)
        cv2.line(panel, (0, cy), (w - 1, cy), _GUIDE_COLOR, 1)
        cv2.putText(panel, left_label, (8, h - 8), _HUD_FONT, 0.45, _TITLE_TEXT_COLOR, 1)
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

    @staticmethod
    def _compute_nav_metrics(depth: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
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

    @staticmethod
    def _draw_semantic_legend(frame: np.ndarray, x: int, y: int) -> None:
        """Draw compact semantic legend for human interpretation."""
        legend_items: list[tuple[str, tuple[int, int, int]]] = [
            ("Wall", _FP_SEMANTIC_COLORS[1]),
            ("Obstacle", _FP_SEMANTIC_COLORS[6]),
            ("Pillar", _FP_SEMANTIC_COLORS[4]),
            ("Road", _FP_SEMANTIC_COLORS[7]),
            ("Building", _FP_SEMANTIC_COLORS[8]),
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

    def _overlay_overhead_annotations(self, panel: np.ndarray, zoom: float) -> None:
        """Add center/range guides and trajectory trail to overhead panel."""
        h, w = panel.shape[:2]
        cx = w // 2
        cy = h // 2

        cv2.line(panel, (cx - 12, cy), (cx + 12, cy), _HEADING_COLOR, 1)
        cv2.line(panel, (cx, cy - 12), (cx, cy + 12), _HEADING_COLOR, 1)

        for r in (int(min(h, w) * 0.14), int(min(h, w) * 0.28), int(min(h, w) * 0.42)):
            cv2.circle(panel, (cx, cy), r, _GUIDE_COLOR, 1)

        if len(self._pose_history) >= 2:
            cur_x, cur_z, cur_yaw = self._pose_history[-1]
            world_radius = 18.0 / max(zoom, 1.0)
            scale = (min(w, h) * 0.5) / max(world_radius, 1e-3)
            trail_points: list[tuple[int, int]] = []
            for hx, hz, _ in self._pose_history:
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

            if len(trail_points) >= 1:
                cv2.circle(panel, trail_points[-1], 4, (255, 0, 255), thickness=-1)
                heading_len = 18
                hx = int(cx + heading_len)
                hy = cy
                cv2.arrowedLine(panel, (cx, cy), (hx, hy), _HEADING_COLOR, 2, tipLength=0.3)

    def _compose_frame(self) -> np.ndarray:
        """Build the full dashboard BGR frame from latest matrix data.

                Layout (1920x1080):
                    Left (2/3 width, full height):   First-person control view
                    Right-left column (1/6 x 1/3):   Interpreted forward/overhead/panorama
                    Right-right column (1/6 x 1/3):  Raw forward/overhead/panorama
        """
        frame = np.full((_DISPLAY_H, _DISPLAY_W, 3), 30, dtype=np.uint8)

        def draw_panel_title(text: str, x0: int, y0: int, width: int) -> None:
            cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + 24), _TITLE_BG_COLOR, thickness=-1)
            cv2.putText(
                frame,
                text,
                (x0 + 8, y0 + 17),
                _HUD_FONT,
                0.5,
                _TITLE_TEXT_COLOR,
                1,
            )

        if self._latest_matrix is not None:
            dm = self._latest_matrix
            depth_2d = dm.depth[0]        # (azimuth, elevation)
            semantic_2d = dm.semantic[0]  # (azimuth, elevation)
            valid_2d = dm.valid_mask[0]

            az_bins = depth_2d.shape[0]

            # Forward FOV slice (120 degrees)
            fov_fraction = 120.0 / 360.0
            fov_bins = max(1, int(az_bins * fov_fraction))
            centre_bin = az_bins // 2
            fov_lo = centre_bin - fov_bins // 2
            fov_hi = fov_lo + fov_bins

            fov_depth = depth_2d[fov_lo:fov_hi, :]
            fov_semantic = semantic_2d[fov_lo:fov_hi, :]
            fov_valid = valid_2d[fov_lo:fov_hi, :]

            # ── top-left: first-person 3D view ──
            fp_img, center_dist_m = self._render_first_person(
                fov_depth, fov_semantic, fov_valid, _MAIN_W, _DISPLAY_H,
            )
            frame[:, :_MAIN_W] = fp_img
            draw_panel_title("FIRST PERSON (PRIMARY CONTROL VIEW)", 0, 0, _MAIN_W)

            side_x0 = _MAIN_W
            side_x1 = _MAIN_W + _SIDE_COL_W

            # ── right-left top: interpreted forward polar scan ──
            depth_panel = self._render_forward_polar(fov_depth, fov_valid, _SIDE_COL_W, _STACK_H)
            self._add_orientation_guides(depth_panel, "LEFT", "RIGHT")
            frame[0:_STACK_H, side_x0:side_x1] = depth_panel
            draw_panel_title("FORWARD POLAR", side_x0, 0, _SIDE_COL_W)

            # ── right-left middle: interpreted overhead minimap (zoomable) ──
            overhead_bgr = dm.overhead.astype(np.uint8, copy=False)
            if self._overhead_zoom > 1.01:
                overhead_bgr = self._zoom_overhead(
                    overhead_bgr, self._overhead_zoom,
                )
            overhead_bgr = cv2.convertScaleAbs(overhead_bgr, alpha=1.1, beta=45)
            oh = cv2.resize(
                overhead_bgr, (_SIDE_COL_W, _STACK_H),
                interpolation=cv2.INTER_NEAREST,
            )
            self._overlay_overhead_annotations(oh, self._overhead_zoom)
            frame[_STACK_H : 2 * _STACK_H, side_x0:side_x1] = oh
            draw_panel_title(
                f"OVERHEAD+HUD {self._overhead_zoom:.1f}x",
                side_x0,
                _STACK_H,
                _SIDE_COL_W,
            )

            # ── right-left bottom: interpreted 360 bird's-eye occupancy ──
            pano = self._render_bev_occupancy(depth_2d, valid_2d, _SIDE_COL_W, _DISPLAY_H - 2 * _STACK_H)
            self._add_orientation_guides(pano, "LEFT", "RIGHT")
            frame[2 * _STACK_H : _DISPLAY_H, side_x0:side_x1] = pano
            draw_panel_title("BIRD'S EYE 360", side_x0, 2 * _STACK_H, _SIDE_COL_W)

            # ── right-right top: raw forward depth heatmap (previous view) ──
            raw_forward = self._depth_to_bgr(fov_depth, fov_valid)
            raw_forward = cv2.resize(raw_forward, (_SIDE_COL_W, _STACK_H), interpolation=cv2.INTER_NEAREST)
            self._add_orientation_guides(raw_forward, "LEFT", "RIGHT")
            frame[0:_STACK_H, side_x1:_DISPLAY_W] = raw_forward
            draw_panel_title("RAW FORWARD DEPTH", side_x1, 0, _SIDE_COL_W)

            # ── right-right middle: raw overhead map (previous view) ──
            overhead_raw = dm.overhead.astype(np.uint8, copy=False)
            if self._overhead_zoom > 1.01:
                overhead_raw = self._zoom_overhead(overhead_raw, self._overhead_zoom)
            overhead_raw = cv2.convertScaleAbs(overhead_raw, alpha=1.1, beta=45)
            overhead_raw = cv2.resize(overhead_raw, (_SIDE_COL_W, _STACK_H), interpolation=cv2.INTER_NEAREST)
            frame[_STACK_H : 2 * _STACK_H, side_x1:_DISPLAY_W] = overhead_raw
            draw_panel_title("RAW OVERHEAD", side_x1, _STACK_H, _SIDE_COL_W)

            # ── right-right bottom: raw panorama depth heatmap (previous view) ──
            raw_pano = self._depth_to_bgr(depth_2d, valid_2d)
            raw_pano = cv2.resize(
                raw_pano,
                (_SIDE_COL_W, _DISPLAY_H - 2 * _STACK_H),
                interpolation=cv2.INTER_NEAREST,
            )
            self._add_orientation_guides(raw_pano, "LEFT", "RIGHT")
            frame[2 * _STACK_H : _DISPLAY_H, side_x1:_DISPLAY_W] = raw_pano
            draw_panel_title("RAW PANORAMA 360", side_x1, 2 * _STACK_H, _SIDE_COL_W)

            # First-person navigation metrics for human observers.
            fwd, left, right = self._compute_nav_metrics(fov_depth, fov_valid)
            nav_txt = f"risk depth  F:{fwd:.2f}  L:{left:.2f}  R:{right:.2f}"
            cv2.putText(frame, nav_txt, (14, 86), _HUD_FONT, 0.52, (235, 235, 235), 1)
            if fwd < 0.18:
                cv2.rectangle(frame, (10, 96), (420, 126), _DANGER_COLOR, thickness=-1)
                cv2.putText(
                    frame,
                    "DANGER: obstacle close ahead",
                    (18, 116),
                    _HUD_FONT,
                    0.55,
                    (255, 255, 255),
                    1,
                )

            self._draw_semantic_legend(frame, 14, 136)

            center_txt = f"center range: {center_dist_m:.2f}m"
            cv2.putText(frame, center_txt, (14, 126), _HUD_FONT, 0.5, (230, 230, 230), 1)

            # ── HUD overlay ──
            p = dm.robot_pose
            step_txt = f"step={dm.step_id}"
            pose_txt = f"({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) yaw={p.yaw:.2f}"
            cv2.putText(
                frame, step_txt, (8, 20),
                _HUD_FONT, _HUD_SCALE, _HUD_COLOR, _HUD_THICK,
            )
            cv2.putText(
                frame, pose_txt, (8, 40),
                _HUD_FONT, _HUD_SCALE, _HUD_COLOR, _HUD_THICK,
            )
            stale_s = max(0.0, time.time() - self._last_rx_time)
            freshness_color = (0, 220, 0) if stale_s < 0.25 else (0, 180, 255) if stale_s < 1.0 else (0, 0, 255)
            cv2.putText(
                frame,
                f"stream_age={stale_s:.2f}s",
                (8, 60),
                _HUD_FONT,
                _HUD_SCALE,
                freshness_color,
                _HUD_THICK,
            )

            # Crosshair on first-person view
            cx_fp, cy_fp = _MAIN_W // 2, _DISPLAY_H // 2
            cv2.line(
                frame, (cx_fp - 10, cy_fp), (cx_fp + 10, cy_fp),
                (0, 255, 0), 1,
            )
            cv2.line(
                frame, (cx_fp, cy_fp - 10), (cx_fp, cy_fp + 10),
                (0, 255, 0), 1,
            )

            # Panel separator lines
            cv2.line(
                frame, (_MAIN_W, 0), (_MAIN_W, _DISPLAY_H), _PANEL_BORDER_COLOR, 2,
            )
            cv2.line(
                frame,
                (_MAIN_W + _SIDE_COL_W, 0),
                (_MAIN_W + _SIDE_COL_W, _DISPLAY_H),
                _PANEL_BORDER_COLOR,
                2,
            )
            cv2.line(
                frame, (_MAIN_W, _STACK_H), (_DISPLAY_W, _STACK_H), _PANEL_BORDER_COLOR, 2,
            )
            cv2.line(
                frame,
                (_MAIN_W, 2 * _STACK_H),
                (_DISPLAY_W, 2 * _STACK_H),
                _PANEL_BORDER_COLOR,
                2,
            )

        else:
            cv2.putText(
                frame, "waiting for distance_matrix_v2 stream...",
                (20, _DISPLAY_H // 2),
                _HUD_FONT, 0.7, _HUD_COLOR, 1,
            )

        # control status bar
        bar_y = _DISPLAY_H - 30
        if self._manual_mode:
            ctrl = (
                f"MANUAL  fwd={self._forward:+.1f}  yaw={self._yaw:+.1f}"
                "  [WASD]  Tab=toggle"
            )
            cv2.putText(
                frame, ctrl, (8, bar_y),
                _HUD_FONT, _HUD_SCALE, (0, 255, 100), _HUD_THICK,
            )
        else:
            mode_txt = "OBSERVER (AI driving) | Tab=manual | +/-=zoom"
            cv2.putText(
                frame, mode_txt, (8, bar_y),
                _HUD_FONT, _HUD_SCALE, (180, 180, 180), _HUD_THICK,
            )

        return frame

    # ── main loop ────────────────────────────────────────────────────

    def _process_key(self, key: int) -> bool:
        """Update velocity state from key code.  Returns False to quit."""
        self._forward = 0.0
        self._yaw = 0.0

        if key == 27:  # ESC
            return False
        if key == ord("q"):
            return False

        # Tab key toggles manual mode (requires step_endpoint)
        if key == 9:  # Tab
            if self._step_socket is not None:
                self._manual_mode = not self._manual_mode
            return True

        # Overhead zoom controls
        if key in (ord("+"), ord("="), 43):
            self._overhead_zoom = min(6.0, self._overhead_zoom + 0.5)
            return True
        if key in (ord("-"), 45):
            self._overhead_zoom = max(1.0, self._overhead_zoom - 0.5)
            return True

        if key in (ord("w"), 82):  # w or Up arrow
            self._forward = self._linear_speed
        elif key in (ord("s"), 84):  # s or Down arrow
            self._forward = -self._linear_speed
        if key in (ord("a"), 81):  # a or Left arrow
            self._yaw = self._yaw_rate
        elif key in (ord("d"), 83):  # d or Right arrow
            self._yaw = -self._yaw_rate

        return True

    def _send_step_request(self, step_id: int) -> None:
        """Send a StepRequest via REQ socket and receive the StepResult."""
        if self._step_socket is None:
            return

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array(
                [[self._forward, 0.0, 0.0]], dtype=np.float32,
            ),
            angular_velocity=np.array(
                [[0.0, 0.0, self._yaw]], dtype=np.float32,
            ),
            policy_id="dashboard-manual",
            step_id=step_id,
            timestamp=time.time(),
        )
        request = StepRequest(
            action=action,
            step_id=step_id,
            timestamp=time.time(),
        )
        self._step_socket.send(serialize(request))
        # Block until Environment replies (usually < 5 ms)
        _reply = self._step_socket.recv()

    def run(self) -> None:
        """Run the OpenCV visualization loop."""
        window_name = "Ghost-Matrix Dashboard"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, _DISPLAY_W, _DISPLAY_H)

        step_id = 0
        tick_ms = max(1, int(1000.0 / self._tick_hz))

        try:
            while True:
                self._drain_messages()

                frame = self._compose_frame()
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(tick_ms) & 0xFF
                if not self._process_key(key):
                    break

                has_input = abs(self._forward) > 0.01 or abs(self._yaw) > 0.01

                # Manual mode: send StepRequest directly via REQ socket
                if self._manual_mode and self._step_socket is not None and has_input:
                    self._send_step_request(step_id)
                    step_id += 1

        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
            self._matrix_sub.close()
            if self._step_socket is not None:
                self._step_socket.close()
            self._context.term()
