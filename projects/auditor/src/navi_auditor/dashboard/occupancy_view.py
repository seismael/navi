"""Live 2D occupancy map for Ghost-Matrix dashboard.

Accumulates depth-ray observations into a world-space occupancy grid,
producing a SLAM-style top-down view showing obstacles (coloured by
depth), explored free-space, and unexplored regions.  The drone
position, heading, and trajectory trail are overlaid.

When a ``scene_path`` is provided the mesh geometry is projected onto
the XZ ground-plane and rendered as a dim blueprint underlay beneath
the SLAM overlay.

Works with any environment — no mesh file required.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

__all__: list[str] = ["OccupancyMap"]

_log = logging.getLogger(__name__)

# ── Grid defaults ────────────────────────────────────────────────────
_CELL_M: float = 0.15  # metres per cell
_EXTENT_M: float = 60.0  # total grid extent (±30 m from origin)
_VIEW_RADIUS_M: float = 18.0  # visible radius around drone
_MAX_TRAIL: int = 2000  # retained trail positions

# ── Palette (BGR) ────────────────────────────────────────────────────
_COL_UNEXPLORED = np.array([13, 13, 18], dtype=np.uint8)
_COL_FREE = np.array([30, 32, 35], dtype=np.uint8)
_DRONE_BGR = (0, 140, 255)  # bright orange
_HEADING_BGR = (200, 255, 0)  # cyan-green
_TRAIL_NEW = np.array([220, 220, 40], dtype=np.float32)  # bright cyan
_TRAIL_OLD = np.array([40, 45, 25], dtype=np.float32)  # dim

# ── Blueprint floor-plan palette (BGR) ───────────────────────────
_COL_WALL_FILL = np.array([58, 48, 45], dtype=np.uint8)  # dim blue-gray
_COL_WALL_EDGE: tuple[int, int, int] = (85, 70, 65)  # brighter outline
_COL_FLOOR_FILL = np.array([28, 22, 20], dtype=np.uint8)  # faint tint

_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


class OccupancyMap:
    """Accumulating world-space 2D occupancy grid with rendering.

    Call :meth:`update` every tick with depth data and pose, then
    :meth:`render` to produce a BGR image for ``ImagePanel.set_image``.
    """

    def __init__(
        self,
        cell_size: float = _CELL_M,
        grid_extent: float = _EXTENT_M,
        max_distance: float = 15.0,
        scene_path: str | None = None,
    ) -> None:
        self._cell = cell_size
        self._n = int(grid_extent / cell_size)
        self._half = grid_extent / 2.0
        self._max_dist = max_distance

        # 0 = unexplored, 1 = free, 2 = occupied
        self._occ = np.zeros((self._n, self._n), dtype=np.uint8)
        self._depth_m = np.full(
            (self._n, self._n), np.nan, dtype=np.float32,
        )

        self._trail: deque[tuple[float, float, float]] = deque(
            maxlen=_MAX_TRAIL,
        )
        self._last_episode: int = -1

        # Grid origin (bottom-left corner in world coords)
        self._ox: float = -self._half
        self._oz: float = -self._half

        # ── Blueprint floor-plan layers (static, survive reset) ──────
        self._wall_grid = np.zeros((self._n, self._n), dtype=np.bool_)
        self._floor_grid = np.zeros((self._n, self._n), dtype=np.bool_)
        self._scene_name: str = ""

        if scene_path is not None:
            self.load_scene(scene_path)

    # ── coordinate helpers ───────────────────────────────────────────

    def _w2g(
        self,
        wx: np.ndarray,
        wz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World (X, Z) → grid (col, row)."""
        gx = ((wx - self._ox) / self._cell).astype(np.int32)
        gz = ((wz - self._oz) / self._cell).astype(np.int32)
        return gx, gz

    def _in_bounds(
        self,
        gx: np.ndarray,
        gz: np.ndarray,
    ) -> np.ndarray:
        """Return boolean mask for grid indices that are within bounds."""
        return (gx >= 0) & (gx < self._n) & (gz >= 0) & (gz < self._n)

    # ── scene geometry loading ────────────────────────────────────────

    def load_scene(self, scene_path: str) -> None:
        """Load a mesh scene and rasterise wall/floor faces onto the grid.

        Uses the same face-normal classification as *MeshSceneBackend*:
        ``ny > 0.7`` → floor, ``ny < -0.7`` → ceiling, else → wall.

        Parameters
        ----------
        scene_path
            Path to a ``.glb`` / ``.obj`` / ``.stl`` mesh file.
        """
        try:
            import trimesh
        except ImportError:
            _log.warning(
                "trimesh not installed — floor-plan underlay disabled",
            )
            return

        try:
            raw = trimesh.load(scene_path, force="mesh")
            if isinstance(raw, trimesh.Scene):
                raw = trimesh.util.concatenate(
                    list(raw.geometry.values()),
                )
            mesh: trimesh.Trimesh = raw  # type: ignore[assignment]
        except Exception:
            _log.warning(
                "Failed to load scene mesh: %s", scene_path, exc_info=True,
            )
            return

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        ny = normals[:, 1]  # Y-up coordinate system

        # Warn if mesh exceeds grid extent
        xmin, xmax = float(verts[:, 0].min()), float(verts[:, 0].max())
        zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
        if (
            xmin < -self._half
            or xmax > self._half
            or zmin < -self._half
            or zmax > self._half
        ):
            _log.warning(
                "Scene mesh bounds (%.1f-%.1f, %.1f-%.1f) exceed grid "
                "extent ±%.0fm — geometry will be clipped",
                xmin, xmax, zmin, zmax, self._half,
            )

        # Classify faces
        is_floor = ny > 0.7
        is_wall = ~is_floor & (ny >= -0.7)  # exclude ceilings

        self._wall_grid[:] = False
        self._floor_grid[:] = False

        self._rasterise_faces(verts, faces[is_floor], self._floor_grid)
        self._rasterise_faces(verts, faces[is_wall], self._wall_grid)

        self._scene_name = Path(scene_path).stem
        _log.info(
            "Floor-plan loaded: %s (%d wall faces, %d floor faces)",
            self._scene_name,
            int(is_wall.sum()),
            int(is_floor.sum()),
        )

    def _rasterise_faces(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Project triangle faces to XZ and fill them on *target* grid."""
        if faces.size == 0:
            return

        for tri_idx in range(faces.shape[0]):
            tri = verts[faces[tri_idx]]  # (3, 3)
            # Project to XZ (cols 0, 2) → grid coords
            wx = tri[:, 0]
            wz = tri[:, 2]
            gx = ((wx - self._ox) / self._cell).astype(np.int32)
            gz = ((wz - self._oz) / self._cell).astype(np.int32)
            pts = np.column_stack((gx, gz)).reshape(1, -1, 2)
            cv2.fillPoly(target.view(np.uint8), pts, 1)  # type: ignore[arg-type]

    # ── per-tick grid update ─────────────────────────────────────────

    def update(
        self,
        depth_2d: np.ndarray,
        valid_2d: np.ndarray,
        x: float,
        z: float,
        yaw: float,
        episode_id: int = 0,
    ) -> None:
        """Project depth rays into world space and update occupancy.

        Parameters
        ----------
        depth_2d
            ``(Az, El)`` normalised depth ``[0, 1]``.
        valid_2d
            ``(Az, El)`` boolean mask.
        x, z
            Drone world position (Y-up sim coords, using XZ plane).
        yaw
            Heading in radians.
        episode_id
            Current episode — grid resets when this changes.
        """
        if episode_id != self._last_episode and self._last_episode >= 0:
            self.reset()
        self._last_episode = episode_id

        az_bins = depth_2d.shape[0]
        if az_bins == 0:
            return

        self._trail.append((x, z, yaw))

        # ── per-azimuth minimum valid depth ──────────────────────────
        local_angles = np.linspace(
            -np.pi, np.pi, az_bins, endpoint=False, dtype=np.float32,
        )
        world_angles = local_angles + yaw

        min_d_m = np.full(az_bins, np.nan, dtype=np.float32)
        for i in range(az_bins):
            v = valid_2d[i]
            if np.any(v):
                min_d_m[i] = float(np.min(depth_2d[i][v])) * self._max_dist

        has_hit = ~np.isnan(min_d_m)
        if not np.any(has_hit):
            return

        # ── obstacle endpoints ───────────────────────────────────────
        d = min_d_m[has_hit]
        a = world_angles[has_hit]
        hit_x = x + d * np.cos(a)
        hit_z = z + d * np.sin(a)

        ogx, ogz = self._w2g(hit_x, hit_z)
        mask_ok = self._in_bounds(ogx, ogz)
        gx_ok = ogx[mask_ok]
        gz_ok = ogz[mask_ok]
        self._occ[gz_ok, gx_ok] = 2
        self._depth_m[gz_ok, gx_ok] = d[mask_ok]

        # ── free-space ray marching (subsampled) ─────────────────────
        step = self._cell * 2.0
        subsample = max(1, az_bins // 64)  # ≤64 rays for free-space
        for idx in range(0, len(d), subsample):
            if not mask_ok[idx]:
                continue
            ray_d = d[idx]
            n_steps = max(1, int(ray_d / step))
            ts = np.linspace(0.0, 0.92, n_steps, dtype=np.float32)
            rx = x + ray_d * np.cos(a[idx]) * ts
            rz = z + ray_d * np.sin(a[idx]) * ts
            rgx, rgz = self._w2g(rx, rz)
            rm = self._in_bounds(rgx, rgz)
            rgx_m = rgx[rm]
            rgz_m = rgz[rm]
            free_cells = self._occ[rgz_m, rgx_m] == 0
            self._occ[rgz_m[free_cells], rgx_m[free_cells]] = 1

    # ── rendering ────────────────────────────────────────────────────

    def render(self, width: int, height: int) -> np.ndarray:
        """Produce a BGR image of the occupancy map centred on the drone.

        Parameters
        ----------
        width, height
            Output image size in pixels.

        Returns
        -------
        BGR ``uint8`` array ``(height, width, 3)``.
        """
        if not self._trail:
            return np.full((height, width, 3), 13, dtype=np.uint8)

        cx, cz, _cyaw = self._trail[-1]

        # ── extract view window around drone ─────────────────────────
        vr = _VIEW_RADIUS_M
        gx_lo = max(0, int((cx - vr - self._ox) / self._cell))
        gx_hi = min(self._n, int((cx + vr - self._ox) / self._cell) + 1)
        gz_lo = max(0, int((cz - vr - self._oz) / self._cell))
        gz_hi = min(self._n, int((cz + vr - self._oz) / self._cell) + 1)

        v_occ = self._occ[gz_lo:gz_hi, gx_lo:gx_hi]
        v_dep = self._depth_m[gz_lo:gz_hi, gx_lo:gx_hi]
        v_wall = self._wall_grid[gz_lo:gz_hi, gx_lo:gx_hi]
        v_floor = self._floor_grid[gz_lo:gz_hi, gx_lo:gx_hi]

        if v_occ.size == 0:
            return np.full((height, width, 3), 13, dtype=np.uint8)

        vh, vw = v_occ.shape

        # ── colour grid cells ────────────────────────────────────────
        img = np.tile(_COL_UNEXPLORED, (vh, vw, 1)).copy()

        # Blueprint underlay — visible in unexplored cells
        unexplored = v_occ == 0
        img[unexplored & v_floor] = _COL_FLOOR_FILL
        img[unexplored & v_wall] = _COL_WALL_FILL  # walls over floor

        img[v_occ == 1] = _COL_FREE

        occ_mask = v_occ == 2
        if np.any(occ_mask):
            d_norm = np.clip(v_dep[occ_mask] / self._max_dist, 0.0, 1.0)
            gray = (d_norm * 255.0).astype(np.uint8)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
            img[occ_mask] = colored.reshape(-1, 3)

        # ── resize to output ─────────────────────────────────────────
        out = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)

        # ── blueprint wall-edge overlay (visible even over SLAM) ─────
        has_blueprint = np.any(v_wall)
        if has_blueprint:
            wall_resized = cv2.resize(
                v_wall.view(np.uint8) * 255,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            edges = cv2.Canny(wall_resized, 50, 150)
            out[edges > 0] = _COL_WALL_EDGE

        # ── pixel-space conversion helper ────────────────────────────
        sx = width / max(vw, 1)
        sz = height / max(vh, 1)

        def _w2px(wx: float, wz: float) -> tuple[int, int]:
            px = int(((wx - self._ox) / self._cell - gx_lo) * sx)
            py = int(((wz - self._oz) / self._cell - gz_lo) * sz)
            return px, py

        # ── trajectory trail ─────────────────────────────────────────
        trail_list = list(self._trail)
        n_trail = len(trail_list)
        if n_trail >= 2:
            for i in range(1, n_trail):
                t = i / max(n_trail - 1, 1)
                c = (_TRAIL_OLD * (1.0 - t) + _TRAIL_NEW * t).astype(int)
                bgr_t = (int(c[0]), int(c[1]), int(c[2]))
                p1 = _w2px(trail_list[i - 1][0], trail_list[i - 1][1])
                p2 = _w2px(trail_list[i][0], trail_list[i][1])
                cv2.line(out, p1, p2, bgr_t, 1, cv2.LINE_AA)

        # ── drone marker ─────────────────────────────────────────────
        dpx, dpy = _w2px(cx, cz)
        cv2.circle(out, (dpx, dpy), 8, _DRONE_BGR, -1, cv2.LINE_AA)
        cv2.circle(out, (dpx, dpy), 9, (0, 80, 160), 1, cv2.LINE_AA)

        # ── heading arrow ────────────────────────────────────────────
        cyaw = self._trail[-1][2]
        arrow_len = 22
        hx = int(dpx + arrow_len * np.cos(cyaw))
        hy = int(dpy + arrow_len * np.sin(cyaw))
        cv2.arrowedLine(
            out, (dpx, dpy), (hx, hy),
            _HEADING_BGR, 2, cv2.LINE_AA, tipLength=0.35,
        )

        # ── range rings ──────────────────────────────────────────────
        ring_color = (50, 52, 55)
        label_color = (100, 105, 110)
        for r_m in (2.0, 5.0, 10.0, 15.0):
            r_px = int(r_m / self._cell * sx)
            if r_px < 8 or r_px > width:
                continue
            cv2.circle(out, (dpx, dpy), r_px, ring_color, 1, cv2.LINE_AA)
            lx = min(dpx + r_px + 3, width - 25)
            cv2.putText(
                out, f"{int(r_m)}m", (lx, dpy - 3),
                _HUD_FONT, 0.30, label_color, 1, cv2.LINE_AA,
            )

        # ── HUD title ────────────────────────────────────────────────
        cv2.putText(
            out, "LIVE MAP", (8, 18),
            _HUD_FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA,
        )
        if self._scene_name:
            cv2.putText(
                out, self._scene_name, (8, 34),
                _HUD_FONT, 0.32, (100, 110, 130), 1, cv2.LINE_AA,
            )

        # Explored-area fraction
        total = max(1, int(np.sum(self._occ > 0)))
        cv2.putText(
            out, f"cells: {total}", (8, height - 10),
            _HUD_FONT, 0.32, (120, 120, 120), 1, cv2.LINE_AA,
        )

        return out

    # ── lifecycle ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all accumulated data (called on episode boundary)."""
        self._occ[:] = 0
        self._depth_m[:] = np.nan
        self._trail.clear()
