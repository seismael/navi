"""3D environment viewer using PyQtGraph OpenGL for Ghost-Matrix dashboard.

Renders the simulation mesh with semantic face colouring and a fixed
overhead camera showing the drone navigating *inside* the environment.
Walls are semi-transparent and ceilings hidden so the interior is
visible.  A coloured trajectory trail and heading arrow track the
drone's motion over time.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWidget

__all__: list[str] = ["SceneView3D"]

_log = logging.getLogger(__name__)

# ── Face colours (RGBA float 0-1) ────────────────────────────────────
_FLOOR_COLOR = (0.30, 0.45, 0.25, 0.95)  # earthy green, nearly opaque
_CEILING_COLOR = (0.30, 0.30, 0.35, 0.06)  # barely visible
_WALL_COLOR_BASE = np.array(
    [0.42, 0.45, 0.55, 0.18], dtype=np.float32
)  # semi-transparent blue-gray
_WALL_EDGE_COLOR = (100, 110, 130, 50)  # subtle wireframe edges

# Drone & trail
_DRONE_COLOR = (1.0, 0.55, 0.0, 1.0)  # bright orange
_HEADING_COLOR = (0.0, 1.0, 0.8, 1.0)  # cyan-green heading arrow
_TRAIL_MAX = 600  # max trail positions
_TRAIL_COLOR_RECENT = np.array([0.2, 0.9, 0.9, 0.9])  # bright cyan (recent)
_TRAIL_COLOR_OLD = np.array([0.15, 0.25, 0.35, 0.2])  # dim (old)


class SceneView3D:
    """3D mesh viewer showing the drone navigating inside the environment.

    Uses ``pyqtgraph.opengl.GLViewWidget`` to render a triangulated
    mesh loaded via ``trimesh``.  Ceilings are hidden and walls are
    semi-transparent so the drone is always visible inside the scene
    from a high fixed camera.

    Parameters
    ----------
    scene_path
        Path to a ``.glb``, ``.obj``, or other trimesh-loadable mesh.
    parent
        Optional parent Qt widget.
    """

    def __init__(self, scene_path: str, parent: QWidget | None = None) -> None:
        import pyqtgraph.opengl as gl
        import trimesh

        self._gl = gl
        self._widget = gl.GLViewWidget(parent=parent)
        self._widget.setBackgroundColor(13, 13, 26)

        # ── Load mesh ────────────────────────────────────────────────
        _log.info("SceneView3D: loading mesh from %s", scene_path)
        raw = trimesh.load(scene_path, force="mesh")

        if isinstance(raw, trimesh.Scene):
            geometries = list(raw.geometry.values())
            if len(geometries) == 0:
                _log.warning("SceneView3D: empty scene, using placeholder")
                raw = trimesh.creation.box(extents=(2, 2, 2))
            else:
                raw = trimesh.util.concatenate(geometries)

        assert isinstance(raw, trimesh.Trimesh)
        verts = np.asarray(raw.vertices, dtype=np.float32)
        faces = np.asarray(raw.faces, dtype=np.int32)

        # Swap Y↔Z for GL convention (simulation is Y-up, GL is Z-up)
        verts_gl = verts.copy()
        verts_gl[:, 1], verts_gl[:, 2] = verts[:, 2].copy(), verts[:, 1].copy()

        # ── Classify faces by normals ────────────────────────────────
        face_normals = np.asarray(raw.face_normals, dtype=np.float32)
        ny = face_normals[:, 1]  # Y-up normal (before swap)

        floor_mask = ny > 0.7
        ceil_mask = ny < -0.7
        wall_mask = ~floor_mask & ~ceil_mask

        n_faces = len(faces)

        # ── Height-based colouring for walls ─────────────────────────
        # Compute per-face centroid height (in GL Z-up coordinates)
        face_centroids_z = np.mean(
            verts_gl[faces, 2],
            axis=1,
        ).astype(np.float32)
        z_min = float(np.min(verts_gl[:, 2]))
        z_max = float(np.max(verts_gl[:, 2]))
        z_span = max(z_max - z_min, 0.01)
        height_t = np.clip((face_centroids_z - z_min) / z_span, 0.0, 1.0)

        face_colors = np.empty((n_faces, 4), dtype=np.float32)
        face_colors[floor_mask] = _FLOOR_COLOR

        # Walls: base colour with height-dependent brightness
        wall_brightness = 0.7 + 0.6 * height_t[wall_mask]
        wall_rgba = np.tile(_WALL_COLOR_BASE, (int(np.sum(wall_mask)), 1))
        wall_rgba[:, :3] *= wall_brightness[:, np.newaxis]
        wall_rgba[:, :3] = np.clip(wall_rgba[:, :3], 0.0, 1.0)
        face_colors[wall_mask] = wall_rgba

        # Ceiling: nearly invisible
        face_colors[ceil_mask] = _CEILING_COLOR

        # ── Floor mesh (opaque, rendered first) ──────────────────────
        floor_faces_idx = np.where(floor_mask)[0]
        if len(floor_faces_idx) > 0:
            floor_item = gl.GLMeshItem(
                vertexes=verts_gl,
                faces=faces[floor_faces_idx],
                faceColors=face_colors[floor_faces_idx],
                smooth=False,
                shader="shaded",
                drawEdges=False,
            )
            floor_item.setGLOptions("opaque")
            self._widget.addItem(floor_item)

        # ── Wall mesh (semi-transparent, with faint edges) ───────────
        wall_faces_idx = np.where(wall_mask)[0]
        if len(wall_faces_idx) > 0:
            wall_item = gl.GLMeshItem(
                vertexes=verts_gl,
                faces=faces[wall_faces_idx],
                faceColors=face_colors[wall_faces_idx],
                smooth=False,
                shader="shaded",
                drawEdges=True,
                edgeColor=_WALL_EDGE_COLOR,
            )
            wall_item.setGLOptions("translucent")
            self._widget.addItem(wall_item)

        # Ceiling: skip entirely — cleaner than nearly-invisible faces

        # ── Grid for reference ───────────────────────────────────────
        grid = gl.GLGridItem()
        grid.setSize(60, 60)
        grid.setSpacing(2, 2)
        grid.setColor((40, 40, 50, 60))
        self._widget.addItem(grid)

        # ── Drone body marker ────────────────────────────────────────
        drone_pos = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        self._drone_scatter = gl.GLScatterPlotItem(
            pos=drone_pos,
            color=_DRONE_COLOR,
            size=22,
            pxMode=True,
        )
        self._widget.addItem(self._drone_scatter)

        # ── Heading arrow (short line from drone position) ───────────
        heading_pts = np.array(
            [[0.0, 0.0, 1.0], [0.8, 0.0, 1.0]],
            dtype=np.float32,
        )
        self._heading_line = gl.GLLinePlotItem(
            pos=heading_pts,
            color=_HEADING_COLOR,
            width=3.0,
            antialias=True,
        )
        self._widget.addItem(self._heading_line)

        # ── Trajectory trail ─────────────────────────────────────────
        self._trail: deque[tuple[float, float, float]] = deque(maxlen=_TRAIL_MAX)
        self._trail_line: gl.GLLinePlotItem | None = None

        # ── Camera: high fixed view centred on environment ───────────
        bbox_center = np.mean(verts_gl, axis=0)
        bbox_extent = np.ptp(verts_gl, axis=0)  # (dx, dy, dz)
        scene_radius = float(np.linalg.norm(bbox_extent)) * 0.5
        cam_dist = max(scene_radius * 1.8, 15.0)

        self._widget.setCameraPosition(
            distance=cam_dist,
            elevation=65,
            azimuth=-90,
        )
        self._widget.opts["center"].setX(float(bbox_center[0]))
        self._widget.opts["center"].setY(float(bbox_center[1]))
        self._widget.opts["center"].setZ(float(bbox_center[2]))

        self._scene_center = bbox_center.copy()
        self._scene_radius = scene_radius

        _log.info(
            "SceneView3D: loaded %d verts, %d faces (floor=%d, wall=%d, ceil=%d)  cam_dist=%.1f",
            len(verts_gl),
            n_faces,
            int(np.sum(floor_mask)),
            int(np.sum(wall_mask)),
            int(np.sum(ceil_mask)),
            cam_dist,
        )

    @property
    def widget(self) -> object:
        """Return the underlying GLViewWidget for embedding in Qt layouts."""
        return self._widget

    def update_pose(self, x: float, y: float, z: float, yaw: float) -> None:
        """Move the drone marker and update trail / heading arrow.

        The camera stays fixed so the user sees the drone moving
        *within* the environment.  Users can still orbit/zoom manually
        with the mouse.

        Parameters
        ----------
        x, y, z
            Robot position in simulation coordinates (Y-up).
        yaw
            Robot heading in radians.
        """
        import pyqtgraph.opengl as gl

        # Swap Y↔Z for GL (Z-up)
        gl_x = float(x)
        gl_y = float(z)
        gl_z = float(y)

        # ── Drone marker ─────────────────────────────────────────────
        self._drone_scatter.setData(
            pos=np.array([[gl_x, gl_y, gl_z]], dtype=np.float32),
        )

        # ── Heading arrow ────────────────────────────────────────────
        arrow_len = 1.2
        # In sim: yaw rotates around Y-up → in GL: rotates around Z-up
        dx = arrow_len * np.cos(yaw)  # forward in sim-X → GL-X
        dy = arrow_len * np.sin(yaw)  # forward in sim-Z → GL-Y
        head_pts = np.array(
            [[gl_x, gl_y, gl_z], [gl_x + dx, gl_y + dy, gl_z]],
            dtype=np.float32,
        )
        self._heading_line.setData(pos=head_pts)

        # ── Trajectory trail ─────────────────────────────────────────
        self._trail.append((gl_x, gl_y, gl_z))

        trail_len = len(self._trail)
        if trail_len >= 2:
            trail_arr = np.array(list(self._trail), dtype=np.float32)
            # Gradient colours: old → dim, recent → bright
            t = np.linspace(0.0, 1.0, trail_len, dtype=np.float32)
            trail_colors = (
                _TRAIL_COLOR_OLD[np.newaxis, :] * (1.0 - t[:, np.newaxis])
                + _TRAIL_COLOR_RECENT[np.newaxis, :] * t[:, np.newaxis]
            )
            trail_colors = trail_colors.astype(np.float32)

            if self._trail_line is not None:
                self._trail_line.setData(pos=trail_arr, color=trail_colors)
            else:
                self._trail_line = gl.GLLinePlotItem(
                    pos=trail_arr,
                    color=trail_colors,
                    width=2.0,
                    antialias=True,
                )
                self._widget.addItem(self._trail_line)
