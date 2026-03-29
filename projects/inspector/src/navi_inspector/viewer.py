"""PyVista interactive 3D viewer for .gmdag files.

Two camera modes (toggle with V):

  FIRST-PERSON (default) — walk inside the scene like the actor.
    Left-drag             look around (yaw / pitch)
    Scroll / W / S        move forward / backward
    A / D                 strafe left / right
    Q / E                 move up / down
    + / -                 double / halve speed

  ORBIT — inspect the scene from outside.
    Left-drag             orbit (rotate)
    Middle-drag           pan
    Right-drag            dolly zoom
    Scroll                move forward / backward (constant pace)

Display toggles:
    V   toggle FIRST-PERSON ↔ ORBIT camera mode
    M   toggle SURFACE ↔ VOXEL rendering mode
    F   wireframe on/off (surface mode)
    H   SDF heatmap colours on/off
    B   bounding-box wireframe
    I   info HUD overlay
    C   clipping plane
    R   refine to full DAG resolution
    X   export mesh to PLY
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser

from navi_inspector.cache import MeshCache
from navi_inspector.config import InspectorConfig
from navi_inspector.dag_extractor import SdfGrid, extract_sdf_grid
from navi_inspector.gmdag_io import GmdagAsset
from navi_inspector.mesh_builder import build_mesh, export_mesh

__all__: list[str] = ["launch_viewer"]

# Show voxels within this many cell-widths of geometry (diagnostic mode)
_SURFACE_THRESHOLD_FACTOR = 2.0


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _ViewerState:
    """Mutable viewer state."""

    def __init__(
        self,
        asset: GmdagAsset,
        initial_resolution: int,
        level: float | None,
        cache: MeshCache,
    ) -> None:
        self.asset = asset
        self.level = level
        self.cache = cache
        self.current_resolution = initial_resolution

        # Display toggles
        self.wireframe = False
        self.show_heatmap = True
        self.show_bbox = True
        self.show_info = True
        self.clip_active = False
        self.move_speed: float = 0.0

        # Camera mode: "fps" (default) or "orbit"
        self.cam_mode = "fps"

        # "surface" (default) or "voxels" (diagnostic)
        self.vis_mode = "surface"

        # Data
        self.voxel_grid: Any | None = None  # thresholded UnstructuredGrid
        self.mesh: pv.PolyData | None = None  # marching-cubes mesh

        # Actors
        self.display_actor: Any = None
        self.bbox_actor: Any = None
        self.info_actor: Any = None
        self.plotter: pv.Plotter | None = None

    def extract_data(self, resolution: int) -> None:
        """Extract SDF grid -> voxel grid + mesh at *resolution*."""
        sdf = extract_sdf_grid(self.asset, target_resolution=resolution)
        self.current_resolution = resolution

        # Voxel blocks (diagnostic mode)
        self.voxel_grid = _build_voxel_grid(sdf)

        # Marching-cubes mesh (cache-aware)
        cached = self.cache.get(self.asset.path, resolution, self.level)
        if cached is not None:
            self.mesh = cached
        else:
            try:
                mesh = build_mesh(sdf, level=self.level)
                self.cache.put(self.asset.path, resolution, self.level, mesh)
                self.mesh = mesh
            except ValueError:
                self.mesh = None  # no isosurface — voxel grid still works


# ---------------------------------------------------------------------------
# Voxel-grid builder
# ---------------------------------------------------------------------------

def _build_voxel_grid(sdf: SdfGrid) -> Any | None:
    """Threshold the SDF into solid near-surface voxel blocks (diagnostic)."""
    nx, ny, nz = sdf.grid.shape

    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.spacing = (sdf.voxel_size, sdf.voxel_size, sdf.voxel_size)
    grid.origin = tuple(sdf.bbox_min)

    vals = sdf.grid.copy()
    vals[~np.isfinite(vals)] = 1e6  # push void cells past threshold
    grid.cell_data["sdf_distance"] = vals.ravel(order="F").astype(np.float32)

    threshold = sdf.voxel_size * _SURFACE_THRESHOLD_FACTOR
    threshed = grid.threshold(value=(0, threshold), scalars="sdf_distance")
    if threshed is None or threshed.n_cells == 0:
        return None
    return threshed


# ---------------------------------------------------------------------------
# HUD / bounding box
# ---------------------------------------------------------------------------

def _build_info_text(state: _ViewerState) -> str:
    a = state.asset
    mode = state.vis_mode.upper()
    if state.vis_mode == "voxels" and state.voxel_grid is not None:
        geom = f"Blocks: {state.voxel_grid.n_cells:,}"
    elif state.mesh is not None:
        geom = f"Verts: {state.mesh.n_points:,}  Faces: {state.mesh.n_faces_strict:,}"
    else:
        geom = "(no data)"
    cam = "FPS" if state.cam_mode == "fps" else "ORBIT"
    if state.cam_mode == "fps":
        nav = f"LOOK: L-drag  MOVE: Scroll/WASD  Speed: {state.move_speed:.2f}m [+/-]"
    else:
        nav = f"ORBIT: L-drag  PAN: Mid-drag  Speed: {state.move_speed:.2f}m [+/-]"
    return (
        f"File: {a.path.name}\n"
        f"DAG Res: {a.resolution}  View Res: {state.current_resolution}\n"
        f"Voxel: {a.voxel_size:.4f} m  Nodes: {len(a.nodes):,}\n"
        f"Mode: {mode}  Camera: {cam}  {geom}\n"
        f"\n"
        f"{nav}\n"
        f"[V] Camera  [M] Voxels  [F] Wire  [H] Heat  [B] Box\n"
        f"[C] Clip  [R] Refine  [I] Info  [X] Export"
    )


def _add_bbox(plotter: pv.Plotter, asset: GmdagAsset) -> Any:
    """Add translucent bounding-box wireframe."""
    bmin = np.array(asset.bbox_min)
    bmax = np.array(asset.bbox_max)
    box = pv.Box(bounds=(bmin[0], bmax[0], bmin[1], bmax[1], bmin[2], bmax[2]))
    return plotter.add_mesh(
        box,
        style="wireframe",
        color="white",
        line_width=1.5,
        opacity=0.4,
        name="bbox",
    )


# ---------------------------------------------------------------------------
# Display refresh
# ---------------------------------------------------------------------------

def _refresh_display(state: _ViewerState) -> None:
    """Re-render with current mode (voxels or surface) and settings."""
    pl = state.plotter
    if pl is None:
        return

    if state.display_actor is not None:
        pl.remove_actor(state.display_actor)
        state.display_actor = None

    if state.vis_mode == "surface":
        mesh = state.mesh
        if mesh is None:
            return
        style = "wireframe" if state.wireframe else "surface"
        kwargs: dict[str, Any] = {
            "style": style,
            "name": "gmdag_display",
            "show_edges": state.wireframe,
            "opacity": 1.0,
        }
        if state.show_heatmap and "sdf_distance" in mesh.point_data:
            kwargs["scalars"] = "sdf_distance"
            kwargs["cmap"] = "turbo"
            kwargs["show_scalar_bar"] = True
            kwargs["scalar_bar_args"] = {"title": "SDF Distance (m)"}
        else:
            kwargs["color"] = "lightblue"
        state.display_actor = pl.add_mesh(mesh, **kwargs)

    else:  # voxels (diagnostic)
        grid = state.voxel_grid
        if grid is None or grid.n_cells == 0:
            return
        kwargs = {
            "name": "gmdag_display",
            "show_edges": False,
        }
        if state.show_heatmap and "sdf_distance" in grid.cell_data:
            kwargs["scalars"] = "sdf_distance"
            kwargs["cmap"] = "turbo"
            kwargs["show_scalar_bar"] = True
            kwargs["scalar_bar_args"] = {"title": "SDF Distance (m)"}
        else:
            kwargs["color"] = "sandybrown"
        state.display_actor = pl.add_mesh(grid, **kwargs)


def _refresh_info(state: _ViewerState) -> None:
    """Update the HUD text overlay."""
    pl = state.plotter
    if pl is None:
        return
    if state.info_actor is not None:
        pl.remove_actor(state.info_actor)
        state.info_actor = None

    if state.show_info:
        state.info_actor = pl.add_text(
            _build_info_text(state),
            position="upper_left",
            font_size=9,
            color="white",
            name="info_hud",
        )


# ---------------------------------------------------------------------------
# Toggle callbacks
# ---------------------------------------------------------------------------

def _toggle_wireframe(state: _ViewerState) -> None:
    state.wireframe = not state.wireframe
    _refresh_display(state)
    _refresh_info(state)


def _toggle_heatmap(state: _ViewerState) -> None:
    state.show_heatmap = not state.show_heatmap
    _refresh_display(state)


def _toggle_bbox(state: _ViewerState) -> None:
    pl = state.plotter
    if pl is None:
        return
    state.show_bbox = not state.show_bbox
    if state.bbox_actor is not None:
        pl.remove_actor(state.bbox_actor)
        state.bbox_actor = None
    if state.show_bbox:
        state.bbox_actor = _add_bbox(pl, state.asset)


def _toggle_info(state: _ViewerState) -> None:
    state.show_info = not state.show_info
    _refresh_info(state)


def _toggle_clip(state: _ViewerState) -> None:
    pl = state.plotter
    if pl is None:
        return
    state.clip_active = not state.clip_active
    if state.clip_active:
        pl.add_clip_plane_widget(
            callback=lambda normal, origin: None,  # noqa: ARG005
            normal="z",
        )
    else:
        pl.clear_plane_widgets()


def _toggle_vis_mode(state: _ViewerState) -> None:
    """Switch between voxel-block and surface-mesh rendering."""
    if state.vis_mode == "voxels":
        if state.mesh is None:
            return  # no mesh — stay in voxels
        state.vis_mode = "surface"
    else:
        state.vis_mode = "voxels"
    _refresh_display(state)
    _refresh_info(state)


def _refine(state: _ViewerState) -> None:
    """Refine extraction to the full DAG resolution."""
    pl = state.plotter
    if pl is None:
        return
    target = state.asset.resolution
    if state.current_resolution >= target:
        return

    pl.add_text(
        f"Extracting at {target}\u00b3 \u2014 please wait\u2026",
        position="lower_left",
        font_size=12,
        color="yellow",
        name="refine_status",
    )
    pl.render()

    state.extract_data(target)
    pl.remove_actor("refine_status")
    _refresh_display(state)
    _refresh_info(state)


def _export_current(state: _ViewerState) -> None:
    if state.mesh is None:
        return
    stem = state.asset.path.stem
    out_dir = Path("artifacts/inspector/exports")
    out_path = out_dir / f"{stem}_{state.current_resolution}.ply"
    saved = export_mesh(state.mesh, out_path, fmt="ply")

    pl = state.plotter
    if pl is not None:
        pl.add_text(
            f"Exported: {saved.name}",
            position="lower_left",
            font_size=10,
            color="lime",
            name="export_status",
        )


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------

# Degrees of rotation per pixel of mouse drag (first-person mode).
_MOUSE_SENSITIVITY = 0.25
# Fixed distance from camera position to focal point (arbitrary, just
# needs to stay constant so Yaw/Pitch rotate around the camera).
_FOCAL_DISTANCE = 1.0


def _sync_focal_point(state: _ViewerState) -> None:
    """Keep the focal point exactly *_FOCAL_DISTANCE* ahead of the camera.

    VTK's Yaw/Pitch rotate around the camera position, which can drift
    the focal distance.  This snaps it back so movement math stays
    consistent.
    """
    pl = state.plotter
    if pl is None:
        return
    cam = pl.camera
    pos = np.array(cam.position)
    foc = np.array(cam.focal_point)
    fwd = foc - pos
    length = float(np.linalg.norm(fwd))
    if length < 1e-12:
        fwd = np.array([0.0, 0.0, -1.0])
    else:
        fwd /= length
    cam.focal_point = tuple(pos + fwd * _FOCAL_DISTANCE)


def _move_camera(state: _ViewerState, direction: np.ndarray) -> None:
    """Translate camera + focal point by *direction* vector."""
    pl = state.plotter
    if pl is None:
        return
    cam = pl.camera
    pos = np.array(cam.position)
    foc = np.array(cam.focal_point)
    cam.position = tuple(pos + direction)
    cam.focal_point = tuple(foc + direction)
    pl.reset_camera_clipping_range()
    pl.render()


def _move_forward(state: _ViewerState, sign: float) -> None:
    """Move along the camera view direction (positive = forward)."""
    pl = state.plotter
    if pl is None:
        return
    cam = pl.camera
    fwd = np.array(cam.focal_point) - np.array(cam.position)
    length = float(np.linalg.norm(fwd))
    if length < 1e-12:
        return
    fwd /= length
    _move_camera(state, fwd * (state.move_speed * sign))


def _strafe(state: _ViewerState, sign: float) -> None:
    """Strafe left/right (positive = right)."""
    pl = state.plotter
    if pl is None:
        return
    cam = pl.camera
    fwd = np.array(cam.focal_point) - np.array(cam.position)
    up = np.array(cam.view_up)
    right = np.cross(fwd, up)
    length = float(np.linalg.norm(right))
    if length < 1e-12:
        return
    right /= length
    _move_camera(state, right * (state.move_speed * sign))


def _move_vertical(state: _ViewerState, sign: float) -> None:
    """Move up/down along the world Y axis (positive = up)."""
    _move_camera(state, np.array([0.0, state.move_speed * sign, 0.0]))


def _adjust_speed(state: _ViewerState, factor: float) -> None:
    """Double or halve movement speed."""
    state.move_speed = max(state.move_speed * factor, 0.001)
    _refresh_info(state)
    if state.plotter is not None:
        state.plotter.render()


# ---------------------------------------------------------------------------
# Mouse-look state
# ---------------------------------------------------------------------------

def _enable_fps_style(state: _ViewerState) -> None:
    """Set a bare VTK interactor that ignores all mouse events.

    This prevents the default trackball from fighting our first-person
    mouse-look observers.
    """
    pl = state.plotter
    if pl is None:
        return
    iren = pl.iren.interactor  # type: ignore[union-attr]
    iren.SetInteractorStyle(vtkInteractorStyleUser())


def _toggle_cam_mode(state: _ViewerState) -> None:
    """Switch between first-person and orbit camera modes."""
    pl = state.plotter
    if pl is None:
        return
    if state.cam_mode == "fps":
        state.cam_mode = "orbit"
        pl.enable_trackball_style()
    else:
        state.cam_mode = "fps"
        _enable_fps_style(state)
        _sync_focal_point(state)
    _refresh_info(state)
    if pl is not None:
        pl.render()


class _MouseLook:
    """Tracks left-button drag to implement first-person mouse-look."""

    __slots__ = ("active", "last_x", "last_y")

    def __init__(self) -> None:
        self.active = False
        self.last_x = 0
        self.last_y = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_viewer(
    asset: GmdagAsset,
    *,
    initial_resolution: int = 128,
    level: float | None = None,
    config: InspectorConfig | None = None,
) -> None:
    """Open an interactive first-person 3D viewer for a .gmdag file.

    Args:
        asset: Loaded GmdagAsset.
        initial_resolution: coarse preview resolution (default 128).
        level: iso-surface level. None = auto-detect.
        config: Optional InspectorConfig for cache dir etc.
    """
    cfg = config or InspectorConfig()
    cache = MeshCache(cfg.cache_dir)
    state = _ViewerState(asset, initial_resolution, level, cache)

    # Extract voxel grid + mesh
    state.extract_data(initial_resolution)
    has_mesh = state.mesh is not None
    has_voxels = state.voxel_grid is not None and state.voxel_grid.n_cells > 0
    if not has_mesh and not has_voxels:
        msg = "Failed to extract any visualisation data"
        raise RuntimeError(msg)
    if not has_mesh:
        state.vis_mode = "voxels"

    pl = pv.Plotter(title=f"GMDAG Inspector \u2014 {asset.path.name}")
    state.plotter = pl

    pl.set_background("black", top="darkblue")

    # Movement speed = 3 % of scene bounding-box diagonal per tick.
    diag = float(np.linalg.norm(
        np.array(asset.bbox_max) - np.array(asset.bbox_min),
    ))
    state.move_speed = diag * 0.03

    # ---- First-person: disable VTK's default trackball interactor ----
    # VTK's default trackball style intercepts all mouse events (orbit,
    # pan, dolly) BEFORE our observers fire.  Setting a bare vtkInteractorStyle
    # makes VTK ignore the mouse so only our first-person observers run.
    _enable_fps_style(state)

    # ---- First-person mouse-look via VTK observers ----
    mlook = _MouseLook()

    def _on_left_down(obj: Any, event: str) -> None:  # noqa: ARG001
        if state.cam_mode != "fps":
            return  # let trackball handle it
        iren = pl.iren.interactor  # type: ignore[union-attr]
        mlook.last_x, mlook.last_y = iren.GetEventPosition()
        mlook.active = True

    def _on_left_up(obj: Any, event: str) -> None:  # noqa: ARG001
        mlook.active = False

    def _on_mouse_move(obj: Any, event: str) -> None:  # noqa: ARG001
        if state.cam_mode != "fps" or not mlook.active:
            return
        iren = pl.iren.interactor  # type: ignore[union-attr]
        x, y = iren.GetEventPosition()
        dx = x - mlook.last_x
        dy = y - mlook.last_y
        mlook.last_x, mlook.last_y = x, y
        if dx == 0 and dy == 0:
            return
        cam = pl.camera
        # Yaw/Pitch rotate around the camera position (first-person).
        cam.Yaw(-dx * _MOUSE_SENSITIVITY)
        cam.Pitch(dy * _MOUSE_SENSITIVITY)
        # Lock the up-vector so the horizon stays level.
        cam.SetViewUp(0.0, 1.0, 0.0)
        _sync_focal_point(state)
        pl.reset_camera_clipping_range()
        pl.render()

    def _wheel_fwd(obj: Any, event: str) -> None:  # noqa: ARG001
        _move_forward(state, 1.0)

    def _wheel_bwd(obj: Any, event: str) -> None:  # noqa: ARG001
        _move_forward(state, -1.0)

    iren = pl.iren.interactor  # type: ignore[union-attr]
    iren.AddObserver("LeftButtonPressEvent", _on_left_down)
    iren.AddObserver("LeftButtonReleaseEvent", _on_left_up)
    iren.AddObserver("MouseMoveEvent", _on_mouse_move)
    iren.AddObserver("MouseWheelForwardEvent", _wheel_fwd)
    iren.AddObserver("MouseWheelBackwardEvent", _wheel_bwd)

    # Render scene data
    _refresh_display(state)

    # Bounding box + axes
    state.bbox_actor = _add_bbox(pl, asset)
    pl.add_axes()

    # Place camera inside the scene — at centre, slightly above floor,
    # looking along -Z (matching actor convention: Y-up, forward = -Z).
    center = (np.array(asset.bbox_min) + np.array(asset.bbox_max)) / 2.0
    eye = center.copy()
    eye[1] = asset.bbox_min[1] + diag * 0.15  # ~15 % above floor
    focal = eye + np.array([0.0, 0.0, -_FOCAL_DISTANCE])
    pl.camera_position = [tuple(eye), tuple(focal), (0.0, 1.0, 0.0)]
    pl.reset_camera_clipping_range()

    # Info HUD
    _refresh_info(state)

    # ---- Display toggle keys ----
    pl.add_key_event("v", lambda: _toggle_cam_mode(state))
    pl.add_key_event("m", lambda: _toggle_vis_mode(state))
    pl.add_key_event("f", lambda: _toggle_wireframe(state))
    pl.add_key_event("h", lambda: _toggle_heatmap(state))
    pl.add_key_event("b", lambda: _toggle_bbox(state))
    pl.add_key_event("i", lambda: _toggle_info(state))
    pl.add_key_event("c", lambda: _toggle_clip(state))
    pl.add_key_event("r", lambda: _refine(state))
    pl.add_key_event("x", lambda: _export_current(state))

    # ---- Movement keys ----
    pl.add_key_event("w", lambda: _move_forward(state, 1.0))
    pl.add_key_event("s", lambda: _move_forward(state, -1.0))
    pl.add_key_event("Up", lambda: _move_forward(state, 1.0))
    pl.add_key_event("Down", lambda: _move_forward(state, -1.0))
    pl.add_key_event("a", lambda: _strafe(state, -1.0))
    pl.add_key_event("d", lambda: _strafe(state, 1.0))
    pl.add_key_event("Left", lambda: _strafe(state, -1.0))
    pl.add_key_event("Right", lambda: _strafe(state, 1.0))
    pl.add_key_event("q", lambda: _move_vertical(state, 1.0))
    pl.add_key_event("e", lambda: _move_vertical(state, -1.0))
    pl.add_key_event("plus", lambda: _adjust_speed(state, 2.0))
    pl.add_key_event("minus", lambda: _adjust_speed(state, 0.5))

    pl.show()
