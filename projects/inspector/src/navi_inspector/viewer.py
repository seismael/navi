"""PyVista interactive 3D viewer for .gmdag files.

Navigation uses VTK's built-in flight style — hold arrow keys or
WASD to fly, drag to look around.

Flight controls (built into VTK):
    Arrow Up  / W       fly forward
    Arrow Down / S      fly backward
    Arrow Left          turn left
    Arrow Right         turn right
    A                   fly up
    Z                   fly down
    +  /  -             speed up / slow down
    Mouse drag          look around (pitch + yaw)

Display toggles:
    M   toggle VOXEL ↔ SURFACE rendering mode
    F   wireframe (surface mode only)
    H   SDF heatmap colours on/off
    B   bounding-box wireframe
    I   info HUD overlay
    C   clipping plane
    R   refine to full DAG resolution
    E   export mesh to PLY
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleFlight

from navi_inspector.cache import MeshCache
from navi_inspector.config import InspectorConfig
from navi_inspector.dag_extractor import SdfGrid, extract_sdf_grid
from navi_inspector.gmdag_io import GmdagAsset
from navi_inspector.mesh_builder import build_mesh, export_mesh

__all__: list[str] = ["launch_viewer"]

# Show voxels within this many cell-widths of geometry
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

        # "voxels" (default) or "surface"
        self.vis_mode = "voxels"

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

        # Voxel blocks — always built
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
    """Threshold the SDF into solid near-surface voxel blocks.

    Each surviving cell is a hexahedron (cube) coloured by SDF distance.
    Doors, windows, and other openings appear as natural gaps because
    those voxels have large distance values that fall outside the
    threshold.
    """
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
    return (
        f"File: {a.path.name}\n"
        f"DAG Res: {a.resolution}  View Res: {state.current_resolution}\n"
        f"Voxel: {a.voxel_size:.4f} m  Nodes: {len(a.nodes):,}\n"
        f"Mode: {mode}  {geom}\n"
        f"\n"
        f"FLY: Arrows/WASD  A/Z=Up/Dn  +/-=Speed\n"
        f"LOOK: Mouse drag\n"
        f"[M] Mode  [F] Wire  [H] Heat  [B] Box\n"
        f"[C] Clip  [R] Refine  [I] Info  [E] Export"
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

    if state.vis_mode == "voxels":
        grid = state.voxel_grid
        if grid is None or grid.n_cells == 0:
            return
        kwargs: dict[str, Any] = {
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

    else:  # surface
        mesh = state.mesh
        if mesh is None:
            return
        style = "wireframe" if state.wireframe else "surface"
        kwargs = {
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
# Entry point
# ---------------------------------------------------------------------------

def launch_viewer(
    asset: GmdagAsset,
    *,
    initial_resolution: int = 128,
    level: float | None = None,
    config: InspectorConfig | None = None,
) -> None:
    """Open an interactive 3D viewer for a .gmdag file.

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
    has_voxels = state.voxel_grid is not None and state.voxel_grid.n_cells > 0
    has_mesh = state.mesh is not None
    if not has_voxels and not has_mesh:
        msg = "Failed to extract any visualisation data"
        raise RuntimeError(msg)
    if not has_voxels:
        state.vis_mode = "surface"

    pl = pv.Plotter(title=f"GMDAG Inspector \u2014 {asset.path.name}")
    state.plotter = pl

    pl.set_background("black", top="darkblue")

    # ---- Flight-style navigation ----
    # VTK flight style gives smooth arrow-key / WASD flying + mouse look.
    # This replaces default trackball-orbit that caused the "stuck at zoom" issue.
    diag = float(np.linalg.norm(
        np.array(asset.bbox_max) - np.array(asset.bbox_min),
    ))
    flight = vtkInteractorStyleFlight()
    flight.SetMotionStepSize(diag * 0.01)
    pl.iren.interactor.SetInteractorStyle(flight)

    # Render scene data
    _refresh_display(state)

    # Bounding box + axes
    state.bbox_actor = _add_bbox(pl, asset)
    pl.add_axes()

    # Info HUD
    _refresh_info(state)

    # ---- Display toggle keys ----
    # None of these conflict with the flight style's keys (arrows/WASD/A/Z/+/-)
    pl.add_key_event("m", lambda: _toggle_vis_mode(state))
    pl.add_key_event("f", lambda: _toggle_wireframe(state))
    pl.add_key_event("h", lambda: _toggle_heatmap(state))
    pl.add_key_event("b", lambda: _toggle_bbox(state))
    pl.add_key_event("i", lambda: _toggle_info(state))
    pl.add_key_event("c", lambda: _toggle_clip(state))
    pl.add_key_event("r", lambda: _refine(state))
    pl.add_key_event("e", lambda: _export_current(state))

    pl.show()
