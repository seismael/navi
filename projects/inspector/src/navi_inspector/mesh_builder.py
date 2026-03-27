"""SDF grid → triangle mesh via marching cubes.

Takes an SdfGrid from the DAG extractor and produces a PyVista PolyData
mesh using scikit-image's marching_cubes.  Supports optional mesh
simplification and export to OBJ/PLY/STL.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from navi_inspector.dag_extractor import SdfGrid

__all__: list[str] = ["build_mesh", "export_mesh"]

# Supported export formats mapped to PyVista save extensions
_EXPORT_FORMATS = {"ply", "obj", "stl"}


def build_mesh(
    sdf_grid: SdfGrid,
    *,
    level: float | None = None,
    simplify_ratio: float = 1.0,
) -> pv.PolyData:
    """Extract an isosurface mesh from a dense SDF grid.

    Args:
        sdf_grid: Dense SDF grid from extract_sdf_grid().
        level: Isosurface level. ``None`` (default) auto-selects: for unsigned
            distance fields (min ≥ 0) it uses ``voxel_size / 2``; for signed
            fields it uses ``0.0``.
        simplify_ratio: Fraction of triangles to keep (1.0 = no reduction,
            0.5 = keep ~50% of faces). Values < 1.0 triggers decimation.

    Returns:
        PyVista PolyData mesh in world coordinates, with SDF distance
        values attached as a 'sdf_distance' scalar field.

    Raises:
        ValueError: If no isosurface could be extracted (e.g., all void or
            all interior).
    """
    grid = sdf_grid.grid

    # Cap +inf / large sentinel values so marching cubes gets a finite volume
    max_cap = sdf_grid.voxel_size * sdf_grid.resolution
    finite_grid = np.where(np.isfinite(grid), grid, max_cap).astype(np.float32)

    # Auto-select iso-level when not specified
    if level is None:
        if float(finite_grid.min()) >= 0.0:
            # Unsigned distance field: surface sits where distance ≈ 0
            level = sdf_grid.voxel_size / 2.0
        else:
            level = 0.0

    # Check that there's actually a surface crossing
    has_below = bool(np.any(finite_grid < level))
    has_above = bool(np.any(finite_grid > level))
    if not (has_below and has_above):
        msg = (
            f"No isosurface at level={level:.6f}: "
            f"grid range [{float(finite_grid.min()):.3f}, {float(finite_grid.max()):.3f}]"
        )
        raise ValueError(msg)

    verts, faces, normals, values = marching_cubes(
        finite_grid,
        level=level,
        spacing=(sdf_grid.voxel_size, sdf_grid.voxel_size, sdf_grid.voxel_size),
    )

    # Translate from grid-local coordinates to world coordinates
    verts[:, 0] += sdf_grid.bbox_min[0]
    verts[:, 1] += sdf_grid.bbox_min[1]
    verts[:, 2] += sdf_grid.bbox_min[2]

    # Build PyVista PolyData: faces need the "n v0 v1 v2" format
    n_faces = faces.shape[0]
    pv_faces = np.empty((n_faces, 4), dtype=np.int64)
    pv_faces[:, 0] = 3
    pv_faces[:, 1:] = faces
    mesh = pv.PolyData(verts, pv_faces.ravel())

    # Attach SDF values as a scalar field for heatmap colouring
    mesh.point_data["sdf_distance"] = values
    # Attach vertex normals from marching cubes
    mesh.point_data["Normals"] = normals

    # Optional simplification
    if 0.0 < simplify_ratio < 1.0:
        target_reduction = 1.0 - simplify_ratio
        mesh = mesh.decimate(target_reduction)

    mesh.compute_normals(inplace=True)
    return mesh


def export_mesh(mesh: pv.PolyData, path: str | Path, fmt: str = "ply") -> Path:
    """Export a PyVista mesh to a file.

    Args:
        mesh: The mesh to export.
        path: Output file path. Extension is added/overridden to match fmt.
        fmt: Output format ('ply', 'obj', or 'stl').

    Returns:
        Path to the saved file.
    """
    fmt_lower = fmt.lower()
    if fmt_lower not in _EXPORT_FORMATS:
        msg = f"Unsupported format '{fmt}'. Supported: {', '.join(sorted(_EXPORT_FORMATS))}"
        raise ValueError(msg)

    out = Path(path).with_suffix(f".{fmt_lower}")
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(out))
    return out
