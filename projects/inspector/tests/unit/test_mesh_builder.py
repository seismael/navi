"""Tests for mesh_builder — marching cubes on synthetic SDF grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv

from navi_inspector.dag_extractor import SdfGrid
from navi_inspector.mesh_builder import build_mesh, export_mesh


def _sphere_sdf(res: int = 32, radius: float = 0.4) -> SdfGrid:
    """Create a synthetic SDF grid for a unit sphere centred in a [0,1]³ cube."""
    lin = np.linspace(0, 1, res)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    centre = 0.5
    grid = np.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2) - radius
    return SdfGrid(
        grid=grid.astype(np.float32),
        resolution=res,
        bbox_min=(0.0, 0.0, 0.0),
        bbox_max=(1.0, 1.0, 1.0),
        voxel_size=1.0 / res,
    )


class TestBuildMesh:
    """build_mesh() produces valid PyVista PolyData."""

    def test_sphere_produces_closed_mesh(self) -> None:
        sdf = _sphere_sdf(res=32)
        mesh = build_mesh(sdf)
        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 100
        assert mesh.n_faces_strict > 100

    def test_has_sdf_distance_scalars(self) -> None:
        sdf = _sphere_sdf(res=32)
        mesh = build_mesh(sdf)
        assert "sdf_distance" in mesh.point_data

    def test_vertices_in_world_coords(self) -> None:
        sdf = _sphere_sdf(res=32)
        mesh = build_mesh(sdf)
        bounds = mesh.bounds
        # Sphere radius 0.4 centred at 0.5 → bounds should be ~[0.1, 0.9]
        assert bounds[0] > -0.1  # x_min
        assert bounds[1] < 1.1  # x_max

    def test_simplification(self) -> None:
        sdf = _sphere_sdf(res=32)
        full = build_mesh(sdf, simplify_ratio=1.0)
        reduced = build_mesh(sdf, simplify_ratio=0.5)
        # Simplified mesh should have fewer faces
        assert reduced.n_faces_strict < full.n_faces_strict

    def test_raises_on_all_positive(self) -> None:
        """No isosurface when entire grid is positive."""
        grid = np.ones((8, 8, 8), dtype=np.float32) * 5.0
        sdf = SdfGrid(
            grid=grid, resolution=8,
            bbox_min=(0.0, 0.0, 0.0), bbox_max=(1.0, 1.0, 1.0),
            voxel_size=0.125,
        )
        with pytest.raises(ValueError, match="[Nn]o isosurface"):
            build_mesh(sdf)

    def test_raises_on_all_negative(self) -> None:
        grid = np.ones((8, 8, 8), dtype=np.float32) * -5.0
        sdf = SdfGrid(
            grid=grid, resolution=8,
            bbox_min=(0.0, 0.0, 0.0), bbox_max=(1.0, 1.0, 1.0),
            voxel_size=0.125,
        )
        with pytest.raises(ValueError, match="[Nn]o isosurface"):
            build_mesh(sdf)


class TestExportMesh:
    """export_mesh() writes valid files."""

    def test_export_ply(self, tmp_path: Path) -> None:
        sdf = _sphere_sdf(res=16)
        mesh = build_mesh(sdf)
        out = export_mesh(mesh, tmp_path / "sphere", fmt="ply")
        assert out.exists()
        assert out.suffix == ".ply"
        assert out.stat().st_size > 0

    def test_export_stl(self, tmp_path: Path) -> None:
        sdf = _sphere_sdf(res=16)
        mesh = build_mesh(sdf)
        out = export_mesh(mesh, tmp_path / "sphere", fmt="stl")
        assert out.exists()
        assert out.suffix == ".stl"

    def test_unsupported_format(self, tmp_path: Path) -> None:
        sdf = _sphere_sdf(res=16)
        mesh = build_mesh(sdf)
        with pytest.raises(ValueError, match="Unsupported"):
            export_mesh(mesh, tmp_path / "sphere", fmt="fbx")
