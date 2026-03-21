"""Phase 1 — Analytical SDF Oracle Tests.

Validates that the Eikonal fast-sweeping SDF compiler produces correct
distance values for the canonical unit box, where the analytical unsigned
SDF is known exactly at every point in space.

Tolerance budget: 1.5 × voxel_size (discretization error from grid sampling).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from voxel_dag.compiler import EikonalSdfComputer, MeshIngestor


def _load_oracle_box() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    helper_path = repo_root / "projects" / "contracts" / "src" / "navi_contracts" / "testing" / "oracle_box.py"
    spec = importlib.util.spec_from_file_location("navi_test_oracle_box", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load oracle_box from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_oracle = _load_oracle_box()
analytical_unsigned_sdf: Callable[..., float] = _oracle.analytical_unsigned_sdf
write_unit_box_obj: Callable[..., None] = _oracle.write_unit_box_obj


# ── Helpers ──────────────────────────────────────────────────────────

def _compile_unit_box(tmp_path: Path, resolution: int = 32) -> tuple[np.ndarray, float, np.ndarray]:
    """Compile the canonical unit box and return (grid, voxel_size, cube_min)."""
    obj = tmp_path / "unit_box.obj"
    write_unit_box_obj(obj)
    vertices, indices, bbox_min, bbox_max = MeshIngestor.load_obj(str(obj))
    computer = EikonalSdfComputer()
    return computer.compute(vertices, indices, bbox_min, bbox_max, resolution, padding=0.0)


def _voxel_center(ix: int, iy: int, iz: int, cube_min: np.ndarray, h: float) -> tuple[float, float, float]:
    """World-space center of voxel (ix, iy, iz).  Grid is indexed [z, y, x]."""
    return (
        float(cube_min[0]) + (ix + 0.5) * h,
        float(cube_min[1]) + (iy + 0.5) * h,
        float(cube_min[2]) + (iz + 0.5) * h,
    )


# ── Tests ────────────────────────────────────────────────────────────

class TestSdfMatchesAnalyticalDistance:
    """Sample specific voxels and compare to the analytical unsigned SDF."""

    RESOLUTION = 32

    @pytest.fixture(autouse=True)
    def _compile(self, tmp_path: Path) -> None:
        self.grid, self.h, self.cube_min = _compile_unit_box(tmp_path, self.RESOLUTION)
        # cube_max for quick reference
        self.cube_max = self.cube_min + self.RESOLUTION * self.h

    # ── Parametrised sample points ──────────────────────────────────
    # We pick points that are: on faces, at corners, in far field, and at center.

    def _check(self, ix: int, iy: int, iz: int, *, rtol: float = 0.0) -> None:
        """Assert grid[z,y,x] ≈ analytical_sdf(center(ix,iy,iz))."""
        cx, cy, cz = _voxel_center(ix, iy, iz, self.cube_min, self.h)
        expected = analytical_unsigned_sdf(cx, cy, cz)
        actual = float(self.grid[iz, iy, ix])
        tol = 1.5 * self.h + rtol
        assert abs(actual - expected) < tol, (
            f"voxel ({ix},{iy},{iz}) center ({cx:.4f},{cy:.4f},{cz:.4f}): "
            f"expected SDF={expected:.6f}, got {actual:.6f}, tol={tol:.6f}"
        )

    def test_center_of_box(self) -> None:
        """Center of box (0, 1, 0) → distance = 1.0 (nearest face at 1 m)."""
        mid = self.RESOLUTION // 2
        self._check(mid, mid, mid)

    def test_near_face_x_plus(self) -> None:
        """Point near the +X face should have small distance."""
        # The +X face is at x=1.0.  Voxel near the right boundary.
        near = self.RESOLUTION - 2  # close to +X face
        mid = self.RESOLUTION // 2
        self._check(near, mid, mid)

    def test_near_face_x_minus(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(1, mid, mid)

    def test_near_face_y_bottom(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(mid, 1, mid)

    def test_near_face_y_top(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(mid, self.RESOLUTION - 2, mid)

    def test_near_face_z_plus(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(mid, mid, self.RESOLUTION - 2)

    def test_near_face_z_minus(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(mid, mid, 1)

    def test_corner_region(self) -> None:
        """Corner voxel (near 7 of 8 corners) should have a distance
        consistent with the minimum distance to the nearest face/edge/corner.
        """
        # Near the (x=+1, y=0, z=+1) corner
        hi = self.RESOLUTION - 2
        self._check(hi, 1, hi)

    def test_far_field(self) -> None:
        """Voxel far outside the geometry (padded region) should have
        large distance matching the analytical expectation."""
        # With padding=0.0 the cube domain exactly wraps the geometry,
        # but diagonal voxels near corner of the resolution cube still
        # measure distance to the nearest face.
        self._check(0, 0, 0)
        self._check(self.RESOLUTION - 1, self.RESOLUTION - 1, self.RESOLUTION - 1)


class TestSdfGridDeterminism:
    """Prove repeated compilation produces byte-identical grids."""

    def test_two_compilations_are_identical(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        g1, h1, c1 = _compile_unit_box(dir_a, resolution=32)
        g2, h2, c2 = _compile_unit_box(dir_b, resolution=32)
        assert h1 == h2, f"voxel_size mismatch: {h1} vs {h2}"
        np.testing.assert_array_equal(c1, c2, err_msg="cube_min mismatch")
        np.testing.assert_array_equal(g1, g2, err_msg="grid mismatch")


class TestNearSurfaceVoxels:
    """Voxels whose centers lie close to a known face should report small distance."""

    RESOLUTION = 64

    def test_surface_voxels_have_small_distance(self, tmp_path: Path) -> None:
        grid, h, cube_min = _compile_unit_box(tmp_path, self.RESOLUTION)
        res = self.RESOLUTION
        violations: list[str] = []

        # Check voxels near the +X face (x ≈ 1.0)
        for iy in range(2, res - 2):
            for iz in range(2, res - 2):
                for ix in range(res):
                    cx, cy, cz = _voxel_center(ix, iy, iz, cube_min, h)
                    # Is this voxel center near the +X face?
                    if abs(cx - 1.0) < 0.5 * h and 0.0 < cy < 2.0 and -1.0 < cz < 1.0:
                        d = float(grid[iz, iy, ix])
                        if d > h:
                            violations.append(
                                f"({ix},{iy},{iz}) center ({cx:.4f},{cy:.4f},{cz:.4f}) "
                                f"dist={d:.6f} > h={h:.6f}"
                            )

        assert not violations, (
            f"{len(violations)} voxels near +X face exceed distance threshold:\n"
            + "\n".join(violations[:10])
        )
