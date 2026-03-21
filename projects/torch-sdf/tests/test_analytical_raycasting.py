"""Phase 4 — Analytical CUDA Ray Casting Tests.

Validates that the CUDA sphere-tracing kernel returns correct distances
for rays cast against the compiled unit box, compared to analytical
ray-box intersection.

Requires CUDA.  Skipped gracefully when unavailable.

Tolerance budget: kHitEpsilon (0.01) + voxel_size/2 + fp16 quantization
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
# Add voxel-dag to path for compiler imports (torch-sdf venv lacks navi_contracts)
_VOXEL_DAG_SRC = str(_REPO / "projects" / "voxel-dag")
if _VOXEL_DAG_SRC not in sys.path:
    sys.path.insert(0, _VOXEL_DAG_SRC)

import torch  # noqa: E402 — import after sys.path fix


def _load_oracle_box() -> Any:
    helper_path = _REPO / "projects" / "contracts" / "src" / "navi_contracts" / "testing" / "oracle_box.py"
    spec = importlib.util.spec_from_file_location("navi_test_oracle_box", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load oracle_box from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_oracle = _load_oracle_box()
analytical_ray_box_distance: Callable[..., float | None] = _oracle.analytical_ray_box_distance
write_unit_box_obj: Callable[..., None] = _oracle.write_unit_box_obj

from voxel_dag.compiler import (  # noqa: E402
    EikonalSdfComputer,
    MeshIngestor,
    SvoDagCompressor,
)

_HIT_EPSILON = 0.01
_RESOLUTION = 64
_MAX_STEPS = 128
_MAX_DISTANCE = 20.0


# ── Module-level compiled box (compile ONCE for speed) ──────────────

_BOX_DAG_CACHE: dict[str, Any] = {}


def _get_box_dag(semantic_id: int = 1) -> tuple["torch.Tensor", list[float], list[float], int, float]:
    """Return cached (dag_tensor, bbox_min, bbox_max, res, voxel_size)."""
    key = f"sem{semantic_id}"
    if key not in _BOX_DAG_CACHE:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        tmp = Path(tempfile.mkdtemp())
        obj = tmp / "unit_box.obj"
        write_unit_box_obj(obj)
        verts, idxs, bmin, bmax = MeshIngestor.load_obj(str(obj))
        grid, h, cube_min = EikonalSdfComputer().compute(
            verts, idxs, bmin, bmax, _RESOLUTION, padding=0.0
        )
        sem = np.full_like(grid, semantic_id, dtype=np.int32)
        dag_np = SvoDagCompressor().compress(grid, _RESOLUTION, cell_size=h, semantic_grid=sem)
        dag_t = torch.from_numpy(dag_np.view(np.int64)).cuda()
        cube_max = cube_min + _RESOLUTION * h
        _BOX_DAG_CACHE[key] = (dag_t, cube_min.tolist(), cube_max.tolist(), _RESOLUTION, h)
    return _BOX_DAG_CACHE[key]


def _cast_single_ray(
    dag: "torch.Tensor",
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    bbox_min: list[float],
    bbox_max: list[float],
    resolution: int,
) -> tuple[float, int]:
    """Cast one ray and return (distance, semantic)."""
    import torch_sdf

    o = torch.tensor([[list(origin)]], device="cuda", dtype=torch.float32)
    d = torch.tensor([[list(direction)]], device="cuda", dtype=torch.float32)
    out_d = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
    out_s = torch.zeros((1, 1), device="cuda", dtype=torch.int32)

    torch_sdf.cast_rays(dag, o, d, out_d, out_s, _MAX_STEPS, _MAX_DISTANCE, bbox_min, bbox_max, resolution)
    return float(out_d[0, 0].item()), int(out_s[0, 0].item())


# ── Tests ────────────────────────────────────────────────────────────


class TestAxisAlignedRays:
    """Origin at box center (0, 1, 0), rays along ±X, ±Y, ±Z."""

    @pytest.fixture(autouse=True)
    def _build(self) -> None:
        self.dag, self.bmin, self.bmax, self.res, self.h = _get_box_dag()
        # Tolerance: SDF discretization can be off by several voxels near
        # mesh edges + kHitEpsilon + fp16 quantization
        self.tol = 5 * self.h + _HIT_EPSILON + 0.01
        self.origin = (0.0, 1.0, 0.0)

    @pytest.mark.parametrize(
        "direction,expected_distance_approx",
        [
            ((1.0, 0.0, 0.0), 1.0),   # +X wall 1 m away
            ((-1.0, 0.0, 0.0), 1.0),  # -X wall 1 m away
            ((0.0, 1.0, 0.0), 1.0),   # ceiling 1 m away
            ((0.0, -1.0, 0.0), 1.0),  # floor 1 m away
            ((0.0, 0.0, 1.0), 1.0),   # +Z wall 1 m away
            ((0.0, 0.0, -1.0), 1.0),  # -Z wall 1 m away
        ],
    )
    def test_hit_distance(self, direction: tuple[float, float, float], expected_distance_approx: float) -> None:
        expected_exact = analytical_ray_box_distance(self.origin, direction)
        assert expected_exact is not None, f"Analytical oracle says ray misses for dir={direction}"

        actual_dist, actual_sem = _cast_single_ray(
            self.dag, self.origin, direction, self.bmin, self.bmax, self.res
        )
        assert actual_sem > 0, f"Ray {direction} should hit (semantic > 0), got {actual_sem}"
        assert abs(actual_dist - expected_exact) < self.tol, (
            f"dir={direction}: CUDA dist={actual_dist:.4f}, expected={expected_exact:.4f}, "
            f"tol={self.tol:.4f}"
        )


class TestDiagonalRays:
    """45-degree rays from box center must hit at analytically predictable distances."""

    @pytest.fixture(autouse=True)
    def _build(self) -> None:
        self.dag, self.bmin, self.bmax, self.res, self.h = _get_box_dag()
        self.tol = 5 * self.h + _HIT_EPSILON + 0.01
        self.origin = (0.0, 1.0, 0.0)

    @pytest.mark.parametrize(
        "raw_direction",
        [
            (1.0, 0.0, 1.0),
            (1.0, 0.0, -1.0),
            (-1.0, 0.0, 1.0),
            (-1.0, 0.0, -1.0),
            (0.0, 1.0, 1.0),
            (0.0, -1.0, -1.0),
        ],
    )
    def test_diagonal_hit(self, raw_direction: tuple[float, float, float]) -> None:
        norm = math.sqrt(sum(c * c for c in raw_direction))
        d = tuple(c / norm for c in raw_direction)
        expected = analytical_ray_box_distance(self.origin, d)
        assert expected is not None, f"Analytical oracle says diagonal ray misses: {d}"

        actual_dist, actual_sem = _cast_single_ray(
            self.dag, self.origin, d, self.bmin, self.bmax, self.res
        )
        assert actual_sem > 0, f"Diagonal ray {d} should hit"
        assert abs(actual_dist - expected) < self.tol, (
            f"dir={d}: CUDA dist={actual_dist:.4f}, expected={expected:.4f}"
        )


class TestMissRays:
    """Rays originating outside the box and pointing away must miss."""

    @pytest.fixture(autouse=True)
    def _build(self) -> None:
        self.dag, self.bmin, self.bmax, self.res, self.h = _get_box_dag()

    def test_ray_pointing_away(self) -> None:
        origin = (5.0, 1.0, 0.0)  # outside +X
        direction = (1.0, 0.0, 0.0)  # pointing further away
        dist, sem = _cast_single_ray(
            self.dag, origin, direction, self.bmin, self.bmax, self.res
        )
        assert sem == 0, f"Miss ray should yield semantic 0, got {sem}"
        assert dist >= _MAX_DISTANCE * 0.9, (
            f"Miss ray distance {dist:.2f} should be near max_distance {_MAX_DISTANCE}"
        )


class TestSemanticPropagation:
    """Compiled with semantic_id=7, every hit must report that ID."""

    def test_custom_semantic(self) -> None:
        dag, bmin, bmax, res, h = _get_box_dag(semantic_id=7)
        origin = (0.0, 1.0, 0.0)
        direction = (1.0, 0.0, 0.0)
        _, sem = _cast_single_ray(dag, origin, direction, bmin, bmax, res)
        assert sem == 7, f"Expected semantic 7, got {sem}"
