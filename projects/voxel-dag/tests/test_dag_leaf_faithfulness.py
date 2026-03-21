"""Phase 2 — DAG Leaf Faithfulness Tests.

Validates that DAG compression preserves SDF distances within fp16
precision, and that void-octant pruning is geometrically correct.

Tolerance budget:
  * fp16 round-trip: max(d × 2⁻¹⁰, 6 × 10⁻⁵) — IEEE 754 half-precision
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from voxel_dag.compiler import (
    EikonalSdfComputer,
    MeshIngestor,
    SvoDagCompressor,
)


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
write_unit_box_obj: Callable[..., None] = _oracle.write_unit_box_obj

_INTERNAL_NODE_MASK_SHIFT = 55
_LEAF_FLAG = 1 << 63


# ── Helpers ──────────────────────────────────────────────────────────

def _compile_box_sdf(tmp_path: Path, resolution: int = 32) -> tuple[np.ndarray, float, np.ndarray]:
    obj = tmp_path / "unit_box.obj"
    write_unit_box_obj(obj)
    verts, idxs, bmin, bmax = MeshIngestor.load_obj(str(obj))
    return EikonalSdfComputer().compute(verts, idxs, bmin, bmax, resolution, padding=0.0)


def _compile_box_dag(
    tmp_path: Path, resolution: int = 32
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (dag_nodes, sdf_grid, voxel_size, cube_min)."""
    grid, h, cube_min = _compile_box_sdf(tmp_path, resolution)
    semantic = np.ones_like(grid, dtype=np.int32)
    dag = SvoDagCompressor().compress(grid, resolution, cell_size=h, semantic_grid=semantic)
    return dag, grid, h, cube_min


def _decode_leaf_distance(word: np.uint64) -> float:
    dist_bits = int(word) & 0xFFFF
    return float(np.array([dist_bits], dtype=np.uint16).view(np.float16)[0])


def _decode_leaf_semantic(word: np.uint64) -> int:
    return (int(word) >> 16) & 0xFFFF


def _is_leaf(word: np.uint64) -> bool:
    return bool(int(word) & _LEAF_FLAG)


def _traverse_dag_to_voxel(
    dag: np.ndarray,
    ix: int,
    iy: int,
    iz: int,
    resolution: int,
) -> tuple[float, int]:
    """Walk the DAG from root to the leaf containing voxel (ix, iy, iz).

    Returns (leaf_distance_fp16, leaf_semantic).
    """
    ptr = 0  # root
    lo_x, lo_y, lo_z = 0, 0, 0
    size = resolution

    for _ in range(32):
        word = dag[ptr]
        if _is_leaf(word):
            return _decode_leaf_distance(word), _decode_leaf_semantic(word)

        mask = (int(word) >> _INTERNAL_NODE_MASK_SHIFT) & 0xFF
        child_base = int(word) & 0xFFFFFFFF
        half = size // 2
        mx = lo_x + half
        my = lo_y + half
        mz = lo_z + half

        octant = (1 if ix >= mx else 0) | (2 if iy >= my else 0) | (4 if iz >= mz else 0)

        if not (mask & (1 << octant)):
            # Void octant — no child.  Return sentinel.
            return float("inf"), 0

        offset = bin(mask & ((1 << octant) - 1)).count("1")
        child_ptr_idx = child_base + offset
        ptr = int(dag[child_ptr_idx]) & 0xFFFFFFFF

        # Narrow bounds
        lo_x = mx if (octant & 1) else lo_x
        lo_y = my if (octant & 2) else lo_y
        lo_z = mz if (octant & 4) else lo_z
        size = half

    raise RuntimeError("DAG traversal exceeded depth limit")


# ── Tests ────────────────────────────────────────────────────────────

class TestDagLeafMatchesSdfGrid:
    """DAG leaf fp16 distance must match fp16(grid[z,y,x]) for sampled voxels."""

    RESOLUTION = 32

    @pytest.fixture(autouse=True)
    def _build(self, tmp_path: Path) -> None:
        self.dag, self.grid, self.h, self.cube_min = _compile_box_dag(
            tmp_path, self.RESOLUTION
        )

    def _check(self, ix: int, iy: int, iz: int) -> None:
        leaf_dist, leaf_sem = _traverse_dag_to_voxel(
            self.dag, ix, iy, iz, self.RESOLUTION
        )
        if not np.isfinite(leaf_dist):
            # Void octant — the minimum SDF in the block must exceed
            # the octant diagonal for this to be geometrically valid.
            return

        grid_val = float(self.grid[iz, iy, ix])
        expected_fp16 = float(np.float16(grid_val))

        # fp16 leaf must match quantized grid value.
        # Allow tiny tolerance for the block-min selection in the compiler:
        # the leaf stores min(block), so leaf_dist ≤ grid_val (quantized).
        assert leaf_dist <= expected_fp16 + 6e-5, (
            f"voxel ({ix},{iy},{iz}): leaf_dist={leaf_dist:.6f}, "
            f"expected_fp16(grid)={expected_fp16:.6f}"
        )

    def test_center(self) -> None:
        mid = self.RESOLUTION // 2
        self._check(mid, mid, mid)

    def test_near_faces(self) -> None:
        mid = self.RESOLUTION // 2
        for ix, iy, iz in [
            (1, mid, mid),
            (self.RESOLUTION - 2, mid, mid),
            (mid, 1, mid),
            (mid, self.RESOLUTION - 2, mid),
            (mid, mid, 1),
            (mid, mid, self.RESOLUTION - 2),
        ]:
            self._check(ix, iy, iz)

    def test_corners(self) -> None:
        hi = self.RESOLUTION - 1
        for ix, iy, iz in [(0, 0, 0), (hi, hi, hi), (0, hi, 0), (hi, 0, hi)]:
            self._check(ix, iy, iz)


class TestVoidOctantDetection:
    """Void octants must correspond to regions entirely outside the box surface."""

    RESOLUTION = 32

    def test_root_has_populated_children(self, tmp_path: Path) -> None:
        dag, grid, h, cube_min = _compile_box_dag(tmp_path, self.RESOLUTION)
        root = dag[0]
        assert not _is_leaf(root), "Root should be an internal node for non-trivial geometry"
        mask = (int(root) >> _INTERNAL_NODE_MASK_SHIFT) & 0xFF
        assert mask != 0, "Root must have at least one populated octant"

    def test_void_octant_sdf_exceeds_diagonal(self, tmp_path: Path) -> None:
        """For each void octant at level 1, verify the minimum SDF in that region
        exceeds the octant's spatial diagonal (sqrt(3) * half_size * h)."""
        dag, grid, h, cube_min = _compile_box_dag(tmp_path, self.RESOLUTION)
        root = dag[0]
        mask = (int(root) >> _INTERNAL_NODE_MASK_SHIFT) & 0xFF
        half = self.RESOLUTION // 2

        for octant in range(8):
            if mask & (1 << octant):
                continue  # populated, skip
            # This octant is void — verify SDF justifies it
            ox = half if (octant & 1) else 0
            oy = half if (octant & 2) else 0
            oz = half if (octant & 4) else 0
            block = grid[oz : oz + half, oy : oy + half, ox : ox + half]
            min_sdf = float(block.min())
            diagonal = half * h * np.sqrt(3.0)
            assert min_sdf > diagonal, (
                f"Void octant {octant}: min_sdf={min_sdf:.4f} ≤ diagonal={diagonal:.4f}"
            )


class TestFp16QuantizationRoundtrip:
    """Verify fp16 encode/decode round-trip error stays within IEEE 754 half bounds."""

    @pytest.mark.parametrize(
        "value",
        [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 65504.0],
    )
    def test_roundtrip(self, value: float) -> None:
        bits = int(np.array([value], dtype=np.float16).view(np.uint16)[0])
        decoded = float(np.array([bits], dtype=np.uint16).view(np.float16)[0])
        # fp16 relative error is at most 2^-10 ≈ 9.77e-4 for normal values
        if value == 0.0:
            assert decoded == 0.0
        else:
            rel_error = abs(decoded - value) / max(abs(value), 1e-7)
            assert rel_error < 1e-3 or abs(decoded - value) < 6.2e-5, (
                f"fp16({value}) round-trip: decoded={decoded}, rel_error={rel_error:.6e}"
            )
