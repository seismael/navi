"""Tests for DAG → SDF grid extraction."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from navi_inspector.dag_extractor import SdfGrid, extract_sdf_grid
from navi_inspector.gmdag_io import GmdagAsset

_HEADER = struct.Struct("<4sIIffffI")
_LEAF_FLAG = np.uint64(1 << 63)
_MASK_SHIFT = 55


def _make_leaf(distance: float, semantic: int = 1) -> np.uint64:
    """Encode a leaf node word."""
    dist_bits = int(np.float16(distance).view(np.uint16))
    return np.uint64(_LEAF_FLAG | (semantic << 16) | dist_bits)


def _make_inner(mask: int, child_base: int) -> np.uint64:
    """Encode an inner node word."""
    return np.uint64((mask << _MASK_SHIFT) | child_base)


def _make_simple_dag() -> tuple[np.ndarray, int, float]:
    """Build a minimal 2-level DAG with distinct leaf values.

    Root (inner) at index 0:
        - child mask = 0xFF (all 8 octants occupied)
        - child base → array of 8 child pointer words at indices 1–8
        - each child pointer points to a leaf at indices 9–16

    Leaves at indices 9–16: distance = 0.1 * octant_index
    """
    resolution = 2
    voxel_size = 1.0

    leaves = []
    for i in range(8):
        leaves.append(_make_leaf(0.1 * i))

    # Leaf indices: 9, 10, 11, ..., 16
    # Child pointer words at indices 1-8, each pointing to leaf index 9+i
    child_ptrs = [np.uint64(9 + i) for i in range(8)]

    # Root: all octants occupied, child_base=1 (child pointer array starts at 1)
    root = _make_inner(0xFF, 1)

    dag = np.array(
        [root] + child_ptrs + leaves,
        dtype=np.uint64,
    )
    return dag, resolution, voxel_size


def _make_asset(
    dag: np.ndarray,
    resolution: int,
    voxel_size: float,
    bbox_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> GmdagAsset:
    extent = voxel_size * resolution
    return GmdagAsset(
        path=Path("test.gmdag"),
        version=1,
        resolution=resolution,
        bbox_min=bbox_min,
        bbox_max=(bbox_min[0] + extent, bbox_min[1] + extent, bbox_min[2] + extent),
        voxel_size=voxel_size,
        nodes=np.ascontiguousarray(dag),
    )


class TestExtractSdfGrid:
    """extract_sdf_grid() produces correct dense grids."""

    def test_single_leaf_fills_entire_grid(self) -> None:
        """A DAG with just one root leaf should fill the entire grid."""
        leaf = _make_leaf(0.42)
        dag = np.array([leaf], dtype=np.uint64)
        asset = _make_asset(dag, resolution=4, voxel_size=0.25)

        result = extract_sdf_grid(asset, target_resolution=4)
        assert isinstance(result, SdfGrid)
        assert result.grid.shape == (4, 4, 4)
        # All cells should have the leaf distance (fp16 precision)
        expected = float(np.float16(0.42))
        np.testing.assert_allclose(result.grid, expected, atol=0.01)

    def test_two_level_dag_octant_values(self) -> None:
        """Each octant should get its distinct leaf value."""
        dag, resolution, voxel_size = _make_simple_dag()
        asset = _make_asset(dag, resolution=resolution, voxel_size=voxel_size)

        result = extract_sdf_grid(asset, target_resolution=resolution)
        assert result.grid.shape == (2, 2, 2)

        # Octant 0 = (0,0,0), distance = 0.0
        assert result.grid[0, 0, 0] == pytest.approx(0.0, abs=0.01)
        # Octant 7 = (1,1,1), distance = 0.7
        assert result.grid[1, 1, 1] == pytest.approx(0.7, abs=0.05)

    def test_void_octant_stays_inf(self) -> None:
        """A void octant (mask bit = 0) should remain +inf in the grid."""
        # Root with only octant 0 occupied
        root = _make_inner(0x01, 1)  # only bit 0 set
        child_ptr = np.uint64(2)  # points to leaf at index 2
        leaf = _make_leaf(0.5)
        dag = np.array([root, child_ptr, leaf], dtype=np.uint64)
        asset = _make_asset(dag, resolution=2, voxel_size=1.0)

        result = extract_sdf_grid(asset, target_resolution=2)
        # Octant 0 → 0.5
        assert result.grid[0, 0, 0] == pytest.approx(0.5, abs=0.01)
        # All other octants → inf (void)
        assert result.grid[1, 0, 0] == np.inf
        assert result.grid[0, 1, 0] == np.inf
        assert result.grid[1, 1, 1] == np.inf

    def test_coarse_extraction(self) -> None:
        """target_resolution < asset.resolution should still produce a grid."""
        dag, resolution, voxel_size = _make_simple_dag()
        asset = _make_asset(dag, resolution=resolution, voxel_size=voxel_size)

        # Request coarser than native 2 → should clamp to 2
        result = extract_sdf_grid(asset, target_resolution=1)
        assert result.resolution == 1
        # Should have gotten some representative value from the DAG
        assert np.isfinite(result.grid[0, 0, 0])

    def test_sdf_grid_metadata(self) -> None:
        """Verify metadata on the returned SdfGrid."""
        leaf = _make_leaf(1.0)
        dag = np.array([leaf], dtype=np.uint64)
        asset = _make_asset(dag, resolution=8, voxel_size=0.5, bbox_min=(1.0, 2.0, 3.0))

        result = extract_sdf_grid(asset, target_resolution=4)
        assert result.bbox_min == (1.0, 2.0, 3.0)
        assert result.resolution == 4
        assert result.voxel_size == pytest.approx(1.0)  # 4.0 / 4
