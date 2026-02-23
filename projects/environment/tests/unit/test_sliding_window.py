"""Tests for SlidingWindow."""

from __future__ import annotations

import numpy as np
import pytest

from navi_environment.matrix import SparseVoxelGrid
from navi_environment.sliding_window import SlidingWindow, WedgeResult


def _make_solid_chunk(cx: int, cy: int, cz: int) -> np.ndarray:
    """Generate a chunk with one dense voxel at local (0, 0, 0)."""
    chunk = np.zeros((4, 4, 4, 2), dtype=np.float32)
    chunk[0, 0, 0, 0] = 1.0  # density
    chunk[0, 0, 0, 1] = 1.0  # semantic_id = wall
    return chunk


class TestSlidingWindow:
    """Unit tests for the sliding window mechanism."""

    def test_initial_shift_populates_window(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        window = SlidingWindow(grid, radius=1)
        result = window.shift(2.0, 2.0, 2.0, _make_solid_chunk)
        # radius=1 → 3x3x3 = 27 chunks
        assert grid.chunk_count == 27
        assert result.new_voxels.shape[1] == 5
        assert result.culled_count == 0

    def test_shift_to_same_position_no_change(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        window = SlidingWindow(grid, radius=1)
        window.shift(2.0, 2.0, 2.0, _make_solid_chunk)

        result = window.shift(2.0, 2.0, 2.0, _make_solid_chunk)
        # No movement → no entering/exiting chunks
        assert result.new_voxels.shape[0] == 0
        assert result.culled_count == 0

    def test_shift_moves_window(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        window = SlidingWindow(grid, radius=1)
        window.shift(2.0, 2.0, 2.0, _make_solid_chunk)

        # Move one chunk in x direction (4 units with chunk_size=4)
        result = window.shift(6.0, 2.0, 2.0, _make_solid_chunk)
        # Some chunks entered, some exited
        assert result.culled_count > 0
        assert result.new_voxels.shape[0] > 0
        # Total chunks should remain 27
        assert grid.chunk_count == 27

    def test_world_to_chunk(self) -> None:
        grid = SparseVoxelGrid(chunk_size=16)
        window = SlidingWindow(grid, radius=1)
        assert window.world_to_chunk(17.0, 33.0, -1.0) == (1, 2, -1)
        assert window.world_to_chunk(0.0, 0.0, 0.0) == (0, 0, 0)

    def test_wedge_result_is_frozen(self) -> None:
        result = WedgeResult(
            new_voxels=np.empty((0, 5), dtype=np.float32),
            culled_count=0,
        )
        with pytest.raises(AttributeError):
            result.culled_count = 5  # type: ignore[misc]
