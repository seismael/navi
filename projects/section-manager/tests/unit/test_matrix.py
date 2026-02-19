"""Tests for SparseVoxelGrid."""

from __future__ import annotations

import numpy as np

from navi_section_manager.matrix import SparseVoxelGrid


class TestSparseVoxelGrid:
    """Unit tests for the chunked voxel storage."""

    def test_set_and_get_chunk(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        data = np.ones((4, 4, 4, 2), dtype=np.float32)
        grid.set_chunk(0, 0, 0, data)
        retrieved = grid.get_chunk(0, 0, 0)
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, data)

    def test_get_missing_chunk_returns_none(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        assert grid.get_chunk(99, 99, 99) is None

    def test_remove_chunk(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        data = np.ones((4, 4, 4, 2), dtype=np.float32)
        grid.set_chunk(1, 2, 3, data)
        assert grid.chunk_count == 1
        grid.remove_chunk(1, 2, 3)
        assert grid.chunk_count == 0
        assert grid.get_chunk(1, 2, 3) is None

    def test_query_region_returns_non_air_voxels(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        data = np.zeros((4, 4, 4, 2), dtype=np.float32)
        # Place one dense voxel at local (1, 2, 3)
        data[1, 2, 3, 0] = 1.0  # density
        data[1, 2, 3, 1] = 5.0  # semantic_id
        grid.set_chunk(0, 0, 0, data)

        result = grid.query_region(
            np.array([0, 0, 0], dtype=np.int32),
            np.array([4, 4, 4], dtype=np.int32),
        )
        assert result.shape == (1, 5)
        assert result[0, 0] == 1.0  # world x
        assert result[0, 1] == 2.0  # world y
        assert result[0, 2] == 3.0  # world z
        assert result[0, 3] == 1.0  # density
        assert result[0, 4] == 5.0  # semantic_id

    def test_query_empty_region(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        result = grid.query_region(
            np.array([0, 0, 0], dtype=np.int32),
            np.array([4, 4, 4], dtype=np.int32),
        )
        assert result.shape == (0, 5)

    def test_chunk_count_and_occupied(self) -> None:
        grid = SparseVoxelGrid(chunk_size=4)
        assert grid.chunk_count == 0
        assert len(grid.occupied_chunks) == 0

        data = np.ones((4, 4, 4, 2), dtype=np.float32)
        grid.set_chunk(0, 0, 0, data)
        grid.set_chunk(1, 0, 0, data)
        assert grid.chunk_count == 2
        assert (0, 0, 0) in grid.occupied_chunks
        assert (1, 0, 0) in grid.occupied_chunks
