"""Tests for MazeGenerator."""

from __future__ import annotations

import numpy as np
import pytest

from navi_environment.generators.maze import MazeGenerator


class TestMazeGenerator:
    """Unit tests for the procedural 3D maze generator."""

    def test_generate_chunk_shape(self) -> None:
        gen = MazeGenerator(seed=42, chunk_size=8)
        chunk = gen.generate_chunk(0, 0, 0)
        assert chunk.shape == (8, 8, 8, 2)
        assert chunk.dtype == np.float32

    def test_ground_chunk_has_floor(self) -> None:
        gen = MazeGenerator(seed=42, chunk_size=8)
        chunk = gen.generate_chunk(0, 0, 0)
        # Floor layer (y=0) should be entirely solid
        assert np.all(chunk[:, 0, :, 0] == 1.0)

    def test_underground_chunk_is_solid(self) -> None:
        gen = MazeGenerator(seed=42, chunk_size=8)
        chunk = gen.generate_chunk(0, -1, 0)
        assert np.all(chunk[:, :, :, 0] == 1.0)

    def test_sky_chunk_is_air(self) -> None:
        gen = MazeGenerator(seed=42, chunk_size=8)
        chunk = gen.generate_chunk(0, 1, 0)
        assert np.all(chunk[:, :, :, 0] == 0.0)

    def test_deterministic_generation(self) -> None:
        gen1 = MazeGenerator(seed=123, chunk_size=8)
        gen2 = MazeGenerator(seed=123, chunk_size=8)
        chunk1 = gen1.generate_chunk(3, 0, 5)
        chunk2 = gen2.generate_chunk(3, 0, 5)
        np.testing.assert_array_equal(chunk1, chunk2)

    def test_different_seeds_differ(self) -> None:
        gen1 = MazeGenerator(seed=1, chunk_size=16)
        gen2 = MazeGenerator(seed=2, chunk_size=16)
        chunk1 = gen1.generate_chunk(0, 0, 0)
        chunk2 = gen2.generate_chunk(0, 0, 0)
        # Compare internal corridor layout (y=2 layer should differ)
        layer1 = chunk1[:, 2, :, 0]
        layer2 = chunk2[:, 2, :, 0]
        assert not np.array_equal(layer1, layer2)

    def test_spawn_position_is_valid(self) -> None:
        gen = MazeGenerator(seed=42, chunk_size=16)
        x, y, z = gen.spawn_position()
        # Should be inside the origin chunk
        assert 0 <= x <= 16
        assert y > 0  # above floor
        assert 0 <= z <= 16

    def test_no_color_in_voxels(self) -> None:
        """Voxel data is (density, semantic_id) only — no RGB."""
        gen = MazeGenerator(seed=42, chunk_size=8)
        chunk = gen.generate_chunk(0, 0, 0)
        # Last dimension must be exactly 2 (density, semantic_id)
        assert chunk.shape[-1] == 2


class TestOpen3DVoxelGenerator:
    """Unit tests for Open3D-native voxel generator."""

    def test_generate_chunk_shape(self) -> None:
        pytest.importorskip("open3d")
        from navi_environment.generators.open3d_voxel import Open3DVoxelGenerator

        gen = Open3DVoxelGenerator(seed=42, chunk_size=8, points_per_chunk=128)
        chunk = gen.generate_chunk(0, 0, 0)
        assert chunk.shape == (8, 8, 8, 2)
        assert chunk.dtype == np.float32

    def test_spawn_position_in_bounds(self) -> None:
        from navi_environment.generators.open3d_voxel import Open3DVoxelGenerator

        gen = Open3DVoxelGenerator(seed=42, chunk_size=16)
        x, y, z = gen.spawn_position()
        assert 0 <= x <= 16
        assert 0 <= z <= 16
        assert y > 0

    def test_deterministic_generation(self) -> None:
        pytest.importorskip("open3d")
        from navi_environment.generators.open3d_voxel import Open3DVoxelGenerator

        gen1 = Open3DVoxelGenerator(seed=123, chunk_size=8, points_per_chunk=128)
        gen2 = Open3DVoxelGenerator(seed=123, chunk_size=8, points_per_chunk=128)
        c1 = gen1.generate_chunk(1, 0, 2)
        c2 = gen2.generate_chunk(1, 0, 2)
        np.testing.assert_array_equal(c1, c2)
