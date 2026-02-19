"""Tests for DistancePruner and OcclusionCuller."""

from __future__ import annotations

import numpy as np

from navi_section_manager.pruning import DistancePruner, OcclusionCuller


class TestDistancePruner:
    """Unit tests for distance-based voxel pruning."""

    def test_prune_removes_far_voxels(self) -> None:
        pruner = DistancePruner(threshold=5.0)
        voxels = np.array(
            [
                [0, 0, 0, 1.0, 1.0],
                [10, 0, 0, 1.0, 1.0],
                [3, 0, 0, 1.0, 2.0],
            ],
            dtype=np.float32,
        )
        center = np.array([0, 0, 0], dtype=np.float32)
        result = pruner.prune(voxels, center)
        assert result.shape[0] == 2  # only (0,0,0) and (3,0,0) survive
        assert result[0, 0] == 0.0
        assert result[1, 0] == 3.0

    def test_prune_empty_input(self) -> None:
        pruner = DistancePruner(threshold=10.0)
        voxels = np.empty((0, 5), dtype=np.float32)
        center = np.array([0, 0, 0], dtype=np.float32)
        result = pruner.prune(voxels, center)
        assert result.shape == (0, 5)

    def test_prune_keeps_all_within_range(self) -> None:
        pruner = DistancePruner(threshold=100.0)
        voxels = np.array(
            [
                [1, 1, 1, 0.5, 0.0],
                [2, 2, 2, 0.5, 0.0],
            ],
            dtype=np.float32,
        )
        center = np.array([0, 0, 0], dtype=np.float32)
        result = pruner.prune(voxels, center)
        assert result.shape[0] == 2


class TestOcclusionCuller:
    """Unit tests for sector-based occlusion culling."""

    def test_cull_empty_input(self) -> None:
        culler = OcclusionCuller()
        voxels = np.empty((0, 5), dtype=np.float32)
        eye = np.array([0, 0, 0], dtype=np.float32)
        result = culler.cull(voxels, eye)
        assert result.shape == (0, 5)

    def test_cull_keeps_close_voxels(self) -> None:
        culler = OcclusionCuller(sectors_theta=8, sectors_phi=4, density_threshold=0.8)
        # Two voxels in same direction: close (dense) and far (dense)
        voxels = np.array(
            [
                [5, 0, 0, 1.0, 1.0],  # close, dense
                [20, 0, 0, 1.0, 1.0],  # far, dense, same direction
            ],
            dtype=np.float32,
        )
        eye = np.array([0, 0, 0], dtype=np.float32)
        result = culler.cull(voxels, eye)
        # The far voxel should be culled (occluded by close one)
        assert result.shape[0] == 1
        assert result[0, 0] == 5.0

    def test_cull_keeps_sparse_voxels(self) -> None:
        culler = OcclusionCuller(sectors_theta=8, sectors_phi=4, density_threshold=0.8)
        # A non-dense voxel won't block anything
        voxels = np.array(
            [
                [5, 0, 0, 0.3, 1.0],  # close, NOT dense
                [20, 0, 0, 1.0, 1.0],  # far, dense
            ],
            dtype=np.float32,
        )
        eye = np.array([0, 0, 0], dtype=np.float32)
        result = culler.cull(voxels, eye)
        # Both should survive — the close one isn't dense enough to block
        assert result.shape[0] == 2
