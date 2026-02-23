"""Orientation correctness tests for DistanceMatrixBuilder."""

from __future__ import annotations

import numpy as np

from navi_contracts import RobotPose
from navi_environment.distance_matrix_v2 import DistanceMatrixBuilder


class TestDistanceMatrixOrientation:
    """Validates spherical-view alignment with robot pose and yaw."""

    def test_yaw_alignment_produces_equivalent_forward_view(self) -> None:
        builder = DistanceMatrixBuilder(azimuth_bins=64, elevation_bins=32, max_distance=30.0)

        voxels_forward_x = np.array([[6.0, 0.0, 0.0, 1.0, 6.0]], dtype=np.float32)
        voxels_forward_z = np.array([[0.0, 0.0, 6.0, 1.0, 6.0]], dtype=np.float32)

        pose_yaw_zero = RobotPose(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            timestamp=0.0,
        )
        pose_yaw_ninety = RobotPose(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=np.pi / 2.0,
            timestamp=0.0,
        )

        matrix_zero = builder.build(
            voxels=voxels_forward_x,
            pose=pose_yaw_zero,
            step_id=1,
            timestamp=1.0,
        )
        matrix_ninety = builder.build(
            voxels=voxels_forward_z,
            pose=pose_yaw_ninety,
            step_id=2,
            timestamp=2.0,
        )

        np.testing.assert_array_equal(matrix_zero.valid_mask, matrix_ninety.valid_mask)
        np.testing.assert_allclose(matrix_zero.depth, matrix_ninety.depth)
        np.testing.assert_array_equal(matrix_zero.semantic, matrix_ninety.semantic)

    def test_forward_voxel_hits_center_azimuth_bin(self) -> None:
        azimuth_bins = 64
        elevation_bins = 32
        builder = DistanceMatrixBuilder(
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            max_distance=30.0,
        )

        voxels = np.array([[8.0, 0.0, 0.0, 1.0, 4.0]], dtype=np.float32)
        pose = RobotPose(
            x=0.0,
            y=0.0,
            z=0.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            timestamp=0.0,
        )
        matrix = builder.build(voxels=voxels, pose=pose, step_id=1, timestamp=1.0)

        valid = matrix.valid_mask[0]
        hit_indices = np.argwhere(valid)
        assert hit_indices.shape[0] == 1

        az_idx, el_idx = int(hit_indices[0, 0]), int(hit_indices[0, 1])
        assert abs(az_idx - (azimuth_bins // 2)) <= 1
        assert abs(el_idx - (elevation_bins // 2)) <= 1
