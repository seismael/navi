from __future__ import annotations

import numpy as np
import pytest

from navi_contracts import RobotPose
from navi_environment.backends.adapter import (
    EquirectangularDatasetAdapter,
    RigidTransformSpec,
    apply_rigid_transform,
    compute_delta_depth,
    habitat_camera_transform_spec,
    materialize_distance_matrix,
    remap_semantic_ids,
    sanitize_depth_metres,
    transpose_equirectangular_grid,
)


def _fixture_transform() -> RigidTransformSpec:
    return RigidTransformSpec(
        name="habitat_fixture",
        matrix=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        handedness="right-handed",
        source_forward_axis="-Z",
    )


def _fixture_pose() -> RobotPose:
    return RobotPose(x=1.0, y=2.0, z=3.0, roll=0.0, pitch=0.0, yaw=0.25, timestamp=123.0)


def test_transpose_equirectangular_grid_converts_el_az_to_batched_az_el() -> None:
    raw = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)

    canonical = transpose_equirectangular_grid(raw)

    assert canonical.shape == (1, 3, 2)
    np.testing.assert_allclose(
        canonical,
        np.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]], dtype=np.float32),
    )


def test_sanitize_depth_metres_preserves_metric_hits_and_zeroes_invalid_cells() -> None:
    depth = np.array([[0.5, 5.0, 50.0, np.inf]], dtype=np.float32)

    sanitized, valid = sanitize_depth_metres(depth)

    np.testing.assert_allclose(
        sanitized,
        np.array([[0.5, 5.0, 50.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(valid, np.array([[True, True, True, False]], dtype=np.bool_))


def test_remap_semantic_ids_defaults_unknown_ids_to_zero() -> None:
    raw = np.array([[7, 8], [42, 7]], dtype=np.int32)

    remapped = remap_semantic_ids(raw, {7: 1, 8: 2}, unknown_id=0)

    np.testing.assert_array_equal(remapped, np.array([[1, 2], [0, 1]], dtype=np.int32))


def test_compute_delta_depth_zeros_first_frame_and_subtracts_afterward() -> None:
    current = np.array([[[0.4, 0.6]]], dtype=np.float32)
    previous = np.array([[[0.1, 0.2]]], dtype=np.float32)

    first_delta = compute_delta_depth(current, None)
    second_delta = compute_delta_depth(current, previous)

    np.testing.assert_array_equal(first_delta, np.zeros_like(current, dtype=np.float32))
    np.testing.assert_allclose(second_delta, np.array([[[0.3, 0.4]]], dtype=np.float32))


def test_apply_rigid_transform_uses_explicit_homogeneous_matrix() -> None:
    transform = RigidTransformSpec(
        name="fixture_yaw_plus_translation",
        matrix=np.array(
            [
                [0.0, -1.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 1.0, 5.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        handedness="right-handed",
        source_forward_axis="+X",
    )
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    transformed = apply_rigid_transform(points, transform)

    np.testing.assert_allclose(transformed, np.array([[8.0, -1.0, 8.0]], dtype=np.float32))


def test_apply_rigid_transform_rejects_non_homogeneous_matrix() -> None:
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        RigidTransformSpec(
            name="bad_fixture",
            matrix=np.eye(3, dtype=np.float32),
            handedness="right-handed",
            source_forward_axis="+X",
        )


def test_rigid_transform_spec_rejects_invalid_forward_axis_metadata() -> None:
    with pytest.raises(ValueError, match="source_forward_axis"):
        RigidTransformSpec(
            name="bad_axis",
            matrix=np.eye(4, dtype=np.float32),
            handedness="right-handed",
            source_forward_axis="forward",
        )


def test_habitat_camera_transform_spec_matches_canonical_camera_axes() -> None:
    transform = habitat_camera_transform_spec()
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    transformed = apply_rigid_transform(points, transform)

    assert transform.handedness == "right-handed"
    assert transform.source_forward_axis == "-Z"
    np.testing.assert_allclose(transformed, points)


def test_equirectangular_dataset_adapter_adapts_raw_observation_dict() -> None:
    adapter = EquirectangularDatasetAdapter(
        azimuth_bins=3,
        elevation_bins=2,
        max_distance=10.0,
        semantic_remap={7: 1, 8: 2},
        transform_spec=_fixture_transform(),
    )

    first = adapter.adapt(
        {
            "equirect_depth": np.array([[1.0, 4.0, 50.0], [0.0, 5.0, np.inf]], dtype=np.float32),
            "equirect_semantic": np.array([[7, 8, 99], [8, 7, 7]], dtype=np.int32),
            "overhead": np.ones((8, 8, 3), dtype=np.float32),
        },
        step_id=1,
    )
    second = adapter.adapt(
        {
            "equirect_depth": np.array([[2.0, 6.0, 10.0], [1.0, 5.0, 8.0]], dtype=np.float32),
            "equirect_semantic": np.array([[7, 8, 8], [8, 7, 7]], dtype=np.int32),
        },
        step_id=2,
    )

    assert adapter.metadata.azimuth_bins == 3
    assert adapter.metadata.elevation_bins == 2
    assert adapter.metadata.semantic_classes == 3
    assert first["depth"].shape == (1, 3, 2)
    assert first["semantic"].dtype == np.int32
    assert first["valid_mask"].dtype == np.bool_
    np.testing.assert_array_equal(first["delta_depth"], np.zeros((1, 3, 2), dtype=np.float32))
    np.testing.assert_allclose(
        first["depth"],
        np.array([[[0.1, 0.0], [0.4, 0.5], [0.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        first["semantic"],
        np.array([[[1, 2], [2, 1], [0, 1]]], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        first["valid_mask"],
        np.array([[[True, False], [True, True], [False, False]]], dtype=np.bool_),
    )
    np.testing.assert_allclose(
        second["delta_depth"],
        np.array([[[0.1, 0.1], [0.2, 0.0], [1.0, 0.8]]], dtype=np.float32),
    )
    assert second["overhead"].shape == (256, 256, 3)
    assert np.count_nonzero(second["overhead"]) == 0


def test_equirectangular_dataset_adapter_reset_clears_delta_state() -> None:
    adapter = EquirectangularDatasetAdapter(
        azimuth_bins=2,
        elevation_bins=2,
        max_distance=5.0,
        semantic_remap={1: 1},
        transform_spec=_fixture_transform(),
    )
    raw_obs = {
        "equirect_depth": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "equirect_semantic": np.array([[1, 1], [1, 1]], dtype=np.int32),
    }

    adapter.adapt(raw_obs, step_id=1)
    adapter.reset()
    adapted = adapter.adapt(raw_obs, step_id=2)

    np.testing.assert_array_equal(adapted["delta_depth"], np.zeros((1, 2, 2), dtype=np.float32))


def test_equirectangular_dataset_adapter_rejects_non_contiguous_semantic_ids() -> None:
    with pytest.raises(ValueError, match="contiguous canonical ids"):
        EquirectangularDatasetAdapter(
            azimuth_bins=2,
            elevation_bins=2,
            max_distance=5.0,
            semantic_remap={7: 2},
            transform_spec=_fixture_transform(),
        )


def test_equirectangular_dataset_adapter_rejects_shape_mismatch_against_metadata() -> None:
    adapter = EquirectangularDatasetAdapter(
        azimuth_bins=4,
        elevation_bins=2,
        max_distance=5.0,
        semantic_remap={1: 1},
        transform_spec=_fixture_transform(),
    )

    with pytest.raises(ValueError, match="canonical shape"):
        adapter.adapt(
            {
                "equirect_depth": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
                "equirect_semantic": np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int32),
            },
            step_id=1,
        )


def test_materialize_distance_matrix_builds_public_contract_from_canonical_arrays() -> None:
    observation = materialize_distance_matrix(
        episode_id=7,
        env_id=3,
        depth=np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32),
        delta_depth=np.array([[[0.0, 0.1], [0.0, -0.1]]], dtype=np.float32),
        semantic=np.array([[[1, 2], [2, 1]]], dtype=np.int32),
        valid_mask=np.array([[[True, False], [True, True]]], dtype=np.bool_),
        overhead=np.ones((4, 4, 3), dtype=np.float32),
        robot_pose=_fixture_pose(),
        step_id=11,
        timestamp=456.0,
    )

    assert observation.episode_id == 7
    np.testing.assert_array_equal(observation.env_ids, np.array([3], dtype=np.int32))
    assert observation.matrix_shape == (2, 2)
    assert observation.robot_pose == _fixture_pose()
    assert observation.step_id == 11
    assert observation.timestamp == 456.0
    np.testing.assert_allclose(
        observation.depth, np.array([[[0.1, 0.2], [0.3, 0.4]]], dtype=np.float32)
    )


def test_materialize_distance_matrix_rejects_noncanonical_depth_shape() -> None:
    with pytest.raises(ValueError, match=r"canonical shape \(1, Az, El\)"):
        materialize_distance_matrix(
            episode_id=1,
            env_id=0,
            depth=np.array([[0.1, 0.2]], dtype=np.float32),
            delta_depth=np.array([[[0.0, 0.0]]], dtype=np.float32),
            semantic=np.array([[[1, 1]]], dtype=np.int32),
            valid_mask=np.array([[[True, True]]], dtype=np.bool_),
            robot_pose=_fixture_pose(),
            step_id=1,
        )


def test_equirectangular_dataset_adapter_materializes_distance_matrix_directly() -> None:
    adapter = EquirectangularDatasetAdapter(
        azimuth_bins=3,
        elevation_bins=2,
        max_distance=10.0,
        semantic_remap={7: 1, 8: 2},
        transform_spec=_fixture_transform(),
    )

    observation = adapter.adapt_distance_matrix(
        {
            "equirect_depth": np.array([[1.0, 4.0, 50.0], [0.0, 5.0, np.inf]], dtype=np.float32),
            "equirect_semantic": np.array([[7, 8, 99], [8, 7, 7]], dtype=np.int32),
        },
        episode_id=9,
        env_id=2,
        robot_pose=_fixture_pose(),
        step_id=4,
        timestamp=789.0,
    )

    assert observation.episode_id == 9
    np.testing.assert_array_equal(observation.env_ids, np.array([2], dtype=np.int32))
    assert observation.matrix_shape == (3, 2)
    assert observation.timestamp == 789.0
    np.testing.assert_array_equal(
        observation.semantic,
        np.array([[[1, 2], [2, 1], [0, 1]]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        observation.depth,
        np.array([[[0.1, 0.0], [0.4, 0.5], [0.0, 0.0]]], dtype=np.float32),
    )
