from __future__ import annotations

import numpy as np

from navi_contracts import RobotPose
from navi_environment.backends.adapter import EquirectangularDatasetAdapter, RigidTransformSpec
from navi_environment.integration.dataset_fixtures import (
    DatasetFixtureFrame,
    adapt_fixture_frame,
    adapt_fixture_sequence,
)


def _fixture_transform() -> RigidTransformSpec:
    return RigidTransformSpec(
        name="fixture_transform",
        matrix=np.eye(4, dtype=np.float32),
        handedness="right-handed",
        source_forward_axis="-Z",
    )


def _fixture_pose(step_id: int) -> RobotPose:
    return RobotPose(
        x=float(step_id),
        y=2.0,
        z=3.0,
        roll=0.0,
        pitch=0.0,
        yaw=0.25,
        timestamp=100.0 + float(step_id),
    )


def _adapter() -> EquirectangularDatasetAdapter:
    return EquirectangularDatasetAdapter(
        azimuth_bins=3,
        elevation_bins=2,
        max_distance=10.0,
        semantic_remap={7: 1, 8: 2},
        transform_spec=_fixture_transform(),
    )


def _frame(*, episode_id: int, step_id: int, depth_value: float) -> DatasetFixtureFrame:
    return DatasetFixtureFrame(
        raw_obs={
            "equirect_depth": np.array(
                [
                    [depth_value, depth_value + 1.0, depth_value + 2.0],
                    [depth_value, depth_value, depth_value],
                ],
                dtype=np.float32,
            ),
            "equirect_semantic": np.array([[7, 8, 99], [8, 7, 7]], dtype=np.int32),
        },
        episode_id=episode_id,
        env_id=0,
        robot_pose=_fixture_pose(step_id),
        step_id=step_id,
        timestamp=200.0 + float(step_id),
    )


def test_adapt_fixture_frame_materializes_public_observation() -> None:
    observation = adapt_fixture_frame(_adapter(), _frame(episode_id=4, step_id=1, depth_value=1.0))

    assert observation.episode_id == 4
    assert observation.matrix_shape == (3, 2)
    assert observation.step_id == 1
    assert observation.timestamp == 201.0
    np.testing.assert_array_equal(observation.env_ids, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(
        observation.semantic,
        np.array([[[1, 2], [2, 1], [0, 1]]], dtype=np.int32),
    )


def test_adapt_fixture_sequence_preserves_delta_within_one_episode() -> None:
    observations = adapt_fixture_sequence(
        _adapter(),
        (
            _frame(episode_id=7, step_id=1, depth_value=1.0),
            _frame(episode_id=7, step_id=2, depth_value=2.0),
        ),
    )

    np.testing.assert_array_equal(
        observations[0].delta_depth, np.zeros((1, 3, 2), dtype=np.float32)
    )
    np.testing.assert_allclose(
        observations[1].delta_depth,
        np.array([[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]], dtype=np.float32),
    )


def test_adapt_fixture_sequence_resets_delta_on_episode_change() -> None:
    observations = adapt_fixture_sequence(
        _adapter(),
        (
            _frame(episode_id=8, step_id=1, depth_value=1.0),
            _frame(episode_id=9, step_id=2, depth_value=2.0),
        ),
    )

    np.testing.assert_array_equal(
        observations[0].delta_depth, np.zeros((1, 3, 2), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        observations[1].delta_depth, np.zeros((1, 3, 2), dtype=np.float32)
    )
