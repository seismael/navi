"""Additional serialization regression tests for v2 models."""

from __future__ import annotations

import numpy as np

from navi_contracts import DistanceMatrix, RobotPose, deserialize, serialize


def test_matrix_shape_round_trip_is_tuple() -> None:
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    obs = DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(8, 4),
        depth=np.zeros((1, 8, 4), dtype=np.float32),
        delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
        semantic=np.zeros((1, 8, 4), dtype=np.int32),
        valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
        overhead=np.zeros((256, 256, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=1,
        timestamp=0.1,
    )

    restored = deserialize(serialize(obs))
    assert isinstance(restored, DistanceMatrix)
    assert isinstance(restored.matrix_shape, tuple)
    assert restored.matrix_shape == (8, 4)
