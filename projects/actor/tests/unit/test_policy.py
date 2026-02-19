"""Tests for ShallowPolicy."""

from __future__ import annotations

import numpy as np

from navi_actor.policy import ShallowPolicy
from navi_contracts import DistanceMatrix, RobotPose


def _obs() -> DistanceMatrix:
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    depth = np.full((1, 8, 4), 0.8, dtype=np.float32)
    valid = np.ones((1, 8, 4), dtype=np.bool_)
    semantic = np.zeros((1, 8, 4), dtype=np.int32)
    return DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(8, 4),
        depth=depth,
        delta_depth=np.zeros_like(depth),
        semantic=semantic,
        valid_mask=valid,
        overhead=np.zeros((256, 256, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=1,
        timestamp=1.0,
    )


def test_policy_act_returns_action() -> None:
    policy = ShallowPolicy(policy_id="test", gain=0.5)
    action = policy.act(_obs(), step_id=4)
    assert action.policy_id == "test"
    assert action.step_id == 4
    assert action.linear_velocity.shape == (1, 3)
    assert action.angular_velocity.shape == (1, 3)
