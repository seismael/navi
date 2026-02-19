"""Tests for learned spherical policy checkpointing and inference."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

from navi_actor.policy import LearnedSphericalPolicy, PolicyCheckpoint
from navi_contracts import DistanceMatrix, RobotPose


def _obs() -> DistanceMatrix:
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    depth = np.full((1, 32, 16), 0.7, dtype=np.float32)
    valid = np.ones((1, 32, 16), dtype=np.bool_)
    semantic = np.zeros((1, 32, 16), dtype=np.int32)
    return DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(32, 16),
        depth=depth,
        delta_depth=np.zeros_like(depth),
        semantic=semantic,
        valid_mask=valid,
        overhead=np.zeros((64, 64, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=3,
        timestamp=1.0,
    )


def _checkpoint() -> PolicyCheckpoint:
    return PolicyCheckpoint(
        w_forward=np.full((13,), 0.2, dtype=np.float32),
        b_forward=0.1,
        w_yaw=np.full((13,), -0.1, dtype=np.float32),
        b_yaw=0.0,
        max_forward=0.8,
        max_yaw=1.2,
    )


def test_learned_policy_act_shape() -> None:
    policy = LearnedSphericalPolicy(_checkpoint())
    action = policy.act(_obs(), step_id=7)
    assert action.step_id == 7
    assert action.linear_velocity.shape == (1, 3)
    assert action.angular_velocity.shape == (1, 3)
    assert float(action.linear_velocity[0, 0]) >= 0.0


def test_checkpoint_roundtrip() -> None:
    tmp_root = Path(".tmp_test_outputs") / f"policy_{uuid4().hex}"
    tmp_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = tmp_root / "policy.npz"
    expected = _checkpoint()
    LearnedSphericalPolicy.save_checkpoint(str(ckpt_path), expected)
    loaded = LearnedSphericalPolicy.load_checkpoint(str(ckpt_path))

    assert np.allclose(expected.w_forward, loaded.w_forward)
    assert np.allclose(expected.w_yaw, loaded.w_yaw)
    assert np.isclose(expected.max_forward, loaded.max_forward)
    assert np.isclose(expected.max_yaw, loaded.max_yaw)
