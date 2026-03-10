"""Unit tests for tensor-friendly MJX kinematic stepping."""

from __future__ import annotations

import math

import numpy as np
import torch

from navi_contracts import Action, RobotPose
from navi_environment.mjx_env import MjxEnvironment


def test_step_pose_commands_matches_numpy_and_tensor_prev_depth() -> None:
    env = MjxEnvironment(dt=0.1)
    env.set_smoothing(0.0)
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    linear = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    angular = np.zeros(3, dtype=np.float32)
    prev_depth_np = np.full((8, 4), 0.5, dtype=np.float32)
    prev_depth_tensor = torch.full((8, 4), 0.5, dtype=torch.float32)

    pose_from_numpy = env.step_pose_commands(
        pose,
        linear,
        angular,
        1.0,
        prev_depth=prev_depth_np,
        max_distance=30.0,
    )
    env.reset_velocity()
    pose_from_tensor = env.step_pose_commands(
        pose,
        linear,
        angular,
        1.0,
        prev_depth=prev_depth_tensor,
        max_distance=30.0,
    )

    assert math.isclose(pose_from_numpy.x, pose_from_tensor.x)
    assert math.isclose(pose_from_numpy.y, pose_from_tensor.y)
    assert math.isclose(pose_from_numpy.z, pose_from_tensor.z)
    assert math.isclose(pose_from_numpy.yaw, pose_from_tensor.yaw)


def test_step_pose_uses_tensor_prev_depth_for_speed_factor() -> None:
    env = MjxEnvironment(dt=0.1)
    env.set_smoothing(0.0)
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    action = Action(
        env_ids=np.array([0], dtype=np.int32),
        linear_velocity=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        angular_velocity=np.zeros((1, 3), dtype=np.float32),
        policy_id="test",
        step_id=0,
        timestamp=0.0,
    )
    prev_depth_tensor = torch.full((8, 4), 0.02, dtype=torch.float32)

    next_pose = env.step_pose(
        pose,
        action,
        1.0,
        prev_depth=prev_depth_tensor,
        max_distance=30.0,
    )

    assert next_pose.x > 0.0
    assert next_pose.x < 1.0
