"""Tests for Ghost-Matrix v2 wire-format models and serialization."""

from __future__ import annotations

import numpy as np

from navi_contracts import (
    Action,
    DistanceMatrix,
    RobotPose,
    StepRequest,
    StepResult,
    TelemetryEvent,
    deserialize,
    serialize,
)


class TestRobotPose:
    """Tests for the RobotPose model."""

    def test_creation(self) -> None:
        pose = RobotPose(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3, timestamp=100.0)
        assert pose.x == 1.0
        assert pose.y == 2.0
        assert pose.z == 3.0
        assert pose.timestamp == 100.0

    def test_round_trip_serialization(self) -> None:
        pose = RobotPose(x=1.5, y=2.5, z=3.5, roll=0.1, pitch=0.2, yaw=0.3, timestamp=42.0)
        data = serialize(pose)
        restored = deserialize(data)
        assert isinstance(restored, RobotPose)
        assert restored == pose


class TestDistanceMatrix:
    """Tests for canonical Distance Matrix v2 observations."""

    def test_creation(self) -> None:
        rng = np.random.default_rng(42)
        pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
        update = DistanceMatrix(
            episode_id=7,
            env_ids=np.array([0, 1], dtype=np.int32),
            matrix_shape=(64, 32),
            depth=rng.random((2, 64, 32), dtype=np.float32),
            delta_depth=rng.standard_normal((2, 64, 32)).astype(np.float32),
            semantic=rng.integers(0, 8, size=(2, 64, 32), dtype=np.int32),
            valid_mask=rng.integers(0, 2, size=(2, 64, 32), dtype=np.int32).astype(np.bool_),
            overhead=np.zeros((256, 256, 3), dtype=np.uint8),
            robot_pose=pose,
            step_id=101,
            timestamp=10.0,
        )
        assert update.episode_id == 7
        assert update.matrix_shape == (64, 32)
        assert update.depth.shape == (2, 64, 32)

    def test_round_trip_serialization(self) -> None:
        rng = np.random.default_rng(0)
        pose = RobotPose(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3, timestamp=1.0)
        update = DistanceMatrix(
            episode_id=12,
            env_ids=np.array([0, 5, 9], dtype=np.int32),
            matrix_shape=(16, 8),
            depth=rng.random((3, 16, 8), dtype=np.float32),
            delta_depth=rng.standard_normal((3, 16, 8)).astype(np.float32),
            semantic=rng.integers(0, 32, size=(3, 16, 8), dtype=np.int32),
            valid_mask=rng.integers(0, 2, size=(3, 16, 8), dtype=np.int32).astype(np.bool_),
            overhead=np.zeros((256, 256, 3), dtype=np.uint8),
            robot_pose=pose,
            step_id=55,
            timestamp=99.0,
        )
        data = serialize(update)
        restored = deserialize(data)
        assert isinstance(restored, DistanceMatrix)
        assert restored.episode_id == update.episode_id
        assert restored.matrix_shape == update.matrix_shape
        assert restored.step_id == update.step_id
        np.testing.assert_array_equal(restored.env_ids, update.env_ids)
        np.testing.assert_array_equal(restored.depth, update.depth)
        np.testing.assert_array_equal(restored.delta_depth, update.delta_depth)
        np.testing.assert_array_equal(restored.semantic, update.semantic)
        np.testing.assert_array_equal(restored.valid_mask, update.valid_mask)


class TestAction:
    """Tests for the Action v2 model."""

    def test_round_trip_serialization(self) -> None:
        action = Action(
            env_ids=np.array([0, 1], dtype=np.int32),
            linear_velocity=np.array([[0.5, -0.3, 0.1], [0.2, 0.0, -0.4]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.1, -0.2], [0.0, 0.0, 0.8]], dtype=np.float32),
            policy_id="brain-default",
            step_id=314,
            timestamp=3.14,
        )
        data = serialize(action)
        restored = deserialize(data)
        assert isinstance(restored, Action)
        assert restored.policy_id == "brain-default"
        assert restored.step_id == 314
        np.testing.assert_array_equal(restored.env_ids, action.env_ids)
        np.testing.assert_array_almost_equal(restored.linear_velocity, action.linear_velocity)
        np.testing.assert_array_almost_equal(restored.angular_velocity, action.angular_velocity)


class TestStepRequest:
    """Tests for StepRequest v2."""

    def test_round_trip_serialization(self) -> None:
        action = Action(
            env_ids=np.array([2], dtype=np.int32),
            linear_velocity=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.5]], dtype=np.float32),
            policy_id="brain-default",
            step_id=42,
            timestamp=1.0,
        )
        req = StepRequest(action=action, step_id=42, timestamp=2.0)
        data = serialize(req)
        restored = deserialize(data)
        assert isinstance(restored, StepRequest)
        assert restored.step_id == 42
        assert restored.timestamp == 2.0
        assert restored.action.policy_id == "brain-default"


class TestStepResult:
    """Tests for StepResult v2."""

    def test_round_trip_serialization(self) -> None:
        result = StepResult(
            step_id=99,
            env_id=0,
            done=True,
            truncated=False,
            reward=-1.0,
            episode_return=12.5,
            timestamp=5.0,
        )
        data = serialize(result)
        restored = deserialize(data)
        assert isinstance(restored, StepResult)
        assert restored.step_id == 99
        assert restored.done is True
        assert restored.truncated is False
        assert restored.reward == -1.0
        assert restored.episode_return == 12.5


class TestTelemetryEvent:
    """Tests for TelemetryEvent v2."""

    def test_round_trip_serialization(self) -> None:
        event = TelemetryEvent(
            event_type="eval_best_episode",
            episode_id=4,
            env_id=3,
            step_id=888,
            payload=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            timestamp=9.9,
        )
        data = serialize(event)
        restored = deserialize(data)
        assert isinstance(restored, TelemetryEvent)
        assert restored.event_type == event.event_type
        assert restored.episode_id == event.episode_id
        assert restored.env_id == event.env_id
        np.testing.assert_array_equal(restored.payload, event.payload)



