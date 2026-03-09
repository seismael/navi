"""Tests for the StreamEngine multi-stream ZMQ ingestion."""

from __future__ import annotations

import numpy as np
import pytest

from navi_auditor.stream_engine import StreamEngine, StreamState
from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    DistanceMatrix,
    RobotPose,
    TelemetryEvent,
)


class TestStreamState:
    """Tests for StreamState defaults."""

    def test_default_state_has_empty_buffers(self) -> None:
        state = StreamState()
        assert state.latest_matrix is None
        assert state.latest_action is None
        assert len(state.reward_history) == 0
        assert len(state.collision_history) == 0
        assert len(state.telemetry_buffer) == 0

    def test_ring_buffer_maxlen(self) -> None:
        state = StreamState()
        for i in range(3000):
            state.reward_history.append(float(i))
        assert len(state.reward_history) == 2000


class TestStreamEngine:
    """Tests for StreamEngine construction."""

    def test_creation_with_matrix_only(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559")
        assert engine.actor_states == {}
        assert not engine.has_step_socket
        engine.close()

    def test_creation_with_actor_sub(self) -> None:
        engine = StreamEngine(
            matrix_sub="tcp://localhost:15559",
            actor_sub="tcp://localhost:15557",
        )
        assert engine.actor_states == {}
        engine.close()

    def test_creation_with_actor_only_passive_mode(self) -> None:
        engine = StreamEngine(actor_sub="tcp://localhost:15557")
        assert engine.actor_states == {}
        assert not engine.has_step_socket
        engine.close()

    def test_creation_with_step_endpoint(self) -> None:
        engine = StreamEngine(
            matrix_sub="tcp://localhost:15559",
            step_endpoint="tcp://localhost:15560",
        )
        assert engine.has_step_socket
        engine.close()

    def test_poll_returns_without_data(self) -> None:
        """poll() should not block or crash when no data is available."""
        engine = StreamEngine(matrix_sub="tcp://localhost:15559")
        engine.poll()  # Should return immediately
        assert engine.actor_states == {}
        engine.close()

    def test_selected_actor_filters_distance_action_and_telemetry(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559", selected_actor_id=0)

        dm_actor0 = DistanceMatrix(
            episode_id=11,
            env_ids=np.array([0], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=1,
            timestamp=1.0,
        )
        dm_actor1 = DistanceMatrix(
            episode_id=22,
            env_ids=np.array([1], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=1.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=1,
            timestamp=1.0,
        )

        action_actor1 = Action(
            env_ids=np.array([1], dtype=np.int32),
            linear_velocity=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
            policy_id="test",
            step_id=1,
            timestamp=1.0,
        )
        telem_actor1 = TelemetryEvent(
            event_type="actor.training.ppo.step",
            episode_id=22,
            env_id=1,
            step_id=1,
            payload=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_DISTANCE_MATRIX, dm_actor1)
        engine._dispatch(TOPIC_ACTION, action_actor1)
        engine._dispatch(TOPIC_TELEMETRY_EVENT, telem_actor1)
        assert 1 not in engine.actor_states

        engine._dispatch(TOPIC_DISTANCE_MATRIX, dm_actor0)
        assert 0 in engine.actor_states
        assert engine.actor_states[0].latest_matrix is not None
        assert engine.actor_states[0].latest_matrix.episode_id == 11
        engine.close()

    def test_selected_actor_none_ingests_all_actors(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559", selected_actor_id=None)

        dm_actor0 = DistanceMatrix(
            episode_id=1,
            env_ids=np.array([0], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=1,
            timestamp=1.0,
        )
        dm_actor1 = DistanceMatrix(
            episode_id=2,
            env_ids=np.array([1], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=1.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=1,
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_DISTANCE_MATRIX, dm_actor0)
        engine._dispatch(TOPIC_DISTANCE_MATRIX, dm_actor1)

        assert 0 in engine.actor_states
        assert 1 in engine.actor_states
        assert engine.actor_states[0].latest_matrix is not None
        assert engine.actor_states[1].latest_matrix is not None
        engine.close()

    def test_actor_only_engine_accepts_distance_matrix_dispatch(self) -> None:
        engine = StreamEngine(actor_sub="tcp://localhost:15557", selected_actor_id=0)

        dm_actor0 = DistanceMatrix(
            episode_id=1,
            env_ids=np.array([0], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=1,
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_DISTANCE_MATRIX, dm_actor0)

        assert 0 in engine.actor_states
        assert engine.actor_states[0].latest_matrix is not None
        engine.close()

    def test_ppo_perf_telemetry_updates_performance_histories(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559", selected_actor_id=0)

        perf_event = TelemetryEvent(
            event_type="actor.training.ppo.perf",
            episode_id=0,
            env_id=0,
            step_id=123,
            payload=np.array([18.5, 21.0, 170.0, 1.2, 3.1, 220.0, 0.04, 145.0], dtype=np.float32),
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_TELEMETRY_EVENT, perf_event)
        state = engine.actor_states[0]

        assert len(state.perf_sps_history) == 1
        assert state.perf_sps_history[-1] == 18.5
        assert state.perf_opt_ms_history[-1] == 145.0
        assert state.perf_zero_wait_history[-1] == pytest.approx(0.04)
        engine.close()

    def test_action_and_telemetry_update_stream_freshness(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559", selected_actor_id=0)

        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
            policy_id="test",
            step_id=1,
            timestamp=1.0,
        )
        telemetry = TelemetryEvent(
            event_type="actor.training.ppo.perf",
            episode_id=0,
            env_id=0,
            step_id=1,
            payload=np.array([18.5, 21.0, 170.0, 1.2, 3.1, 220.0, 0.04, 145.0], dtype=np.float32),
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_ACTION, action)
        action_rx = engine.actor_states[0].last_rx_time
        assert action_rx > 0.0

        engine._dispatch(TOPIC_TELEMETRY_EVENT, telemetry)
        telemetry_rx = engine.actor_states[0].last_rx_time
        assert telemetry_rx >= action_rx
        engine.close()

    def test_environment_sdfdag_perf_updates_environment_histories(self) -> None:
        engine = StreamEngine(matrix_sub="tcp://localhost:15559", selected_actor_id=0)

        perf_event = TelemetryEvent(
            event_type="environment.sdfdag.perf",
            episode_id=0,
            env_id=0,
            step_id=200,
            payload=np.array([61.5, 12.0, 11.2, 10.8, 2.7, 200.0, 800.0], dtype=np.float32),
            timestamp=1.0,
        )

        engine._dispatch(TOPIC_TELEMETRY_EVENT, perf_event)
        state = engine.actor_states[0]

        assert state.env_perf_sps_history[-1] == pytest.approx(61.5)
        assert state.env_perf_batch_ms_history[-1] == pytest.approx(10.8)
        assert state.env_perf_actor_ms_history[-1] == pytest.approx(2.7)
        engine.close()
