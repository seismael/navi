"""Tests for the StreamEngine multi-stream ZMQ ingestion."""

from __future__ import annotations

from navi_auditor.stream_engine import StreamEngine, StreamState


class TestStreamState:
    """Tests for StreamState defaults."""

    def test_default_state_has_empty_buffers(self) -> None:
        state = StreamState()
        assert state.latest_matrix is None
        assert state.latest_action is None
        assert len(state.reward_history) == 0
        assert len(state.collision_history) == 0
        assert len(state.eval_reward_history) == 0
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
        assert engine.state.latest_matrix is None
        assert not engine.has_step_socket
        engine.close()

    def test_creation_with_actor_sub(self) -> None:
        engine = StreamEngine(
            matrix_sub="tcp://localhost:15559",
            actor_sub="tcp://localhost:15557",
        )
        assert engine.state.latest_action is None
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
        assert engine.state.latest_matrix is None
        engine.close()
