"""Unit tests for dashboard status-line telemetry formatting."""

from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.status_line import build_status_metrics_line
from navi_auditor.stream_engine import StreamState
from navi_contracts import TelemetryEvent


def test_build_status_metrics_line_waiting_state() -> None:
    text = build_status_metrics_line(None)
    assert "stall=--" in text
    assert "rollout_sps=--" in text
    assert "env_sps=--" in text
    assert "per_actor_sps=--" in text
    assert "step=--" in text


def test_build_status_metrics_line_training_values() -> None:
    state = StreamState()
    state.last_rx_time = 100.0
    state.perf_sps_history.append(22.5)
    state.ppo_reward_ema_history.append(-0.734)
    state.episode_return_history.append(1.2)
    state.perf_opt_ms_history.append(145.0)
    state.perf_zero_wait_history.append(0.03)
    state.telemetry_buffer.append(
        TelemetryEvent(
            event_type="actor.training.ppo.perf",
            episode_id=0,
            env_id=0,
            step_id=6400,
            payload=np.zeros((8,), dtype=np.float32),
            timestamp=100.2,
        ),
    )

    text = build_status_metrics_line(state, now=100.2, actor_count=4)

    assert "stall=200ms" in text
    assert "rollout_sps=22.5" in text
    assert "env_sps=--" in text
    assert "per_actor_sps=5.6" in text
    assert "ema=-0.734" in text
    assert "ep=1" in text
    assert "step=6400" in text
    assert "opt=145ms" in text
    assert "zw=3.0%" in text


def test_build_status_metrics_line_falls_back_to_environment_perf() -> None:
    state = StreamState()
    state.last_rx_time = 50.0
    state.env_perf_sps_history.append(64.2)
    state.env_perf_batch_ms_history.append(14.0)
    state.telemetry_buffer.append(
        TelemetryEvent(
            event_type="environment.sdfdag.perf",
            episode_id=0,
            env_id=0,
            step_id=320,
            payload=np.zeros((7,), dtype=np.float32),
            timestamp=50.1,
        ),
    )

    text = build_status_metrics_line(state, now=50.2)

    assert "rollout_sps=--" in text
    assert "env_sps=64.2" in text
    assert "per_actor_sps=--" in text
    assert "opt=14ms" in text
    assert "step=320" in text


def test_build_status_metrics_line_uses_shared_fallback_metrics() -> None:
    state = StreamState()
    state.last_rx_time = 75.0
    state.latest_matrix = type("Matrix", (), {"step_id": 912})()

    fallback = StreamState()
    fallback.perf_sps_history.append(58.4)
    fallback.ppo_reward_ema_history.append(1.234)
    fallback.perf_opt_ms_history.append(91.0)
    fallback.perf_zero_wait_history.append(0.02)
    fallback.episode_return_history.append(3.0)

    text = build_status_metrics_line(state, now=75.2, fallback_state=fallback, actor_count=8)

    assert "stall=200ms" in text
    assert "rollout_sps=58.4" in text
    assert "env_sps=--" in text
    assert "per_actor_sps=7.3" in text
    assert "ema=1.234" in text
    assert "ep=1" in text
    assert "step=912" in text
    assert "opt=91ms" in text
    assert "zw=2.0%" in text
