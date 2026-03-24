"""Unit tests for dashboard status-line telemetry formatting."""

from __future__ import annotations

import numpy as np

from navi_auditor.dashboard.status_line import build_status_metrics_line
from navi_auditor.stream_engine import StreamState
from navi_contracts import TelemetryEvent


def test_build_status_metrics_line_waiting_state() -> None:
    text = build_status_metrics_line(None)
    assert "SPS=--" in text
    assert "Env=--" in text
    assert "Step=--" in text


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

    text = build_status_metrics_line(state, now=100.2)

    assert "SPS=22.5" in text
    assert "Env=--" in text
    assert "EMA=-0.734" in text
    assert "Ep=1" in text
    assert "Step=6400" in text
    assert "Opt=145ms" in text
    assert "ZW=3%" in text
    # Stall < 1s should not appear
    assert "Stall=" not in text


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

    assert "SPS=--" in text
    assert "Env=64.2" in text
    assert "Opt=14ms" in text
    assert "Step=320" in text


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

    text = build_status_metrics_line(state, now=75.2, fallback_state=fallback)

    assert "SPS=58.4" in text
    assert "Env=--" in text
    assert "EMA=1.234" in text
    assert "Ep=1" in text
    assert "Step=912" in text
    assert "Opt=91ms" in text
    assert "ZW=2%" in text
