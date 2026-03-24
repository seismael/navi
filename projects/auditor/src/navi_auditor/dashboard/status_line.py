"""Formatting helpers for compact dashboard status-line telemetry."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from navi_auditor.stream_engine import StreamState


def _last_value(values: Sequence[float] | None) -> float | None:
    if values is None:
        return None
    if len(values) == 0:
        return None
    return float(values[-1])


def _fmt_number(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "--"
    return f"{value:.{decimals}f}"


def _fmt_stall_seconds(value: float | None) -> str:
    if value is None:
        return "--"
    if value < 1.0:
        return f"{value * 1000.0:.0f}ms"
    return f"{value:.1f}s"


def build_status_metrics_line(
    state: StreamState | None,
    *,
    now: float | None = None,
    fallback_state: StreamState | None = None,
) -> str:
    """Build a compact one-line telemetry summary for the dashboard top bar."""
    if state is None:
        return "SPS=-- | Env=-- | EMA=-- | Ep=0 | Step=-- | Opt=--"

    now_ts = time.time() if now is None else now
    stall_s = None
    if state.last_rx_time > 0.0:
        stall_s = max(0.0, now_ts - state.last_rx_time)

    sps = _last_value(state.perf_sps_history)
    if sps is None and fallback_state is not None:
        sps = _last_value(fallback_state.perf_sps_history)

    env_sps = _last_value(state.env_perf_sps_history)
    if env_sps is None and fallback_state is not None:
        env_sps = _last_value(fallback_state.env_perf_sps_history)

    reward_ema = _last_value(state.ppo_reward_ema_history)
    if reward_ema is None:
        reward_ema = _last_value(state.reward_history)
    if reward_ema is None and fallback_state is not None:
        reward_ema = _last_value(fallback_state.ppo_reward_ema_history)
        if reward_ema is None:
            reward_ema = _last_value(fallback_state.reward_history)

    opt_ms = _last_value(state.perf_opt_ms_history)
    if opt_ms is None:
        opt_ms = _last_value(state.env_perf_batch_ms_history)
    if opt_ms is None and fallback_state is not None:
        opt_ms = _last_value(fallback_state.perf_opt_ms_history)
        if opt_ms is None:
            opt_ms = _last_value(fallback_state.env_perf_batch_ms_history)

    zero_wait = _last_value(state.perf_zero_wait_history)
    if zero_wait is None and fallback_state is not None:
        zero_wait = _last_value(fallback_state.perf_zero_wait_history)

    episodes = len(state.episode_return_history)
    if episodes == 0 and fallback_state is not None:
        episodes = len(fallback_state.episode_return_history)

    step_id: int | None = None
    if len(state.telemetry_buffer) > 0:
        step_id = int(state.telemetry_buffer[-1].step_id)
    elif state.latest_matrix is not None:
        step_id = int(state.latest_matrix.step_id)

    parts: list[str] = []
    if stall_s is not None and stall_s > 1.0:
        parts.append(f"Stall={_fmt_stall_seconds(stall_s)}")
    parts.append(f"SPS={_fmt_number(sps, 1)}")
    parts.append(f"Env={_fmt_number(env_sps, 1)}")
    parts.append(f"EMA={_fmt_number(reward_ema, 3)}")
    parts.append(f"Ep={episodes}")
    parts.append(f"Step={step_id if step_id is not None else '--'}")
    parts.append(f"Opt={_fmt_number(opt_ms, 0)}ms")
    if zero_wait is not None:
        parts.append(f"ZW={zero_wait * 100.0:.0f}%")
    return " | ".join(parts)
