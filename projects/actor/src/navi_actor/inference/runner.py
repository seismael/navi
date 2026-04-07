"""In-process inference runner for the canonical sdfdag backend.

Mirrors the ``PpoTrainer`` architecture (direct ``SdfDagBackend``, tensor-native
throughout, async ZMQ telemetry worker) but without PPO/rollout/reward machinery.
Pure policy evaluation with optional deterministic action mode.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import zmq

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import ActorConfig
from navi_actor.spherical_features import extract_spherical_features
from navi_contracts import (
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    DistanceMatrix,
    JsonlMetricsSink,
    TelemetryEvent,
    build_phase_metrics_payload,
    get_or_create_run_context,
    serialize,
)

if TYPE_CHECKING:
    from navi_environment.backends.sdfdag_backend import SdfDagPerfSnapshot, SdfDagTensorStepBatch

_LOGGER = logging.getLogger(__name__)

# Telemetry decimation: publish features/action every Nth step
_TELEMETRY_EVERY_N: int = 10

_PERF_TELEMETRY_FIELDS: tuple[str, ...] = (
    "sps",
    "forward_pass_ms",
    "batch_step_ms",
    "tick_total_ms",
    "n_actors",
)


class _CanonicalRuntime:
    """Minimal runtime surface required by the inference runner."""

    def reset_tensor(
        self,
        episode_id: int,
        *,
        actor_id: int = 0,
        materialize: bool = False,
    ) -> tuple[torch.Tensor, DistanceMatrix | None]: ...

    def perf_snapshot(self) -> SdfDagPerfSnapshot: ...

    def batch_step_tensor_actions(
        self,
        action_tensor: torch.Tensor,
        step_id: int,
        *,
        actor_indices: torch.Tensor | None = None,
        scratch_slot: int = 0,
        publish_actor_ids: tuple[int, ...] = (),
        materialize_results: bool = False,
    ) -> tuple[SdfDagTensorStepBatch, tuple[Any, ...]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class InferenceMetrics:
    """Summary metrics from an inference run."""

    total_steps: int
    total_episodes: int
    reward_mean: float
    reward_std: float
    episode_return_mean: float
    episode_return_std: float
    episode_length_mean: float
    episode_length_std: float
    sps_mean: float
    forward_pass_ms_mean: float
    batch_step_ms_mean: float
    wall_clock_seconds: float


def _enqueue_telemetry(
    telemetry_queue: queue.Queue[tuple[str, bytes | Callable[[], bytes], str] | None] | None,
    *,
    topic: str,
    payload: bytes | Callable[[], bytes],
    label: str,
) -> None:
    """Best-effort async telemetry enqueue."""
    if telemetry_queue is None:
        return
    try:
        telemetry_queue.put_nowait((topic, payload, label))
    except queue.Full:
        _LOGGER.warning("Dropping %s publish: telemetry queue full", label)
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("Failed to enqueue %s: %s", label, exc)


class InferenceRunner:
    """Single-process inference runner with direct sdfdag backend.

    Evaluates a trained policy on the canonical SDF/DAG environment runtime,
    publishing observations and telemetry for the dashboard. No PPO, no reward
    shaping, no RND, no episodic memory — pure policy execution.
    """

    def __init__(
        self,
        config: ActorConfig,
        *,
        runtime: Any,
        gmdag_file: str = "",
        scene_pool: tuple[str, ...] = (),
        deterministic: bool = False,
    ) -> None:
        self._config = config
        self._runtime: Any = runtime
        self._gmdag_file = gmdag_file
        self._scene_pool = scene_pool
        self._deterministic = deterministic
        self._n_actors = config.n_actors

        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._run_context = get_or_create_run_context("actor-infer")
        self._metrics_sink: JsonlMetricsSink | None = None
        if config.emit_internal_stats:
            self._metrics_sink = JsonlMetricsSink(
                self._run_context.metrics_root / "actor_inference.jsonl",
                run_id=self._run_context.run_id,
                project_name="navi_actor_infer",
            )

        # CUDA validation
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for canonical inference runtime.",
            )
        self._device = torch.device("cuda")

        # Policy in eval mode
        self._policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            temporal_core=config.temporal_core,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)
        self._policy.eval()

        # Dashboard observation state
        self._last_dashboard_observation: DistanceMatrix | None = None
        self._last_dashboard_heartbeat_at: float = 0.0

        # Telemetry worker
        self._telemetry_queue: queue.Queue[
            tuple[str, bytes | Callable[[], bytes], str] | None
        ] = queue.Queue(maxsize=1024)
        self._telemetry_thread: threading.Thread | None = None

        _LOGGER.info(
            "InferenceRunner init: actors=%d deterministic=%s temporal=%s",
            self._n_actors,
            self._deterministic,
            config.temporal_core,
        )

    # ── Checkpoint loading ────────────────────────────────────────

    def load_checkpoint(self, path: str) -> None:
        """Load policy weights from a training checkpoint."""
        state = torch.load(path, weights_only=False, map_location="cpu")
        if "policy_state_dict" in state:
            state = state["policy_state_dict"]
        self._policy.load_state_dict(state)
        self._policy.eval()
        _LOGGER.info("Loaded inference checkpoint: %s", path)

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background telemetry worker."""
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_worker, daemon=True
        )
        self._telemetry_thread.start()
        _LOGGER.info(
            "Inference runner started: actors=%d pub=%s deterministic=%s",
            self._n_actors,
            self._config.pub_address,
            self._deterministic,
        )

    def stop(self) -> None:
        """Clean up telemetry and runtime resources."""
        if self._telemetry_thread is not None:
            self._telemetry_queue.put(None)
            self._telemetry_thread.join(timeout=2.0)
            self._telemetry_thread = None

        if self._runtime is not None:
            with contextlib.suppress(Exception):
                self._runtime.close()
            self._runtime = None

        with contextlib.suppress(Exception):
            self._ctx.term()

        if self._metrics_sink is not None:
            self._metrics_sink.close()

    # ── Main inference loop ───────────────────────────────────────

    def run(
        self,
        total_steps: int = 0,
        total_episodes: int = 0,
        *,
        log_every: int = 100,
    ) -> InferenceMetrics:
        """Execute the inference loop.

        Args:
            total_steps: Stop after this many env steps (0 = unlimited).
            total_episodes: Stop after this many completed episodes (0 = unlimited).
            log_every: Log summary every N steps.

        Returns:
            Aggregate inference metrics.
        """
        n = self._n_actors
        unbounded = total_steps <= 0 and total_episodes <= 0

        # Episode tracking tensors (device-resident)
        episode_returns = torch.zeros((n,), dtype=torch.float32, device=self._device)
        episode_lengths = torch.zeros((n,), dtype=torch.int32, device=self._device)

        # Completed episode accumulators (host)
        completed_returns: list[float] = []
        completed_lengths: list[float] = []

        # Initialize observations
        current_obs_batch, obs_dict = self._load_initial_observation_batch()
        self._publish_dashboard_observations(obs_dict)
        self._publish_scene_telemetry(step_id=0)

        # Accumulators
        step_id = 0
        acc_fwd = 0.0
        acc_step = 0.0
        acc_tick = 0.0
        t_cnt = 0
        reward_sum = 0.0
        t_start = time.perf_counter()
        last_log_step = 0

        try:
            while True:
                # Termination checks
                if not unbounded:
                    if total_steps > 0 and step_id >= total_steps:
                        break
                    if total_episodes > 0 and len(completed_returns) >= total_episodes:
                        break

                t_tick = time.perf_counter()

                # Determine whether to publish dashboard observation this step
                publish_observations = self._should_publish_dashboard_observation()
                publish_actor_ids = (0,) if publish_observations else ()

                # ── Forward pass (no_grad, eval mode) ──────────
                t_fwd = time.perf_counter()
                with torch.no_grad():
                    if self._deterministic:
                        actions = self._deterministic_forward(current_obs_batch)
                    else:
                        actions, _lp, _v, _h, _z = self._policy.forward(
                            current_obs_batch, None
                        )
                fwd_ms = (time.perf_counter() - t_fwd) * 1000

                # ── Environment step ───────────────────────────
                t_step = time.perf_counter()
                step_batch, _results = self._runtime.batch_step_tensor_actions(
                    actions.detach(),
                    step_id,
                    publish_actor_ids=publish_actor_ids,
                    materialize_results=False,
                )
                step_ms = (time.perf_counter() - t_step) * 1000

                # ── Update observations ────────────────────────
                current_obs_batch = step_batch.observation_tensor

                # ── Rewards and episode tracking ───────────────
                rewards = step_batch.reward_tensor.to(
                    device=self._device, dtype=torch.float32
                )
                dones = step_batch.done_tensor.to(device=self._device, dtype=torch.bool)
                truncateds = step_batch.truncated_tensor.to(
                    device=self._device, dtype=torch.bool
                )
                done_or_truncated = dones | truncateds

                episode_returns += rewards
                episode_lengths += 1
                reward_sum += float(rewards.sum().item())

                # ── Detect completed episodes ──────────────────
                done_indices = torch.nonzero(done_or_truncated, as_tuple=False).flatten()
                if int(done_indices.numel()) > 0:
                    done_returns = episode_returns[done_indices].cpu().tolist()
                    done_lens = episode_lengths[done_indices].cpu().tolist()
                    completed_returns.extend(done_returns)
                    completed_lengths.extend(float(ln) for ln in done_lens)

                    # Emit episode telemetry for actor 0
                    for idx in range(int(done_indices.numel())):
                        actor_id = int(done_indices[idx].item())
                        self._publish_episode_telemetry(
                            step_id=step_id,
                            actor_id=actor_id,
                            episode_return=done_returns[idx],
                            episode_length=int(done_lens[idx]),
                        )

                    # Reset episode accumulators
                    not_done_f32 = (~done_or_truncated).to(dtype=torch.float32)
                    not_done_i32 = (~done_or_truncated).to(dtype=torch.int32)
                    episode_returns *= not_done_f32
                    episode_lengths *= not_done_i32

                # ── Observation publishing ─────────────────────
                if publish_observations and step_batch.published_observations:
                    self._publish_dashboard_observations(
                        step_batch.published_observations
                    )

                # ── Decimated telemetry ────────────────────────
                if step_id % _TELEMETRY_EVERY_N == 0:
                    self._publish_inference_features(step_id)
                    self._publish_step_result_telemetry(
                        step_id=step_id,
                        rewards=rewards,
                        dones=dones,
                        truncateds=truncateds,
                    )
                    self._publish_action_telemetry(
                        step_id=step_id, actions=actions,
                    )

                tick_ms = (time.perf_counter() - t_tick) * 1000
                acc_fwd += fwd_ms
                acc_step += step_ms
                acc_tick += tick_ms
                t_cnt += 1
                step_id += n  # n actors each stepped once

                # ── Periodic logging and perf telemetry ────────
                if step_id - last_log_step >= log_every:
                    elapsed = max(0.001, time.perf_counter() - t_start)
                    sps = step_id / elapsed
                    ep_count = len(completed_returns)
                    mean_ret = (
                        float(np.mean(completed_returns)) if completed_returns else 0.0
                    )
                    _LOGGER.info(
                        "step=%d episodes=%d sps=%.1f mean_return=%.3f "
                        "fwd_ms=%.2f step_ms=%.2f tick_ms=%.2f",
                        step_id,
                        ep_count,
                        sps,
                        mean_ret,
                        acc_fwd / max(1, t_cnt),
                        acc_step / max(1, t_cnt),
                        acc_tick / max(1, t_cnt),
                    )
                    self._publish_perf_telemetry(
                        step_id=step_id,
                        sps=sps,
                        forward_pass_ms=acc_fwd / max(1, t_cnt),
                        batch_step_ms=acc_step / max(1, t_cnt),
                        tick_total_ms=acc_tick / max(1, t_cnt),
                    )
                    self._publish_scene_telemetry(step_id=step_id)
                    last_log_step = step_id

        except KeyboardInterrupt:
            _LOGGER.info("Inference interrupted at step %d", step_id)

        # ── Summary ────────────────────────────────────────────────
        wall_clock = time.perf_counter() - t_start
        returns_arr = np.array(completed_returns, dtype=np.float32) if completed_returns else np.zeros(1, dtype=np.float32)
        lengths_arr = np.array(completed_lengths, dtype=np.float32) if completed_lengths else np.zeros(1, dtype=np.float32)

        metrics = InferenceMetrics(
            total_steps=step_id,
            total_episodes=len(completed_returns),
            reward_mean=reward_sum / max(1, step_id),
            reward_std=float(np.std(returns_arr)),
            episode_return_mean=float(np.mean(returns_arr)),
            episode_return_std=float(np.std(returns_arr)),
            episode_length_mean=float(np.mean(lengths_arr)),
            episode_length_std=float(np.std(lengths_arr)),
            sps_mean=step_id / max(0.001, wall_clock),
            forward_pass_ms_mean=acc_fwd / max(1, t_cnt),
            batch_step_ms_mean=acc_step / max(1, t_cnt),
            wall_clock_seconds=wall_clock,
        )

        self._emit_metrics_record(
            "inference_summary",
            build_phase_metrics_payload(
                "inference_summary",
                elapsed_ms=wall_clock * 1000.0,
                step_id=step_id,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "total_steps": metrics.total_steps,
                    "total_episodes": metrics.total_episodes,
                    "reward_mean": metrics.reward_mean,
                    "episode_return_mean": metrics.episode_return_mean,
                    "episode_return_std": metrics.episode_return_std,
                    "episode_length_mean": metrics.episode_length_mean,
                    "sps_mean": metrics.sps_mean,
                    "deterministic": self._deterministic,
                },
            ),
        )
        return metrics

    # ── Policy helpers ────────────────────────────────────────────

    def _deterministic_forward(
        self, obs_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass returning the action mean (no sampling noise)."""
        z_t = self._policy.encoder(obs_tensor)
        features, _hidden = self._policy.temporal_core.forward_step(z_t, None)
        actions: torch.Tensor = self._policy.heads.mean(features)
        return actions

    # ── Observation helpers ───────────────────────────────────────

    def _load_initial_observation_batch(
        self,
    ) -> tuple[torch.Tensor, dict[int, DistanceMatrix]]:
        """Seed initial observations from the runtime."""
        observation_tensors: list[torch.Tensor] = []
        obs_dict: dict[int, DistanceMatrix] = {}
        for actor_id in range(self._n_actors):
            materialize = actor_id == 0
            obs_tensor, published = self._runtime.reset_tensor(
                0,
                actor_id=actor_id,
                materialize=materialize,
            )
            observation_tensors.append(obs_tensor.to(self._device))
            if published is not None:
                obs_dict[actor_id] = published
        _LOGGER.info(
            "Inference runner seeded %d initial observations.",
            self._n_actors,
        )
        return torch.stack(observation_tensors), obs_dict

    def _should_publish_dashboard_observation(self) -> bool:
        """Return True when a fresh dashboard frame should be materialized."""
        if not self._config.emit_observation_stream:
            return False
        interval = 1.0 / max(float(self._config.dashboard_observation_hz), 1.0)
        return (time.perf_counter() - self._last_dashboard_heartbeat_at) >= interval

    def _publish_observation(self, observation: DistanceMatrix) -> None:
        """Publish a low-volume live observation for dashboard rendering."""
        if not self._config.emit_observation_stream:
            return
        self._last_dashboard_observation = observation
        self._last_dashboard_heartbeat_at = time.perf_counter()
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_DISTANCE_MATRIX,
            payload=lambda obs=observation: serialize(obs),
            label="observation",
        )

    def _publish_dashboard_observations(
        self, observations: dict[int, DistanceMatrix]
    ) -> None:
        """Publish dashboard observations for visible actors."""
        if not observations:
            return
        for actor_id in sorted(observations):
            self._publish_observation(observations[actor_id])

    # ── Telemetry ─────────────────────────────────────────────────

    def _telemetry_worker(self) -> None:
        """Background thread for non-blocking ZMQ telemetry emission."""
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 50)
        pub.bind(self._config.pub_address)

        _LOGGER.info("Inference telemetry worker started on %s", self._config.pub_address)

        while True:
            item = self._telemetry_queue.get()
            if item is None:
                break
            topic, payload, label = item
            try:
                raw = payload() if callable(payload) else payload
                pub.send_multipart([topic.encode("utf-8"), raw], flags=zmq.NOBLOCK)
            except Exception as exc:
                _LOGGER.warning("Telemetry send failure (%s): %s", label, exc)
            finally:
                self._telemetry_queue.task_done()

        pub.close(linger=0)
        ctx.term()
        _LOGGER.info("Inference telemetry worker stopped")

    def _publish_inference_features(self, step_id: int) -> None:
        """Publish spherical features for dashboard INFERENCE mode detection."""
        obs = self._last_dashboard_observation
        if obs is None:
            return
        features = extract_spherical_features(obs)
        event = TelemetryEvent(
            event_type="actor.inference.features",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=features.astype(np.float32, copy=False),
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="inference features",
        )

    def _publish_step_result_telemetry(
        self,
        *,
        step_id: int,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        truncateds: torch.Tensor,
    ) -> None:
        """Publish actor.step_result for dashboard reward chart (actor 0 only)."""
        reward_val = float(rewards[0].item())
        done_val = float(dones[0].item())
        truncated_val = float(truncateds[0].item())
        event = TelemetryEvent(
            event_type="actor.step_result",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=np.array(
                [reward_val, 0.0, done_val, truncated_val], dtype=np.float32
            ),
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="step result",
        )

    def _publish_action_telemetry(
        self, *, step_id: int, actions: torch.Tensor
    ) -> None:
        """Publish actor.action_published for dashboard forward/yaw charts (actor 0)."""
        a = actions[0].detach().cpu()
        event = TelemetryEvent(
            event_type="actor.action_published",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=np.array(
                [float(a[0]), float(a[1]), float(a[2]), float(a[3])],
                dtype=np.float32,
            ),
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="action published",
        )

    def _publish_episode_telemetry(
        self,
        *,
        step_id: int,
        actor_id: int,
        episode_return: float,
        episode_length: int,
    ) -> None:
        """Publish episode completion events for dashboard."""
        event = TelemetryEvent(
            event_type="actor.inference.episode",
            episode_id=0,
            env_id=actor_id,
            step_id=step_id,
            payload=np.array(
                [episode_return, float(episode_length)], dtype=np.float32
            ),
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="inference episode",
        )

    def _publish_perf_telemetry(
        self,
        *,
        step_id: int,
        sps: float,
        forward_pass_ms: float,
        batch_step_ms: float,
        tick_total_ms: float,
    ) -> None:
        """Publish performance telemetry for dashboard SPS display."""
        if not self._config.emit_perf_telemetry:
            return
        p = np.array(
            [sps, forward_pass_ms, batch_step_ms, tick_total_ms, float(self._n_actors)],
            dtype=np.float32,
        )
        event = TelemetryEvent(
            event_type="actor.inference.perf",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=p,
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="inference perf",
        )

    def _publish_scene_telemetry(self, *, step_id: int) -> None:
        """Publish current scene name so the dashboard can display it."""
        if not self._config.emit_perf_telemetry:
            return
        scene_name: str = getattr(self._runtime, "current_scene_name", "")
        if not scene_name:
            return
        payload = np.array(
            [float(ord(c)) for c in scene_name], dtype=np.float32
        )
        event = TelemetryEvent(
            event_type="actor.scene",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=payload,
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="scene telemetry",
        )

    def _emit_metrics_record(
        self, category: str, payload: dict[str, Any]
    ) -> None:
        """Write to the run-scoped JSONL metrics sink if enabled."""
        if self._metrics_sink is None:
            return
        self._metrics_sink.emit(category, payload)
