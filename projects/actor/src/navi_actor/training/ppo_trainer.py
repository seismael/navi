"""Single canonical PPO trainer for direct in-process sdfdag training."""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
import zmq

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import ActorConfig
from navi_actor.learner_ppo import PpoLearner, PpoMetrics
from navi_actor.memory.episodic import EpisodicMemory
from navi_actor.reward_shaping import RewardShaper
from navi_actor.rnd import RNDModule
from navi_actor.rollout_buffer import MultiTrajectoryBuffer
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


class CanonicalRuntime(Protocol):
    """Minimal runtime surface required by the canonical trainer."""

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


__all__: list[str] = ["PpoTrainer", "PpoTrainingMetrics"]

_LOGGER = logging.getLogger(__name__)

_SOFT_WARN_MIN_SPS: float = 15.0
_SOFT_WARN_MAX_ZERO_WAIT: float = 0.20
_SOFT_WARN_MAX_OPT_MS: float = 30_000.0

_STEP_TELEMETRY_FIELDS: tuple[str, ...] = (
    "raw_reward",
    "shaped_reward",
    "intrinsic_reward",
    "loop_similarity",
    "is_loop",
    "beta",
    "done",
    "forward_vel",
    "yaw_vel",
    "exploration_reward",
    "progress_reward",
    "clearance_reward",
    "starvation_penalty",
    "proximity_penalty",
    "structure_reward",
    "forward_structure_reward",
    "inspection_reward",
    "collision_reward",
)

_UPDATE_TELEMETRY_FIELDS: tuple[str, ...] = (
    "reward_ema",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "total_loss",
    "rnd_loss",
    "beta",
)

_PERF_TELEMETRY_FIELDS: tuple[str, ...] = (
    "sps",
    "forward_pass_ms",
    "batch_step_ms",
    "memory_query_ms",
    "transition_ms",
    "tick_total_ms",
    "zero_wait_ratio",
    "ppo_update_ms",
    "host_extract_ms",
    "telemetry_publish_ms",
    "ppo_inference_ms",
    "ppo_optimization_ms",
    "ppo_aux_ms",
    "n_actors",
)

_RUNTIME_PERF_FIELDS: tuple[str, ...] = (
    "sps",
    "last_batch_step_ms",
    "ema_batch_step_ms",
    "avg_batch_step_ms",
    "avg_actor_step_ms",
    "total_batches",
    "total_actor_steps",
)


def _explained_learner_stage_ms_total(metrics: PpoMetrics) -> float:
    """Return the summed learner-side stage time across all PPO minibatch updates."""
    return (
        metrics.epoch_setup_ms
        + metrics.iterator_setup_ms_total
        + metrics.minibatch_fetch_ms_total
        + metrics.minibatch_prep_ms_total
        + metrics.policy_eval_ms_total
        + metrics.backward_ms_total
        + metrics.grad_clip_ms_total
        + metrics.optimizer_step_ms_total
        + metrics.rnd_step_ms_total
        + metrics.post_update_stats_ms_total
        + metrics.update_loop_overhead_ms_total
        + metrics.progress_callback_ms_total
        + metrics.epoch_finalize_ms
    )


def _explained_learner_device_stage_ms_total(metrics: PpoMetrics) -> float:
    """Return the summed GPU execution time for diagnostic CUDA-profiled learner stages."""
    return (
        metrics.policy_eval_device_ms_total
        + metrics.backward_device_ms_total
        + metrics.grad_clip_device_ms_total
        + metrics.optimizer_step_device_ms_total
        + metrics.rnd_step_device_ms_total
        + metrics.post_update_stats_device_ms_total
    )


def _enqueue_telemetry(
    telemetry_queue: queue.Queue[tuple[str, bytes | Callable[[], bytes], str] | None] | None,
    *,
    topic: str,
    payload: bytes | Callable[[], bytes],
    label: str,
) -> None:
    """Best-effort async telemetry enqueue.

    *payload* may be pre-serialized ``bytes`` or a zero-arg callable that produces
    bytes.  Callables are resolved lazily by the telemetry worker thread so
    serialization cost does not block the training hot path.
    """
    if telemetry_queue is None:
        return
    try:
        telemetry_queue.put_nowait((topic, payload, label))
    except queue.Full:
        _LOGGER.warning("Dropping %s publish: telemetry queue full", label)
    except Exception as exc:  # pragma: no cover
        _LOGGER.warning("Failed to enqueue %s: %s", label, exc)


def _should_save_checkpoint(
    step_id: int, last_checkpoint_step: int, checkpoint_every: int
) -> bool:
    """Return True when a full checkpoint interval has elapsed.

    Rollout updates advance in chunks, so modulo-based checks can miss saves when
    boundaries do not align exactly with ``checkpoint_every``.
    """
    return checkpoint_every > 0 and (step_id - last_checkpoint_step) >= checkpoint_every


def _cpu_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Recursively move all tensors in an optimizer state dict to CPU."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu()
        elif isinstance(v, dict):
            out[k] = _cpu_state_dict(v)
        elif isinstance(v, list):
            out[k] = [
                x.cpu() if isinstance(x, torch.Tensor) else x for x in v
            ]
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class PpoTrainingMetrics:
    """Summary metrics from a PPO training run."""

    total_steps: int
    episodes: int
    reward_mean: float
    reward_ema: float
    policy_loss: float
    value_loss: float
    entropy: float
    rnd_loss: float
    intrinsic_reward_mean: float
    loop_detections: int
    beta_final: float
    zero_wait_ratio: float = 0.0
    # Performance instrumentation
    sps_mean: float = 0.0
    forward_pass_ms_mean: float = 0.0
    action_pack_ms_mean: float = 0.0
    batch_step_ms_mean: float = 0.0
    memory_query_ms_mean: float = 0.0
    transition_ms_mean: float = 0.0
    reward_shape_ms_mean: float = 0.0
    memory_add_ms_mean: float = 0.0
    buffer_append_ms_mean: float = 0.0
    host_extract_ms_mean: float = 0.0
    telemetry_publish_ms_mean: float = 0.0
    tick_total_ms_mean: float = 0.0
    ppo_update_ms_mean: float = 0.0
    ppo_inference_ms_mean: float = 0.0
    ppo_optimization_ms_mean: float = 0.0
    ppo_aux_ms_mean: float = 0.0
    wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class _RolloutHostBatch:
    """Detached rollout tensors retained on device for rollout bookkeeping."""

    embedding_tensor: torch.Tensor
    action_tensor: torch.Tensor
    aux_tensor: torch.Tensor
    log_prob_tensor: torch.Tensor
    value_tensor: torch.Tensor
    intrinsic_reward_tensor: torch.Tensor


@dataclass(frozen=True)
class _RolloutScalarBatch:
    """Selected rollout telemetry tensors retained on device until materialization."""

    raw_reward_tensor: torch.Tensor | None
    shaped_reward_tensor: torch.Tensor | None
    intrinsic_reward_tensor: torch.Tensor | None
    loop_similarity_tensor: torch.Tensor | None
    done_tensor: torch.Tensor | None


@dataclass(frozen=True)
class _StepTelemetryHostBatch:
    """One packed CPU mirror for sparse step telemetry publication."""

    raw_reward_tensor: torch.Tensor | None
    shaped_reward_tensor: torch.Tensor | None
    intrinsic_reward_tensor: torch.Tensor | None
    loop_similarity_tensor: torch.Tensor | None
    done_tensor: torch.Tensor | None
    forward_velocity_tensor: torch.Tensor | None
    yaw_velocity_tensor: torch.Tensor | None
    reward_component_tensor: torch.Tensor | None


@dataclass(frozen=True)
class _CompletedEpisodeHostBatch:
    """CPU mirror for the sparse set of completed episodes in one rollout tick."""

    actor_id_tensor: torch.Tensor
    episode_return_tensor: torch.Tensor
    episode_length_tensor: torch.Tensor


@dataclass(frozen=True)
class _PreparedRolloutGroup:
    """One actor subgroup snapshot prepared from the current rollout state."""

    group_slot: int
    actor_indices: torch.Tensor
    observations: torch.Tensor
    aux_tensor: torch.Tensor


@dataclass(frozen=True)
class _ForwardRolloutGroup:
    """Forward-pass outputs for one actor subgroup."""

    actions: torch.Tensor
    embeddings: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    intrinsic_rewards: torch.Tensor
    fwd_ms: float


@dataclass(frozen=True)
class _PrimedRolloutGroup:
    """Prepared subgroup state plus its already-computed next action."""

    prepared: _PreparedRolloutGroup
    forward: _ForwardRolloutGroup


@dataclass(frozen=True)
class _PendingRolloutGroup:
    """One rollout subgroup launched asynchronously on its own CUDA stream."""

    group_slot: int
    actor_indices: torch.Tensor
    observations: torch.Tensor
    aux_tensor: torch.Tensor
    actions: torch.Tensor
    embeddings: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    intrinsic_rewards: torch.Tensor
    step_batch: SdfDagTensorStepBatch
    fwd_ms: float
    step_ms: float


@dataclass(frozen=True)
class _FinalizeRolloutGroupResult:
    """Per-group rollout accounting emitted after one env step is finalized."""

    reward_ema: torch.Tensor
    normalized_memory_batch: torch.Tensor | None
    episode_count: torch.Tensor
    pack_ms: float
    mem_ms: float
    trans_ms: float
    shape_ms: float
    buffer_ms: float
    host_extract_ms: float
    telemetry_ms: float


class PpoTrainer:
    """Single canonical PPO rollout/update trainer.

    This trainer owns the only supported hot path: direct in-process stepping
    against the compiled `sdfdag` runtime while preserving the `DistanceMatrix`
    actor contract unchanged.
    """

    def __init__(
        self,
        config: ActorConfig,
        *,
        runtime: CanonicalRuntime | None = None,
        gmdag_file: str = "",
        scene_pool: tuple[str, ...] = (),
    ) -> None:
        init_started_at = time.perf_counter()
        self._config = config
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._runtime = runtime
        self._gmdag_file = gmdag_file
        self._scene_pool = scene_pool
        self._last_dashboard_observation: DistanceMatrix | None = None
        self._last_dashboard_heartbeat_at: float = 0.0
        self._run_context = get_or_create_run_context("actor-train")
        self._metrics_sink: JsonlMetricsSink | None = None
        if config.emit_internal_stats:
            self._metrics_sink = JsonlMetricsSink(
                self._run_context.metrics_root / "actor_training.jsonl",
                run_id=self._run_context.run_id,
                project_name="navi_actor_train",
            )

        self._n_actors = config.n_actors

        # Canonical runtime policy: PPO training must run on CUDA.
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for canonical PPO training runtime, but torch.cuda.is_available() is False.",
            )
        self._device = torch.device("cuda")
        try:
            x = torch.randn(8, 8, device=self._device)
            _ = torch.mm(x, x)
            torch.cuda.synchronize()
        except Exception as exc:  # pragma: no cover - hardware/runtime specific
            raise RuntimeError(
                "CUDA runtime is present but kernel execution failed for canonical PPO training runtime.",
            ) from exc

        _LOGGER.info(
            "PpoTrainer CUDA preflight OK: device=%s capability=sm_%d%d cuda=%s",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_capability(0)[0],
            torch.cuda.get_device_capability(0)[1],
            str(torch.version.cuda),
        )

        # Keep a dedicated rollout copy so training updates happen only at
        # rollout boundaries and never mutate the inference model mid-step.
        self._learner_policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            temporal_core=config.temporal_core,
            encoder_backend=config.encoder_backend,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)

        self._rollout_policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            temporal_core=config.temporal_core,
            encoder_backend=config.encoder_backend,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)

        # Shared RND curiosity module
        self._rnd = RNDModule(
            input_dim=config.embedding_dim,
        ).to(self._device)

        # Initial sync
        self._sync_rollout_policy()

        # SHARED episodic memory for all actors (High Efficiency)
        self._memory = EpisodicMemory(
            embedding_dim=config.embedding_dim,
            capacity=config.memory_capacity,
            exclusion_window=config.memory_exclusion_window,
            similarity_threshold=config.loop_threshold,
        )

        # Shared reward shaper with annealing
        self._reward_shaper = RewardShaper(
            collision_penalty=config.collision_penalty,
            existential_tax=config.existential_tax,
            velocity_weight=config.velocity_weight,
            intrinsic_coeff_init=config.intrinsic_coeff_init,
            intrinsic_coeff_final=config.intrinsic_coeff_final,
            intrinsic_anneal_steps=config.intrinsic_anneal_steps,
            loop_penalty_coeff=config.loop_penalty_coeff,
            loop_threshold=config.loop_threshold,
            torch_compile=config.reward_shaping_torch_compile,
        )
        _LOGGER.info(
            "RewardShaper compile status: requested=%s active=%s",
            self._reward_shaper.torch_compile_requested,
            self._reward_shaper.torch_compile_enabled,
        )

        # Shared learner
        self._learner = PpoLearner(
            gamma=config.gamma,
            clip_ratio=config.clip_ratio,
            entropy_coeff=config.entropy_coeff,
            value_coeff=config.value_coeff,
            learning_rate=config.learning_rate,
            profile_cuda_events=config.profile_cuda_events,
        )
        self._learner.prime_update_runtime(self._learner_policy, rnd=self._rnd)

        if config.profile_cuda_events:
            _LOGGER.warning(
                "CUDA event profiling is enabled: PPO learner stage timings are synchronized diagnostic timings and throughput will be reduced.",
            )

        # Canonical batched rollout buffer
        self._multi_buffers = MultiTrajectoryBuffer(
            n_actors=self._n_actors,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            capacity=config.rollout_length,
            normalize_returns=config.normalize_returns,
        )

        self._sim_steps_during_opt: int = 0
        self._total_sim_steps: int = 0
        self._total_episodes: int = 0
        self._reward_ema: float = 0.0
        self._training_wall_start: float = time.perf_counter()
        self._parent_checkpoint: str | None = None
        self._last_opt_duration_ms: float = 0.0
        self._opt_duration_acc: float = 0.0
        self._opt_duration_count: int = 0
        self._opt_inference_acc: float = 0.0
        self._opt_optimization_acc: float = 0.0
        self._opt_aux_acc: float = 0.0
        self._last_opt_metrics: PpoMetrics | None = None

        # Phase 11: Async Telemetry
        self._telemetry_queue: queue.Queue[
            tuple[str, bytes | Callable[[], bytes], str] | None
        ] = queue.Queue(
            maxsize=1024
        )
        self._telemetry_thread: threading.Thread | None = None
        self._checkpoint_thread: threading.Thread | None = None
        self._rollout_group_count = max(1, min(self._config.rollout_overlap_groups, self._n_actors))
        self._rollout_group_indices = tuple(
            group.contiguous()
            for group in torch.chunk(
                torch.arange(self._n_actors, dtype=torch.int64, device=self._device),
                self._rollout_group_count,
            )
            if int(group.numel()) > 0
        )
        self._rollout_group_actor_ids = tuple(
            tuple(
                int(actor_id)
                for actor_id in group.detach().to(device="cpu", dtype=torch.int64).tolist()
            )
            for group in self._rollout_group_indices
        )
        self._rollout_streams = tuple(
            torch.cuda.Stream(device=self._device)  # type: ignore[no-untyped-call]
            for _ in self._rollout_group_indices
        )
        self._emit_metrics_record(
            "lifecycle",
            build_phase_metrics_payload(
                "trainer_init",
                started_at=init_started_at,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "actors": self._n_actors,
                    "temporal_core": self._config.temporal_core,
                    "scene_count": len(self._scene_pool),
                    "gmdag_file": self._gmdag_file or "<state-only>",
                },
            ),
        )

    def _emit_metrics_record(self, stream: str, payload: dict[str, Any]) -> None:
        if self._metrics_sink is None:
            return
        self._metrics_sink.emit(stream, payload)

    def _telemetry_worker(self) -> None:
        """Background thread for non-blocking ZMQ telemetry emission (Phase 11)."""
        import zmq

        # Create a dedicated context and socket for this thread
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 50)
        pub.bind(self._config.pub_address)

        _LOGGER.info("Async telemetry worker started on %s", self._config.pub_address)

        while True:
            item = self._telemetry_queue.get()
            if item is None:  # Sentinel for shutdown
                break

            topic, payload, label = item
            try:
                # Resolve lazy serialization callables
                raw = payload() if callable(payload) else payload
                pub.send_multipart([topic.encode("utf-8"), raw], flags=zmq.NOBLOCK)
            except Exception as exc:
                _LOGGER.warning("Telemetry send failure (%s): %s", label, exc)
            finally:
                self._telemetry_queue.task_done()

        pub.close(linger=0)
        ctx.term()
        _LOGGER.info("Async telemetry worker stopped")

    def _sync_rollout_policy(self) -> None:
        """Copy weights from learner_policy to rollout_policy with thread safety."""
        self._rollout_policy.load_state_dict(self._learner_policy.state_dict())
        self._rollout_policy.eval()

    def _run_ppo_update(
        self,
        *,
        multi_buffer: MultiTrajectoryBuffer,
        obs_batch: torch.Tensor,
        aux_batch: torch.Tensor,
        ppo_epochs: int,
        minibatch_size: int,
        seq_len: int,
    ) -> None:
        t_opt_start = time.perf_counter()

        # PPO Inference sub-stage (GAE + advantage calculation)
        t_inf_start = time.perf_counter()
        with torch.no_grad():
            # Rollout is paused during PPO update, so the latest bootstrap tensors can be
            # reused directly instead of cloning full CUDA state before the value pass.
            b_obs = obs_batch.detach()
            b_aux = aux_batch.detach()
            if b_obs.device != self._device:
                b_obs = b_obs.to(self._device)
            if b_aux.device != self._device:
                b_aux = b_aux.to(self._device)

            self._learner_policy.eval()
            _, _, b_val, _, _ = self._learner_policy.forward(b_obs, None, aux_tensor=b_aux)
            multi_buffer.compute_returns_and_advantages(last_values=b_val)
        inf_ms = (time.perf_counter() - t_inf_start) * 1000
        self._opt_inference_acc += inf_ms

        # PPO Optimization sub-stage (The actual loss.backward() and optimizer.step() loop)
        t_train_start = time.perf_counter()
        progress_callback: Callable[[], None] | None = None
        if self._config.emit_observation_stream and self._last_dashboard_observation is not None:

            def emit_progress_callback() -> None:
                self._maybe_publish_dashboard_heartbeat(step_id=self._total_sim_steps)

            progress_callback = emit_progress_callback

        self._last_opt_metrics = self._learner.train_ppo_epoch(
            self._learner_policy,
            multi_buffer,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            seq_len=seq_len,
            rnd=self._rnd,
            progress_callback=progress_callback,
            materialize_summary_scalars=(
                (self._config.emit_training_telemetry and self._config.emit_update_loss_telemetry)
                or _LOGGER.isEnabledFor(logging.DEBUG)
            ),
        )
        opt_ms = (time.perf_counter() - t_train_start) * 1000
        self._opt_optimization_acc += opt_ms

        # PPO Aux sub-stage (Metrics, logging, and checkpointing overhead)
        t_aux_start = time.perf_counter()
        self._sync_rollout_policy()
        multi_buffer.clear()
        aux_ms = (time.perf_counter() - t_aux_start) * 1000
        self._opt_aux_acc += aux_ms

        total_ms = (time.perf_counter() - t_opt_start) * 1000
        self._last_opt_duration_ms = total_ms
        self._opt_duration_acc += total_ms
        self._opt_duration_count += 1

        if self._last_opt_metrics is None:
            _LOGGER.info("Inline PPO optimization completed in %.2fms. Weights synced.", total_ms)
        else:
            explained_ms_total = _explained_learner_stage_ms_total(self._last_opt_metrics)
            learner_gap_ms = max(0.0, self._last_opt_metrics.epoch_total_ms - explained_ms_total)
            trainer_gap_ms = max(0.0, opt_ms - self._last_opt_metrics.epoch_total_ms)
            gap_ms = max(0.0, opt_ms - explained_ms_total)
            gap_pct = (gap_ms / opt_ms * 100.0) if opt_ms > 0.0 else 0.0
            diagnostic_suffix = ""
            if self._config.profile_cuda_events:
                gpu_explained_ms_total = _explained_learner_device_stage_ms_total(
                    self._last_opt_metrics
                )
                eval_host_gap = max(
                    0.0,
                    self._last_opt_metrics.policy_eval_ms_total
                    - self._last_opt_metrics.policy_eval_device_ms_total,
                )
                backward_host_gap = max(
                    0.0,
                    self._last_opt_metrics.backward_ms_total
                    - self._last_opt_metrics.backward_device_ms_total,
                )
                clip_host_gap = max(
                    0.0,
                    self._last_opt_metrics.grad_clip_ms_total
                    - self._last_opt_metrics.grad_clip_device_ms_total,
                )
                optimizer_host_gap = max(
                    0.0,
                    self._last_opt_metrics.optimizer_step_ms_total
                    - self._last_opt_metrics.optimizer_step_device_ms_total,
                )
                rnd_host_gap = max(
                    0.0,
                    self._last_opt_metrics.rnd_step_ms_total
                    - self._last_opt_metrics.rnd_step_device_ms_total,
                )
                stats_host_gap = max(
                    0.0,
                    self._last_opt_metrics.post_update_stats_ms_total
                    - self._last_opt_metrics.post_update_stats_device_ms_total,
                )
                eval_encode_avg = self._last_opt_metrics.policy_eval_encode_ms_total / max(
                    1, self._last_opt_metrics.n_updates
                )
                eval_temporal_avg = self._last_opt_metrics.policy_eval_temporal_ms_total / max(
                    1, self._last_opt_metrics.n_updates
                )
                eval_heads_avg = self._last_opt_metrics.policy_eval_heads_ms_total / max(
                    1, self._last_opt_metrics.n_updates
                )
                diagnostic_suffix = (
                    f" diag(gpu_eval={self._last_opt_metrics.policy_eval_device_ms_total:.2f}ms "
                    f"gpu_eval_enc={self._last_opt_metrics.policy_eval_encode_device_ms_total:.2f}ms "
                    f"gpu_eval_temp={self._last_opt_metrics.policy_eval_temporal_device_ms_total:.2f}ms "
                    f"gpu_eval_heads={self._last_opt_metrics.policy_eval_heads_device_ms_total:.2f}ms "
                    f"gpu_bwd={self._last_opt_metrics.backward_device_ms_total:.2f}ms "
                    f"gpu_clip={self._last_opt_metrics.grad_clip_device_ms_total:.2f}ms "
                    f"gpu_optim={self._last_opt_metrics.optimizer_step_device_ms_total:.2f}ms "
                    f"gpu_rnd={self._last_opt_metrics.rnd_step_device_ms_total:.2f}ms "
                    f"gpu_stats={self._last_opt_metrics.post_update_stats_device_ms_total:.2f}ms "
                    f"gpu_explained={gpu_explained_ms_total:.2f}ms "
                    f"eval_split(avg_enc={eval_encode_avg:.2f}ms avg_temp={eval_temporal_avg:.2f}ms avg_heads={eval_heads_avg:.2f}ms "
                    f"tot_enc={self._last_opt_metrics.policy_eval_encode_ms_total:.2f}ms tot_temp={self._last_opt_metrics.policy_eval_temporal_ms_total:.2f}ms tot_heads={self._last_opt_metrics.policy_eval_heads_ms_total:.2f}ms) "
                    f"eval_host_gap={eval_host_gap:.2f}ms "
                    f"bwd_host_gap={backward_host_gap:.2f}ms "
                    f"clip_host_gap={clip_host_gap:.2f}ms "
                    f"optim_host_gap={optimizer_host_gap:.2f}ms "
                    f"rnd_host_gap={rnd_host_gap:.2f}ms "
                    f"stats_host_gap={stats_host_gap:.2f}ms)"
                )
            _LOGGER.info(
                "Inline PPO optimization completed in %.2fms (inf=%.2fms opt=%.2fms aux=%.2fms | updates=%d avg(fetch=%.2fms prep=%.2fms eval=%.2fms bwd=%.2fms clip=%.2fms optim=%.2fms rnd=%.2fms stats=%.2fms loop=%.2fms cb=%.2fms setup=%.2fms finalize=%.2fms) total(iter=%.2fms fetch=%.2fms prep=%.2fms eval=%.2fms bwd=%.2fms clip=%.2fms optim=%.2fms rnd=%.2fms stats=%.2fms loop=%.2fms cb=%.2fms learner=%.2fms explained=%.2fms learner_gap=%.2fms trainer_gap=%.2fms gap=%.2fms %.1f%% setup(policy=%.2fms rnd=%.2fms mode=%.2fms params=%.2fms other=%.2fms)%s). Weights synced.",
                total_ms,
                inf_ms,
                opt_ms,
                aux_ms,
                self._last_opt_metrics.n_updates,
                self._last_opt_metrics.minibatch_fetch_ms,
                self._last_opt_metrics.minibatch_prep_ms,
                self._last_opt_metrics.policy_eval_ms,
                self._last_opt_metrics.backward_ms,
                self._last_opt_metrics.grad_clip_ms,
                self._last_opt_metrics.optimizer_step_ms,
                self._last_opt_metrics.rnd_step_ms,
                self._last_opt_metrics.post_update_stats_ms,
                self._last_opt_metrics.update_loop_overhead_ms,
                self._last_opt_metrics.progress_callback_ms,
                self._last_opt_metrics.epoch_setup_ms,
                self._last_opt_metrics.epoch_finalize_ms,
                self._last_opt_metrics.iterator_setup_ms_total,
                self._last_opt_metrics.minibatch_fetch_ms_total,
                self._last_opt_metrics.minibatch_prep_ms_total,
                self._last_opt_metrics.policy_eval_ms_total,
                self._last_opt_metrics.backward_ms_total,
                self._last_opt_metrics.grad_clip_ms_total,
                self._last_opt_metrics.optimizer_step_ms_total,
                self._last_opt_metrics.rnd_step_ms_total,
                self._last_opt_metrics.post_update_stats_ms_total,
                self._last_opt_metrics.update_loop_overhead_ms_total,
                self._last_opt_metrics.progress_callback_ms_total,
                self._last_opt_metrics.epoch_total_ms,
                explained_ms_total,
                learner_gap_ms,
                trainer_gap_ms,
                gap_ms,
                gap_pct,
                self._last_opt_metrics.policy_optimizer_init_ms,
                self._last_opt_metrics.rnd_optimizer_init_ms,
                self._last_opt_metrics.policy_train_mode_ms,
                self._last_opt_metrics.policy_param_cache_ms,
                self._last_opt_metrics.setup_overhead_ms,
                diagnostic_suffix,
            )
        if total_ms > _SOFT_WARN_MAX_OPT_MS:
            _LOGGER.warning(
                "Soft stall monitor: optimizer wall-time high (%.1fms > %.1fms)",
                total_ms,
                _SOFT_WARN_MAX_OPT_MS,
            )
        if self._last_opt_metrics is not None:
            self._emit_metrics_record(
                "ppo_update_summary",
                build_phase_metrics_payload(
                    "ppo_update",
                    elapsed_ms=total_ms,
                    step_id=self._total_sim_steps,
                    cuda_device=self._device,
                    include_resources=self._config.attach_resource_snapshots,
                    metadata={
                        "total_ms": total_ms,
                        "inference_ms": inf_ms,
                        "optimization_ms": opt_ms,
                        "aux_ms": aux_ms,
                        "n_updates": self._last_opt_metrics.n_updates,
                        "policy_eval_ms_total": self._last_opt_metrics.policy_eval_ms_total,
                        "loss_build_ms_total": self._last_opt_metrics.loss_build_ms_total,
                        "backward_ms_total": self._last_opt_metrics.backward_ms_total,
                        "zero_grad_ms_total": self._last_opt_metrics.zero_grad_ms_total,
                        "optimizer_step_ms_total": self._last_opt_metrics.optimizer_step_ms_total,
                        "rnd_step_ms_total": self._last_opt_metrics.rnd_step_ms_total,
                        "epoch_total_ms": self._last_opt_metrics.epoch_total_ms,
                    },
                ),
            )

    def _run_staggered_ppo_update(
        self,
        *,
        actor_ids: torch.Tensor,
        multi_buffer: MultiTrajectoryBuffer,
        obs_batch: torch.Tensor,
        aux_batch: torch.Tensor,
        ppo_epochs: int,
        minibatch_size: int,
        seq_len: int,
    ) -> None:
        """Run a PPO update for a subset of actors (staggered per-actor PPO).

        Only the specified actors' rollout data is used for GAE, optimization,
        and buffer clearing.  The shared rollout policy is synced afterwards so
        all actors benefit from the update.
        """
        t_opt_start = time.perf_counter()
        n_ready = int(actor_ids.numel())

        # Stage 1: Inference — compute bootstrap values for ready actors only
        t_inf_start = time.perf_counter()
        with torch.no_grad():
            b_obs = obs_batch[actor_ids].detach()
            b_aux = aux_batch[actor_ids].detach()
            if b_obs.device != self._device:
                b_obs = b_obs.to(self._device)
            if b_aux.device != self._device:
                b_aux = b_aux.to(self._device)
            self._learner_policy.eval()
            _, _, b_val, _, _ = self._learner_policy.forward(b_obs, None, aux_tensor=b_aux)
            multi_buffer.compute_returns_and_advantages_for_actors(
                actor_ids=actor_ids, last_values=b_val
            )
        inf_ms = (time.perf_counter() - t_inf_start) * 1000
        self._opt_inference_acc += inf_ms

        # Stage 2: Optimization — train on ready actors' data only
        t_train_start = time.perf_counter()
        progress_callback: Callable[[], None] | None = None
        if self._config.emit_observation_stream and self._last_dashboard_observation is not None:

            def emit_progress_callback() -> None:
                self._maybe_publish_dashboard_heartbeat(step_id=self._total_sim_steps)

            progress_callback = emit_progress_callback

        self._last_opt_metrics = self._learner.train_ppo_epoch(
            self._learner_policy,
            multi_buffer,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            seq_len=seq_len,
            actor_ids=actor_ids,
            rnd=self._rnd,
            progress_callback=progress_callback,
            materialize_summary_scalars=(
                (self._config.emit_training_telemetry and self._config.emit_update_loss_telemetry)
                or _LOGGER.isEnabledFor(logging.DEBUG)
            ),
        )
        opt_ms = (time.perf_counter() - t_train_start) * 1000
        self._opt_optimization_acc += opt_ms

        # Stage 3: Auxiliary — sync policy and clear only the ready actors
        t_aux_start = time.perf_counter()
        self._sync_rollout_policy()
        multi_buffer.clear_actors(actor_ids)
        aux_ms = (time.perf_counter() - t_aux_start) * 1000
        self._opt_aux_acc += aux_ms

        total_ms = (time.perf_counter() - t_opt_start) * 1000
        self._last_opt_duration_ms = total_ms
        self._opt_duration_acc += total_ms
        self._opt_duration_count += 1

        actor_id_list = actor_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        _LOGGER.info(
            "Staggered PPO update for actors %s completed in %.2fms "
            "(inf=%.2fms opt=%.2fms aux=%.2fms n_ready=%d)",
            actor_id_list,
            total_ms,
            inf_ms,
            opt_ms,
            aux_ms,
            n_ready,
        )
        if total_ms > _SOFT_WARN_MAX_OPT_MS:
            _LOGGER.warning(
                "Soft stall monitor: staggered optimizer wall-time high (%.1fms > %.1fms) actors=%s",
                total_ms,
                _SOFT_WARN_MAX_OPT_MS,
                actor_id_list,
            )
        if self._last_opt_metrics is not None:
            self._emit_metrics_record(
                "ppo_update_summary",
                build_phase_metrics_payload(
                    "ppo_update",
                    elapsed_ms=total_ms,
                    step_id=self._total_sim_steps,
                    cuda_device=self._device,
                    include_resources=self._config.attach_resource_snapshots,
                    metadata={
                        "staggered": True,
                        "actor_ids": actor_id_list,
                        "total_ms": total_ms,
                        "inference_ms": inf_ms,
                        "optimization_ms": opt_ms,
                        "aux_ms": aux_ms,
                        "n_updates": self._last_opt_metrics.n_updates,
                    },
                ),
            )

    def start(self) -> None:
        """Initialize trainer resources for the canonical runtime."""
        started_at = time.perf_counter()
        if self._runtime is None:
            raise RuntimeError("Canonical sdfdag runtime is not configured")

        # Start background telemetry thread
        self._telemetry_thread = threading.Thread(target=self._telemetry_worker, daemon=True)
        self._telemetry_thread.start()

        _LOGGER.info(
            "Canonical PPO trainer started: actors=%d gmdag=%s pub=%s temporal=%s (async)",
            self._n_actors,
            self._gmdag_file or "<state-only>",
            self._config.pub_address,
            self._config.temporal_core,
        )
        self._emit_metrics_record(
            "lifecycle",
            build_phase_metrics_payload(
                "trainer_start",
                started_at=started_at,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "event": "trainer_started",
                    "actors": self._n_actors,
                    "gmdag_file": self._gmdag_file or "<state-only>",
                    "pub_address": self._config.pub_address,
                    "temporal_core": self._config.temporal_core,
                    "run_root": str(self._run_context.run_root),
                },
            ),
        )

    def stop(self) -> None:
        """Close trainer transport resources."""
        started_at = time.perf_counter()
        # Wait for any pending async checkpoint
        if self._checkpoint_thread is not None:
            self._checkpoint_thread.join(timeout=10.0)
            self._checkpoint_thread = None
        # Stop telemetry thread first
        if self._telemetry_thread:
            self._telemetry_queue.put(None)
            self._telemetry_thread.join(timeout=2.0)
            self._telemetry_thread = None

        if self._runtime is not None:
            with contextlib.suppress(Exception):
                self._runtime.close()
        self._runtime = None

        with contextlib.suppress(Exception):
            self._ctx.term()
        self._emit_metrics_record(
            "lifecycle",
            build_phase_metrics_payload(
                "trainer_stop",
                started_at=started_at,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "event": "trainer_stopped",
                    "total_sim_steps": self._total_sim_steps,
                    "optimizer_updates": self._opt_duration_count,
                },
            ),
        )
        if self._metrics_sink is not None:
            self._metrics_sink.close()

    def save_training_state(self, path: str | Path) -> None:
        """Save complete training state asynchronously.

        Copies state dicts to CPU on the main thread (fast), then writes
        to disk in a background thread to avoid blocking the training loop.
        """
        started_at = time.perf_counter()
        save_path = Path(path)
        _LOGGER.info("Saving training checkpoint to %s", save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Wait for any previous async checkpoint to complete
        if self._checkpoint_thread is not None:
            self._checkpoint_thread.join()
            self._checkpoint_thread = None

        # Corpus summary for provenance
        corpus_summary = f"{len(self._scene_pool)} scenes" if self._scene_pool else (self._gmdag_file or "unknown")

        # Snapshot state to CPU-resident dicts (fast H2D copy)
        state: dict[str, Any] = {
            "version": 3,
            "run_id": self._run_context.run_id,
            "step_id": self._total_sim_steps,
            "episode_count": self._total_episodes,
            "reward_ema": self._reward_ema,
            "wall_time_hours": (time.perf_counter() - self._training_wall_start) / 3600.0,
            "parent_checkpoint": self._parent_checkpoint,
            "training_source": "rl",
            "temporal_core": self._config.temporal_core,
            "encoder_backend": self._config.encoder_backend,
            "corpus_summary": corpus_summary,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "policy_state_dict": {
                k: v.cpu() for k, v in self._learner_policy.state_dict().items()
            },
            "rnd_state_dict": {k: v.cpu() for k, v in self._rnd.state_dict().items()},
            "reward_shaper_step": self._reward_shaper._global_step,
        }
        if self._learner._optimizer:
            state["optimizer_state_dict"] = _cpu_state_dict(
                self._learner._optimizer.state_dict()
            )
        if self._learner._rnd_optimizer:
            state["rnd_optimizer_state_dict"] = _cpu_state_dict(
                self._learner._rnd_optimizer.state_dict()
            )
        snapshot_ms = (time.perf_counter() - started_at) * 1000

        # Write to disk in background thread
        def _write() -> None:
            write_start = time.perf_counter()
            torch.save(state, save_path)
            total_ms = (time.perf_counter() - started_at) * 1000
            write_ms = (time.perf_counter() - write_start) * 1000
            _LOGGER.info(
                "Checkpoint written (snapshot=%.1fms write=%.1fms total=%.1fms): %s",
                snapshot_ms, write_ms, total_ms, save_path,
            )
            self._emit_metrics_record(
                "checkpoint",
                build_phase_metrics_payload(
                    "checkpoint_save",
                    started_at=started_at,
                    step_id=self._total_sim_steps,
                    cuda_device=self._device,
                    include_resources=self._config.attach_resource_snapshots,
                    metadata={
                        "event": "saved",
                        "path": str(save_path),
                        "reward_shaper_step": self._reward_shaper._global_step,
                        "snapshot_ms": round(snapshot_ms, 1),
                        "write_ms": round(write_ms, 1),
                    },
                ),
            )

        self._checkpoint_thread = threading.Thread(target=_write, daemon=True)
        self._checkpoint_thread.start()

    def load_training_state(self, path: str | Path) -> None:
        """Restore training state from a v3 canonical checkpoint."""
        started_at = time.perf_counter()
        _LOGGER.info("Loading training checkpoint from %s", path)
        data = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(data, dict) or data.get("version") != 3:
            raise RuntimeError(
                "Unsupported training checkpoint format: expected version 3 canonical state",
            )

        # Validate encoder backend matches
        stored_encoder = data.get("encoder_backend", "rayvit")
        if stored_encoder != self._config.encoder_backend:
            raise RuntimeError(
                f"Cannot load checkpoint: encoder backend mismatch. "
                f"Checkpoint uses '{stored_encoder}', "
                f"config specifies '{self._config.encoder_backend}'"
            )

        self._learner_policy.load_state_dict(data["policy_state_dict"])
        self._rnd.load_state_dict(data["rnd_state_dict"])
        self._reward_shaper._global_step = int(data["reward_shaper_step"])
        if "optimizer_state_dict" in data:
            self._learner._get_optimizer(self._learner_policy).load_state_dict(
                data["optimizer_state_dict"]
            )
        if "rnd_optimizer_state_dict" in data:
            self._learner._get_rnd_optimizer(self._rnd).load_state_dict(
                data["rnd_optimizer_state_dict"]
            )

        # Restore accumulated counters from v3 metadata
        self._total_sim_steps = int(data["step_id"])
        self._total_episodes = int(data["episode_count"])
        self._reward_ema = float(data["reward_ema"])
        self._parent_checkpoint = str(path)

        _LOGGER.info(
            "Checkpoint loaded (v3): step_id=%d episodes=%d reward_ema=%.4f source=%s",
            self._total_sim_steps,
            self._total_episodes,
            self._reward_ema,
            data["training_source"],
        )

        # Sync weights to rollout policy after loading
        self._sync_rollout_policy()
        self._emit_metrics_record(
            "checkpoint",
            build_phase_metrics_payload(
                "checkpoint_load",
                started_at=started_at,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "event": "loaded",
                    "path": str(path),
                    "checkpoint_version": 3,
                    "checkpoint_run_id": data["run_id"],
                    "reward_shaper_step": self._reward_shaper._global_step,
                    "step_id": self._total_sim_steps,
                    "episode_count": self._total_episodes,
                    "reward_ema": self._reward_ema,
                    "training_source": data["training_source"],
                    "parent_checkpoint": data["parent_checkpoint"],
                },
            ),
        )

    def _request_batch_step_tensor_actions(
        self,
        action_tensor: torch.Tensor,
        step_id: int,
        *,
        actor_indices: torch.Tensor | None = None,
        scratch_slot: int = 0,
        publish_actor_ids: tuple[int, ...] = (),
    ) -> SdfDagTensorStepBatch:
        runtime = self._require_tensor_runtime()
        step_batch, _results = runtime.batch_step_tensor_actions(
            action_tensor,
            step_id,
            actor_indices=actor_indices,
            scratch_slot=scratch_slot,
            publish_actor_ids=publish_actor_ids,
            materialize_results=False,
        )
        return step_batch

    def _prepare_rollout_group(
        self,
        *,
        group_slot: int,
        actor_indices: torch.Tensor,
        current_obs_batch: torch.Tensor,
        aux_states: torch.Tensor,
    ) -> _PreparedRolloutGroup:
        return _PreparedRolloutGroup(
            group_slot=group_slot,
            actor_indices=actor_indices,
            observations=self._select_actor_rows(current_obs_batch, actor_indices),
            aux_tensor=self._select_actor_rows(aux_states, actor_indices),
        )

    def _launch_rollout_forward(
        self,
        *,
        prepared: _PreparedRolloutGroup,
    ) -> _ForwardRolloutGroup:
        stream = self._rollout_streams[prepared.group_slot]
        stream.wait_stream(torch.cuda.current_stream(self._device))

        with torch.cuda.stream(stream), torch.no_grad():
            self._rollout_policy.eval()
            t_fwd_start = time.perf_counter()
            actions, log_probs, values, _hidden, embeddings = self._rollout_policy.forward(
                prepared.observations,
                None,
                aux_tensor=prepared.aux_tensor,
            )
            if self._config.enable_reward_shaping:
                intrinsic_rewards = self._rnd.intrinsic_reward(embeddings)
            else:
                intrinsic_rewards = torch.zeros(
                    (embeddings.shape[0],), dtype=torch.float32, device=embeddings.device
                )
            fwd_ms = (time.perf_counter() - t_fwd_start) * 1000

        return _ForwardRolloutGroup(
            actions=actions,
            embeddings=embeddings,
            log_probs=log_probs,
            values=values,
            intrinsic_rewards=intrinsic_rewards,
            fwd_ms=fwd_ms,
        )

    def _launch_rollout_step(
        self,
        *,
        group_slot: int,
        actor_indices: torch.Tensor,
        actions: torch.Tensor,
        step_id: int,
        publish_actor_ids: tuple[int, ...],
    ) -> tuple[SdfDagTensorStepBatch, float]:
        stream = self._rollout_streams[group_slot]
        actor_id_filter = set(self._rollout_group_actor_ids[group_slot])
        group_publish_actor_ids = tuple(
            actor_id for actor_id in publish_actor_ids if actor_id in actor_id_filter
        )

        with torch.cuda.stream(stream), torch.no_grad():
            t_step_start = time.perf_counter()
            step_batch = self._request_batch_step_tensor_actions(
                actions.detach(),
                step_id,
                actor_indices=actor_indices,
                scratch_slot=group_slot,
                publish_actor_ids=group_publish_actor_ids,
            )
            step_ms = (time.perf_counter() - t_step_start) * 1000

        return step_batch, step_ms

    def _compose_pending_rollout_group(
        self,
        *,
        prepared: _PreparedRolloutGroup,
        forward: _ForwardRolloutGroup,
        step_batch: SdfDagTensorStepBatch,
        step_ms: float,
    ) -> _PendingRolloutGroup:
        return _PendingRolloutGroup(
            group_slot=prepared.group_slot,
            actor_indices=prepared.actor_indices,
            observations=prepared.observations,
            aux_tensor=prepared.aux_tensor,
            actions=forward.actions,
            embeddings=forward.embeddings,
            log_probs=forward.log_probs,
            values=forward.values,
            intrinsic_rewards=forward.intrinsic_rewards,
            step_batch=step_batch,
            fwd_ms=forward.fwd_ms,
            step_ms=step_ms,
        )

    def _finalize_rollout_group(
        self,
        *,
        pending: _PendingRolloutGroup,
        current_obs_batch: torch.Tensor,
        aux_states: torch.Tensor,
        reward_sum_tensor: torch.Tensor,
        reward_ema: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        step_id: int,
        publish_step_telemetry: bool,
        publish_observations: bool,
    ) -> _FinalizeRolloutGroupResult:
        pack_ms = 0.0
        mem_ms = 0.0
        trans_ms = 0.0
        shape_ms = 0.0
        buffer_ms = 0.0
        host_extract_ms = 0.0
        telemetry_ms = 0.0

        t_pack_start = time.perf_counter()
        host_batch = self._pack_rollout_host_batch(
            actions=pending.actions,
            embeddings=pending.embeddings,
            log_probs=pending.log_probs,
            values=pending.values,
            intrinsic_rewards=pending.intrinsic_rewards,
            aux_batch=pending.aux_tensor,
        )
        pack_ms += (time.perf_counter() - t_pack_start) * 1000

        group_size = int(pending.actor_indices.shape[0])
        t_mem_start = time.perf_counter()
        if self._config.enable_episodic_memory:
            normalized_memory_batch = self._memory.normalize_batch_tensor(
                host_batch.embedding_tensor
            )
            loop_similarities, _, loop_temporal_distances = self._memory.query_normalized_batch_tensor(
                normalized_memory_batch
            )
        else:
            normalized_memory_batch = None
            loop_similarities = torch.zeros(
                (group_size,),
                dtype=torch.float32,
                device=host_batch.embedding_tensor.device,
            )
            loop_temporal_distances = torch.zeros(
                (group_size,),
                dtype=torch.float32,
                device=host_batch.embedding_tensor.device,
            )
        mem_ms += (time.perf_counter() - t_mem_start) * 1000

        t_shape_start = time.perf_counter()
        reward_device = host_batch.action_tensor.device
        raw_rewards = pending.step_batch.reward_tensor.to(
            device=reward_device, dtype=torch.float32
        )
        dones = pending.step_batch.done_tensor.to(device=reward_device, dtype=torch.bool)
        intrinsic_rewards = host_batch.intrinsic_reward_tensor.to(
            device=reward_device, dtype=torch.float32
        )
        if self._config.enable_reward_shaping:
            shaped_rewards = self._reward_shaper.shape_batch(
                raw_rewards=raw_rewards,
                dones=dones,
                forward_velocities=host_batch.action_tensor[:, 0],
                angular_velocities=host_batch.action_tensor[:, 3],
                intrinsic_rewards=intrinsic_rewards,
                loop_similarities=loop_similarities,
                loop_temporal_distances=loop_temporal_distances,
            )
            self._reward_shaper.step_batch(group_size)
        else:
            shaped_rewards = raw_rewards
        shape_ms += (time.perf_counter() - t_shape_start) * 1000
        truncateds = pending.step_batch.truncated_tensor.to(device=reward_device, dtype=torch.bool)
        done_or_truncated = dones | truncateds

        t_trans_start = time.perf_counter()
        step_actor_indices = pending.step_batch.env_id_tensor.to(
            device=self._device, dtype=torch.int64
        )
        t_buffer_start = time.perf_counter()
        self._multi_buffers.append_batch(
            observations=pending.observations,
            actions=host_batch.action_tensor,
            log_probs=host_batch.log_prob_tensor,
            values=host_batch.value_tensor,
            rewards=shaped_rewards,
            dones=dones,
            truncateds=truncateds,
            aux_tensors=host_batch.aux_tensor,
            actor_indices=step_actor_indices,
        )
        buffer_ms += (time.perf_counter() - t_buffer_start) * 1000
        reward_sum_tensor.add_(raw_rewards.detach().sum())
        reward_ema = self._update_reward_ema(reward_ema, shaped_rewards)
        self._accumulate_actor_rows(episode_returns, step_actor_indices, shaped_rewards.detach())
        self._accumulate_actor_rows(
            episode_lengths,
            step_actor_indices,
            torch.ones_like(step_actor_indices, dtype=episode_lengths.dtype),
        )
        self._write_actor_rows(
            current_obs_batch, step_actor_indices, pending.step_batch.observation_tensor
        )
        self._write_actor_rows(
            aux_states[:, 0:1], step_actor_indices, shaped_rewards.detach().unsqueeze(1)
        )
        self._write_actor_rows(
            aux_states[:, 1:2], step_actor_indices, loop_similarities.detach().unsqueeze(1)
        )
        self._write_actor_rows(
            aux_states[:, 2:3],
            step_actor_indices,
            host_batch.intrinsic_reward_tensor.detach().unsqueeze(1),
        )
        episode_count = done_or_truncated.sum()

        if publish_step_telemetry:
            done_local_indices = torch.nonzero(done_or_truncated, as_tuple=False).flatten()
            done_actor_indices = (
                step_actor_indices.index_select(0, done_local_indices)
                if int(done_local_indices.numel()) > 0
                else done_local_indices
            )
            if int(done_actor_indices.numel()) > 0:
                selected_episode_indices = self._selected_episode_telemetry_actor_indices(
                    done_actor_indices
                )
                completed_episode_batch = self._extract_completed_episode_host_batch(
                    actor_indices=selected_episode_indices,
                    episode_returns=episode_returns,
                    episode_lengths=episode_lengths,
                )
                done_actor_ids_all = (
                    done_actor_indices.detach().to(device="cpu", dtype=torch.int64).tolist()
                )
                done_episode_ids_all = (
                    pending.step_batch.episode_id_tensor.index_select(0, done_local_indices)
                    .detach()
                    .to(device="cpu", dtype=torch.int64)
                    .tolist()
                )
                done_episode_by_actor = dict(
                    zip(done_actor_ids_all, done_episode_ids_all, strict=True)
                )
            else:
                completed_episode_batch = None
                done_episode_by_actor = {}

            if completed_episode_batch is not None:
                done_actor_ids = completed_episode_batch.actor_id_tensor.tolist()
                for row, actor_id in enumerate(done_actor_ids):
                    t_tel_start = time.perf_counter()
                    self._publish_episode_telemetry(
                        step_id=step_id,
                        episode_id=int(done_episode_by_actor[actor_id]),
                        actor_id=actor_id,
                        episode_return=float(completed_episode_batch.episode_return_tensor[row]),
                        episode_length=int(completed_episode_batch.episode_length_tensor[row]),
                    )
                    telemetry_ms += (time.perf_counter() - t_tel_start) * 1000

        # Zero out accumulated episodic stats inline via masked multiplication (no host sync)
        not_done_mask_f32 = (~done_or_truncated).to(dtype=torch.float32)
        not_done_mask_i32 = (~done_or_truncated).to(dtype=torch.int32)

        episode_returns[step_actor_indices] *= not_done_mask_f32
        episode_lengths[step_actor_indices] *= not_done_mask_i32
        aux_states[step_actor_indices] *= not_done_mask_f32.unsqueeze(1)

        if publish_observations:
            t_tel_start = time.perf_counter()
            self._publish_dashboard_observations(pending.step_batch.published_observations)
            telemetry_ms += (time.perf_counter() - t_tel_start) * 1000
        trans_ms += (time.perf_counter() - t_trans_start) * 1000

        return _FinalizeRolloutGroupResult(
            reward_ema=reward_ema,
            normalized_memory_batch=normalized_memory_batch,
            episode_count=episode_count,
            pack_ms=pack_ms,
            mem_ms=mem_ms,
            trans_ms=trans_ms,
            shape_ms=shape_ms,
            buffer_ms=buffer_ms,
            host_extract_ms=host_extract_ms,
            telemetry_ms=telemetry_ms,
        )

    def _load_initial_observation_batch(
        self,
    ) -> tuple[torch.Tensor, dict[int, DistanceMatrix]]:
        runtime = self._require_tensor_runtime()
        observation_tensors: list[torch.Tensor] = []
        obs_dict: dict[int, DistanceMatrix] = {}
        publish_actor_ids = set(self._observation_publish_actor_ids())
        for actor_id in range(self._n_actors):
            materialize = actor_id in publish_actor_ids
            obs_tensor, published = runtime.reset_tensor(
                0,
                actor_id=actor_id,
                materialize=materialize,
            )
            observation_tensors.append(obs_tensor.to(self._device))
            if published is not None:
                obs_dict[actor_id] = published
        _LOGGER.info(
            "Canonical trainer seeded %d initial observations from direct sdfdag tensor resets.",
            self._n_actors,
        )
        return torch.stack(observation_tensors), obs_dict

    def _publish_observation(self, observation: DistanceMatrix) -> None:
        """Publish a low-volume live observation for dashboard rendering."""
        if not self._config.emit_observation_stream:
            return
        self._last_dashboard_observation = observation
        self._last_dashboard_heartbeat_at = time.perf_counter()
        # Defer serialize() to the telemetry worker thread
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_DISTANCE_MATRIX,
            payload=lambda obs=observation: serialize(obs),  # type: ignore[misc]
            label="observation",
        )

    def _publish_dashboard_observations(self, observations: dict[int, DistanceMatrix]) -> None:
        """Publish selector-visible observations without reusing telemetry sparsity rules."""
        if not observations:
            return
        for actor_id in sorted(observations):
            self._publish_observation(observations[actor_id])

    def _should_publish_dashboard_observation(self) -> bool:
        """Return True when a fresh dashboard frame should be materialized."""
        if not self._config.emit_observation_stream:
            return False
        return (
            time.perf_counter() - self._last_dashboard_heartbeat_at
        ) >= self._dashboard_publish_interval_seconds()

    def _maybe_publish_dashboard_heartbeat(self, *, step_id: int) -> None:
        """Re-publish the last observation while PPO optimization is running."""
        if not self._config.emit_observation_stream:
            return
        observation = self._last_dashboard_observation
        if observation is None:
            return
        now_perf = time.perf_counter()
        if (
            now_perf - self._last_dashboard_heartbeat_at
        ) < self._dashboard_publish_interval_seconds():
            return

        heartbeat = replace(
            observation,
            step_id=step_id,
            timestamp=time.time(),
        )
        self._last_dashboard_observation = heartbeat
        self._last_dashboard_heartbeat_at = now_perf
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_DISTANCE_MATRIX,
            payload=lambda obs=heartbeat: serialize(obs),  # type: ignore[misc]
            label="dashboard heartbeat",
        )

    def _dashboard_publish_interval_seconds(self) -> float:
        """Return the passive selected-actor dashboard publish interval."""
        return 1.0 / max(float(self._config.dashboard_observation_hz), 1.0)

    def _observation_publish_actor_ids(self) -> tuple[int, ...]:
        """Return the actor ids whose low-rate dashboard observations may be published."""
        if not self._config.emit_observation_stream:
            return ()
        return (0,)

    def _should_publish_actor_telemetry(self, actor_id: int) -> bool:
        return actor_id == 0

    def _publish_step_telemetry(
        self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any
    ) -> None:
        if not self._config.emit_training_telemetry:
            return
        if not self._should_publish_actor_telemetry(actor_id):
            return
        p = np.array([kwargs.get(k, 0.0) for k in _STEP_TELEMETRY_FIELDS], dtype=np.float32)
        event = TelemetryEvent(
            event_type="actor.training.ppo.step",
            episode_id=episode_id,
            env_id=actor_id,
            step_id=step_id,
            payload=p,
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="step telemetry",
        )
        self._emit_metrics_record(
            "training_step",
            {
                "step_id": step_id,
                "episode_id": episode_id,
                "actor_id": actor_id,
                **{field: float(kwargs.get(field, 0.0)) for field in _STEP_TELEMETRY_FIELDS},
            },
        )

    def _publish_update_telemetry(
        self, step_id: int, reward_ema: float, metrics: PpoMetrics
    ) -> None:
        if not self._config.emit_training_telemetry:
            return
        update_payload = {
            "reward_ema": reward_ema,
            "policy_loss": metrics.policy_loss,
            "value_loss": metrics.value_loss,
            "entropy": metrics.entropy,
            "approx_kl": metrics.approx_kl,
            "clip_fraction": metrics.clip_fraction,
            "total_loss": metrics.total_loss,
            "rnd_loss": metrics.rnd_loss,
            "beta": self._reward_shaper.beta,
        }
        p = np.array(
            [update_payload[field] for field in _UPDATE_TELEMETRY_FIELDS], dtype=np.float32
        )
        event = TelemetryEvent(
            event_type="actor.training.ppo.update",
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
            label="update telemetry",
        )
        self._emit_metrics_record(
            "training_update",
            {"step_id": step_id, **update_payload},
        )

    def _publish_episode_telemetry(
        self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any
    ) -> None:
        if not self._config.emit_training_telemetry:
            return
        if not self._should_publish_actor_telemetry(actor_id):
            return
        p = np.array([kwargs["episode_return"], float(kwargs["episode_length"])], dtype=np.float32)
        event = TelemetryEvent(
            event_type="actor.training.ppo.episode",
            episode_id=episode_id,
            env_id=actor_id,
            step_id=step_id,
            payload=p,
            timestamp=time.time(),
        )
        _enqueue_telemetry(
            self._telemetry_queue,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="episode telemetry",
        )
        self._emit_metrics_record(
            "training_episode",
            {
                "step_id": step_id,
                "episode_id": episode_id,
                "actor_id": actor_id,
                "episode_return": float(kwargs["episode_return"]),
                "episode_length": int(kwargs["episode_length"]),
            },
        )

    def _publish_perf_telemetry(self, *, step_id: int, **kwargs: Any) -> None:
        if not self._config.emit_perf_telemetry:
            return
        p = np.array([kwargs.get(k, 0.0) for k in _PERF_TELEMETRY_FIELDS], dtype=np.float32)
        event = TelemetryEvent(
            event_type="actor.training.ppo.perf",
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
            label="perf telemetry",
        )
        self._emit_metrics_record(
            "training_perf",
            build_phase_metrics_payload(
                "rollout_heartbeat",
                elapsed_ms=float(kwargs.get("tick_total_ms", 0.0)),
                step_id=step_id,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    field: float(kwargs.get(field, 0.0)) for field in _PERF_TELEMETRY_FIELDS
                },
            ),
        )

    def _publish_scene_telemetry(self, *, step_id: int) -> None:
        """Publish current scene name so the dashboard can display it."""
        if not self._config.emit_perf_telemetry:
            return
        runtime = self._runtime
        scene_name: str = ""
        if runtime is not None:
            scene_name = getattr(runtime, "current_scene_name", "")
        if not scene_name:
            return
        payload = np.array([float(ord(c)) for c in scene_name], dtype=np.float32)
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

    def _publish_runtime_perf(self, *, step_id: int) -> None:
        """Publish coarse runtime perf from the canonical sdfdag backend."""
        if not self._config.emit_perf_telemetry:
            return

        try:
            snapshot = self._require_runtime().perf_snapshot()
        except Exception as exc:  # pragma: no cover - defensive runtime path
            _LOGGER.warning("Skipping runtime perf snapshot after backend failure: %s", exc)
            return
        payload = np.array(
            [
                snapshot.sps,
                snapshot.last_batch_step_ms,
                snapshot.ema_batch_step_ms,
                snapshot.avg_batch_step_ms,
                snapshot.avg_actor_step_ms,
                float(snapshot.total_batches),
                float(snapshot.total_actor_steps),
            ],
            dtype=np.float32,
        )
        event = TelemetryEvent(
            event_type="environment.sdfdag.perf",
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
            label="runtime perf telemetry",
        )
        self._emit_metrics_record(
            "runtime_perf",
            build_phase_metrics_payload(
                "runtime_heartbeat",
                elapsed_ms=float(snapshot.last_batch_step_ms),
                step_id=step_id,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata=dict(
                    zip(
                        _RUNTIME_PERF_FIELDS,
                        (
                            float(snapshot.sps),
                            float(snapshot.last_batch_step_ms),
                            float(snapshot.ema_batch_step_ms),
                            float(snapshot.avg_batch_step_ms),
                            float(snapshot.avg_actor_step_ms),
                            float(snapshot.total_batches),
                            float(snapshot.total_actor_steps),
                        ),
                        strict=True,
                    )
                ),
            ),
        )

    def _require_runtime(self) -> CanonicalRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Canonical sdfdag runtime is not configured")
        return runtime

    def _require_tensor_runtime(self) -> CanonicalRuntime:
        runtime = self._require_runtime()
        if not hasattr(runtime, "reset_tensor") or not hasattr(
            runtime, "batch_step_tensor_actions"
        ):
            raise RuntimeError(
                "Canonical PPO training requires tensor-native runtime seams: reset_tensor() and batch_step_tensor_actions().",
            )
        return runtime

    def _update_reward_ema(
        self, reward_ema: torch.Tensor, batch_rewards: torch.Tensor
    ) -> torch.Tensor:
        """Apply the per-actor EMA update order without synchronizing rollout GPU state to host."""
        rewards = batch_rewards.detach().to(device=self._device, dtype=torch.float32).reshape(-1)
        count = int(rewards.shape[0])
        if count == 0:
            return reward_ema
        decay = 0.99
        alpha = 0.01
        exponents = torch.arange(count - 1, -1, -1, device=rewards.device, dtype=torch.float32)
        weights = alpha * torch.pow(torch.full((count,), decay, device=rewards.device), exponents)
        decay_tensor = torch.tensor(decay**count, dtype=torch.float32, device=rewards.device)
        return decay_tensor * reward_ema + (weights * rewards).sum()

    def _pack_rollout_host_batch(
        self,
        *,
        actions: torch.Tensor,
        embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        aux_batch: torch.Tensor,
    ) -> _RolloutHostBatch:
        """Detach rollout tensors while keeping sparse telemetry actions on device."""
        action_tensor = actions.detach()
        embedding_tensor = embeddings.detach()
        aux_tensor = aux_batch.detach()

        return _RolloutHostBatch(
            embedding_tensor=embedding_tensor,
            action_tensor=action_tensor,
            aux_tensor=aux_tensor,
            log_prob_tensor=log_probs.detach(),
            value_tensor=values.detach(),
            intrinsic_reward_tensor=intrinsic_rewards.detach(),
        )

    def _extract_host_rollout_scalars(
        self,
        *,
        raw_rewards: torch.Tensor,
        shaped_rewards: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        loop_similarities: torch.Tensor,
        done_flags: torch.Tensor,
        actor_indices: torch.Tensor | None,
    ) -> _RolloutScalarBatch:
        """Select sparse telemetry tensors while keeping them on device."""
        if actor_indices is not None and int(actor_indices.numel()) > 0:
            return _RolloutScalarBatch(
                raw_reward_tensor=raw_rewards.index_select(0, actor_indices).detach(),
                shaped_reward_tensor=shaped_rewards.index_select(0, actor_indices).detach(),
                intrinsic_reward_tensor=intrinsic_rewards.index_select(0, actor_indices).detach(),
                loop_similarity_tensor=loop_similarities.index_select(0, actor_indices).detach(),
                done_tensor=done_flags.index_select(0, actor_indices).detach(),
            )

        return _RolloutScalarBatch(
            raw_reward_tensor=None,
            shaped_reward_tensor=None,
            intrinsic_reward_tensor=None,
            loop_similarity_tensor=None,
            done_tensor=None,
        )

    def _materialize_step_telemetry_host_batch(
        self,
        *,
        scalar_batch: _RolloutScalarBatch,
        action_tensor: torch.Tensor,
        reward_component_tensor: torch.Tensor | None,
        actor_indices: torch.Tensor,
    ) -> _StepTelemetryHostBatch:
        """Batch one sparse telemetry host transfer immediately before publication."""
        if (
            scalar_batch.raw_reward_tensor is None
            or scalar_batch.shaped_reward_tensor is None
            or scalar_batch.intrinsic_reward_tensor is None
            or scalar_batch.loop_similarity_tensor is None
            or scalar_batch.done_tensor is None
            or int(actor_indices.numel()) == 0
        ):
            return _StepTelemetryHostBatch(
                raw_reward_tensor=None,
                shaped_reward_tensor=None,
                intrinsic_reward_tensor=None,
                loop_similarity_tensor=None,
                done_tensor=None,
                forward_velocity_tensor=None,
                yaw_velocity_tensor=None,
                reward_component_tensor=None,
            )

        raw_reward_tensor = scalar_batch.raw_reward_tensor
        shaped_reward_tensor = scalar_batch.shaped_reward_tensor
        intrinsic_reward_tensor = scalar_batch.intrinsic_reward_tensor
        loop_similarity_tensor = scalar_batch.loop_similarity_tensor
        done_tensor = scalar_batch.done_tensor
        if (
            raw_reward_tensor is None
            or shaped_reward_tensor is None
            or intrinsic_reward_tensor is None
            or loop_similarity_tensor is None
            or done_tensor is None
        ):
            raise RuntimeError("step telemetry scalar batch unexpectedly missing after validation")

        column_tensors = [
            raw_reward_tensor.to(dtype=torch.float32),
            shaped_reward_tensor.to(dtype=torch.float32),
            intrinsic_reward_tensor.to(dtype=torch.float32),
            loop_similarity_tensor.to(dtype=torch.float32),
            done_tensor.to(dtype=torch.float32),
            action_tensor.index_select(0, actor_indices)[:, 0].detach().to(dtype=torch.float32),
            action_tensor.index_select(0, actor_indices)[:, 3].detach().to(dtype=torch.float32),
        ]
        component_count = 0
        if reward_component_tensor is not None:
            selected_components = (
                reward_component_tensor.index_select(0, actor_indices)
                .detach()
                .to(dtype=torch.float32)
            )
            component_count = int(selected_components.shape[1])
            column_tensors.extend(selected_components[:, idx] for idx in range(component_count))

        packed = torch.stack(column_tensors, dim=1).detach().to(device="cpu", dtype=torch.float32)
        reward_components_cpu = packed[:, 7 : 7 + component_count] if component_count > 0 else None
        return _StepTelemetryHostBatch(
            raw_reward_tensor=packed[:, 0],
            shaped_reward_tensor=packed[:, 1],
            intrinsic_reward_tensor=packed[:, 2],
            loop_similarity_tensor=packed[:, 3],
            done_tensor=packed[:, 4] > 0.5,
            forward_velocity_tensor=packed[:, 5],
            yaw_velocity_tensor=packed[:, 6],
            reward_component_tensor=reward_components_cpu,
        )

    def _extract_completed_episode_host_batch(
        self,
        *,
        actor_indices: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
    ) -> _CompletedEpisodeHostBatch | None:
        """Batch sparse completed-episode host extraction into one packed transfer."""
        selected_indices = actor_indices.to(
            device=episode_returns.device, dtype=torch.int64
        ).flatten()
        if int(selected_indices.numel()) == 0:
            return None

        packed = (
            torch.stack(
                (
                    selected_indices.to(dtype=torch.float32),
                    episode_returns.index_select(0, selected_indices),
                    episode_lengths.index_select(0, selected_indices).to(dtype=torch.float32),
                ),
                dim=1,
            )
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
        return _CompletedEpisodeHostBatch(
            actor_id_tensor=packed[:, 0].to(dtype=torch.int64),
            episode_return_tensor=packed[:, 1],
            episode_length_tensor=packed[:, 2].to(dtype=torch.int32),
        )

    def _selected_episode_telemetry_actor_indices(
        self, done_indices: torch.Tensor
    ) -> torch.Tensor:
        """Return only the done actors whose episode telemetry will be published."""
        if not self._config.emit_training_telemetry:
            return done_indices[:0]
        return done_indices[done_indices == 0]

    def _selected_step_telemetry_actor_indices(self) -> torch.Tensor:
        """Return only the actors whose per-step telemetry will be mirrored to CPU."""
        if not self._config.emit_training_telemetry:
            return torch.empty((0,), dtype=torch.int64, device=self._device)
        return torch.tensor([0], dtype=torch.int64, device=self._device)

    def _active_actor_local_indices(
        self,
        active_actor_indices: torch.Tensor,
        selected_actor_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Map selected global actor ids onto local batch rows for one rollout step."""
        if selected_actor_indices is None or int(selected_actor_indices.numel()) == 0:
            return torch.empty((0,), dtype=torch.int64, device=active_actor_indices.device)

        matches = selected_actor_indices.unsqueeze(1).eq(active_actor_indices.unsqueeze(0))
        present_mask = matches.any(dim=1)
        if not bool(present_mask.any()):
            return torch.empty((0,), dtype=torch.int64, device=active_actor_indices.device)
        return matches[present_mask].to(dtype=torch.int64).argmax(dim=1)

    def _select_actor_rows(
        self, tensor: torch.Tensor, actor_indices: torch.Tensor
    ) -> torch.Tensor:
        return tensor.index_select(0, actor_indices)

    def _write_actor_rows(
        self, tensor: torch.Tensor, actor_indices: torch.Tensor, values: torch.Tensor
    ) -> None:
        tensor[actor_indices] = values

    def _accumulate_actor_rows(
        self, tensor: torch.Tensor, actor_indices: torch.Tensor, values: torch.Tensor
    ) -> None:
        tensor[actor_indices] = tensor.index_select(0, actor_indices) + values

    def _zero_actor_rows(self, tensor: torch.Tensor, actor_indices: torch.Tensor) -> None:
        if int(actor_indices.numel()) == 0:
            return
        tensor.index_fill_(0, actor_indices, 0)

    def _prime_rollout_group(
        self,
        *,
        group_slot: int,
        actor_indices: torch.Tensor,
        current_obs_batch: torch.Tensor,
        aux_states: torch.Tensor,
    ) -> _PrimedRolloutGroup:
        prepared = self._prepare_rollout_group(
            group_slot=group_slot,
            actor_indices=actor_indices,
            current_obs_batch=current_obs_batch,
            aux_states=aux_states,
        )
        forward = self._launch_rollout_forward(prepared=prepared)
        return _PrimedRolloutGroup(prepared=prepared, forward=forward)

    def train(
        self,
        total_steps: int,
        *,
        log_every: int = 100,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "checkpoints",
    ) -> PpoTrainingMetrics:
        n = self._n_actors
        rl, ep, mb, sl = (
            self._config.rollout_length,
            self._config.ppo_epochs,
            self._config.minibatch_size,
            self._config.bptt_len,
        )
        # Restore counters from instance state (populated by load_training_state)
        r_ema = torch.tensor(self._reward_ema, dtype=torch.float32, device=self._device)
        ep_cnt = torch.tensor(self._total_episodes, dtype=torch.int32, device=self._device)
        step_id = self._total_sim_steps
        unbounded = total_steps <= 0
        reward_sum_tensor = torch.zeros((), dtype=torch.float32, device=self._device)
        episode_returns = torch.zeros((n,), dtype=torch.float32, device=self._device)
        episode_lengths = torch.zeros((n,), dtype=torch.int32, device=self._device)
        aux_states = torch.zeros((n, 3), dtype=torch.float32, device=self._device)
        current_obs_batch, obs_dict = self._load_initial_observation_batch()
        self._publish_dashboard_observations(obs_dict)

        t_start = time.perf_counter()
        self._training_wall_start = t_start
        (
            acc_fwd,
            acc_pack,
            acc_step,
            acc_mem,
            acc_trans,
            acc_shape,
            acc_madd,
            acc_buf,
            acc_host,
            acc_tel,
            acc_tick,
            t_cnt,
        ) = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
        )

        last_checkpoint_step = 0

        # ── Staggered per-actor PPO: compute initial targets ──
        stagger_active = self._config.stagger_ppo and n > 1
        if stagger_active:
            stagger_step = max(1, rl // n)
            actor_rollout_targets = torch.tensor(
                [rl - i * stagger_step for i in range(n)],
                dtype=torch.int64,
                device=self._device,
            )
            _LOGGER.info(
                "Staggered PPO enabled: initial per-actor targets=%s stagger_step=%d",
                actor_rollout_targets.tolist(),
                stagger_step,
            )
        else:
            actor_rollout_targets = None

        while unbounded or step_id < total_steps:
            rollout_actor_steps = 0
            primed_group: _PrimedRolloutGroup | None = None
            if self._rollout_group_count > 1:
                primed_group = self._prime_rollout_group(
                    group_slot=1,
                    actor_indices=self._rollout_group_indices[1],
                    current_obs_batch=current_obs_batch,
                    aux_states=aux_states,
                )
            # Rollout Phase
            for rollout_step in range(rl):
                if (not unbounded) and step_id >= total_steps:
                    break
                t_tick = time.perf_counter()
                fwd_ms = 0.0
                pack_ms = 0.0
                step_ms = 0.0
                mem_ms = 0.0
                trans_ms = 0.0
                shape_ms = 0.0
                memory_add_ms = 0.0
                buffer_ms = 0.0
                host_extract_ms = 0.0
                telemetry_ms = 0.0
                publish_actor_ids: tuple[int, ...] = ()
                # Tier 1 optimization: Zero step telemetry in the hot loop.
                # Per-step extraction is too heavy. All necessary metrics
                # are deferred to the PPO Epoch update boundary.
                publish_step_telemetry = False
                publish_observations = self._should_publish_dashboard_observation()
                if publish_observations:
                    publish_actor_ids = self._observation_publish_actor_ids()

                deferred_memory_batches: list[torch.Tensor] = []
                if self._rollout_group_count == 1:
                    prepared = self._prepare_rollout_group(
                        group_slot=0,
                        actor_indices=self._rollout_group_indices[0],
                        current_obs_batch=current_obs_batch,
                        aux_states=aux_states,
                    )
                    forward = self._launch_rollout_forward(prepared=prepared)
                    step_batch, group_step_ms = self._launch_rollout_step(
                        group_slot=prepared.group_slot,
                        actor_indices=prepared.actor_indices,
                        actions=forward.actions.detach(),
                        step_id=step_id,
                        publish_actor_ids=publish_actor_ids,
                    )
                    torch.cuda.current_stream(self._device).wait_stream(
                        self._rollout_streams[prepared.group_slot]
                    )
                    pending = self._compose_pending_rollout_group(
                        prepared=prepared,
                        forward=forward,
                        step_batch=step_batch,
                        step_ms=group_step_ms,
                    )
                    finalized = self._finalize_rollout_group(
                        pending=pending,
                        current_obs_batch=current_obs_batch,
                        aux_states=aux_states,
                        reward_sum_tensor=reward_sum_tensor,
                        reward_ema=r_ema,
                        episode_returns=episode_returns,
                        episode_lengths=episode_lengths,
                        step_id=step_id,
                        publish_step_telemetry=publish_step_telemetry,
                        publish_observations=publish_observations,
                    )
                    r_ema = finalized.reward_ema
                    if finalized.normalized_memory_batch is not None:
                        deferred_memory_batches.append(finalized.normalized_memory_batch)
                    ep_cnt += finalized.episode_count
                    fwd_ms += pending.fwd_ms
                    step_ms += pending.step_ms
                    pack_ms += finalized.pack_ms
                    mem_ms += finalized.mem_ms
                    trans_ms += finalized.trans_ms
                    shape_ms += finalized.shape_ms
                    buffer_ms += finalized.buffer_ms
                    host_extract_ms += finalized.host_extract_ms
                    telemetry_ms += finalized.telemetry_ms
                else:
                    if primed_group is None:
                        raise RuntimeError(
                            "Expected a primed rollout group for grouped overlap execution"
                        )

                    prepared_group_a = self._prepare_rollout_group(
                        group_slot=0,
                        actor_indices=self._rollout_group_indices[0],
                        current_obs_batch=current_obs_batch,
                        aux_states=aux_states,
                    )
                    forward_group_a = self._launch_rollout_forward(prepared=prepared_group_a)
                    step_batch_b, step_ms_b = self._launch_rollout_step(
                        group_slot=primed_group.prepared.group_slot,
                        actor_indices=primed_group.prepared.actor_indices,
                        actions=primed_group.forward.actions.detach(),
                        step_id=step_id,
                        publish_actor_ids=publish_actor_ids,
                    )

                    torch.cuda.current_stream(self._device).wait_stream(
                        self._rollout_streams[prepared_group_a.group_slot]
                    )
                    torch.cuda.current_stream(self._device).wait_stream(
                        self._rollout_streams[primed_group.prepared.group_slot]
                    )

                    pending_group_b = self._compose_pending_rollout_group(
                        prepared=primed_group.prepared,
                        forward=primed_group.forward,
                        step_batch=step_batch_b,
                        step_ms=step_ms_b,
                    )
                    finalized_group_b = self._finalize_rollout_group(
                        pending=pending_group_b,
                        current_obs_batch=current_obs_batch,
                        aux_states=aux_states,
                        reward_sum_tensor=reward_sum_tensor,
                        reward_ema=r_ema,
                        episode_returns=episode_returns,
                        episode_lengths=episode_lengths,
                        step_id=step_id,
                        publish_step_telemetry=publish_step_telemetry,
                        publish_observations=publish_observations,
                    )
                    r_ema = finalized_group_b.reward_ema
                    if finalized_group_b.normalized_memory_batch is not None:
                        deferred_memory_batches.append(finalized_group_b.normalized_memory_batch)
                    ep_cnt += finalized_group_b.episode_count

                    has_future_tick = (rollout_step + 1) < rl and (
                        unbounded or (step_id + n) < total_steps
                    )
                    if has_future_tick:
                        primed_group = self._prime_rollout_group(
                            group_slot=1,
                            actor_indices=self._rollout_group_indices[1],
                            current_obs_batch=current_obs_batch,
                            aux_states=aux_states,
                        )
                    else:
                        primed_group = None

                    step_batch_a, step_ms_a = self._launch_rollout_step(
                        group_slot=prepared_group_a.group_slot,
                        actor_indices=prepared_group_a.actor_indices,
                        actions=forward_group_a.actions.detach(),
                        step_id=step_id,
                        publish_actor_ids=publish_actor_ids,
                    )

                    torch.cuda.current_stream(self._device).wait_stream(
                        self._rollout_streams[prepared_group_a.group_slot]
                    )
                    if has_future_tick:
                        assert primed_group is not None
                        torch.cuda.current_stream(self._device).wait_stream(
                            self._rollout_streams[primed_group.prepared.group_slot]
                        )

                    pending_group_a = self._compose_pending_rollout_group(
                        prepared=prepared_group_a,
                        forward=forward_group_a,
                        step_batch=step_batch_a,
                        step_ms=step_ms_a,
                    )
                    finalized_group_a = self._finalize_rollout_group(
                        pending=pending_group_a,
                        current_obs_batch=current_obs_batch,
                        aux_states=aux_states,
                        reward_sum_tensor=reward_sum_tensor,
                        reward_ema=r_ema,
                        episode_returns=episode_returns,
                        episode_lengths=episode_lengths,
                        step_id=step_id,
                        publish_step_telemetry=publish_step_telemetry,
                        publish_observations=publish_observations,
                    )
                    r_ema = finalized_group_a.reward_ema
                    if finalized_group_a.normalized_memory_batch is not None:
                        deferred_memory_batches.append(finalized_group_a.normalized_memory_batch)
                    ep_cnt += finalized_group_a.episode_count

                    fwd_ms += pending_group_b.fwd_ms + pending_group_a.fwd_ms
                    step_ms += pending_group_b.step_ms + pending_group_a.step_ms
                    pack_ms += finalized_group_b.pack_ms + finalized_group_a.pack_ms
                    mem_ms += finalized_group_b.mem_ms + finalized_group_a.mem_ms
                    trans_ms += finalized_group_b.trans_ms + finalized_group_a.trans_ms
                    shape_ms += finalized_group_b.shape_ms + finalized_group_a.shape_ms
                    buffer_ms += finalized_group_b.buffer_ms + finalized_group_a.buffer_ms
                    host_extract_ms += (
                        finalized_group_b.host_extract_ms + finalized_group_a.host_extract_ms
                    )
                    telemetry_ms += finalized_group_b.telemetry_ms + finalized_group_a.telemetry_ms

                if self._config.enable_episodic_memory:
                    for normalized_memory_batch in deferred_memory_batches:
                        t_memory_add_start = time.perf_counter()
                        self._memory.add_normalized_batch_tensor(normalized_memory_batch)
                        memory_add_ms += (time.perf_counter() - t_memory_add_start) * 1000

                step_id += n
                self._total_sim_steps += n
                rollout_actor_steps += n

                tick_ms = (time.perf_counter() - t_tick) * 1000
                acc_fwd += fwd_ms
                acc_pack += pack_ms
                acc_step += step_ms
                acc_mem += mem_ms
                acc_trans += trans_ms
                acc_shape += shape_ms
                acc_madd += memory_add_ms
                acc_buf += buffer_ms
                acc_host += host_extract_ms
                acc_tel += telemetry_ms
                acc_tick += tick_ms
                t_cnt += 1

                # ── Staggered per-actor PPO trigger ──
                if stagger_active and actor_rollout_targets is not None:
                    step_counts = self._multi_buffers.get_actor_step_counts()
                    ready_mask = step_counts >= actor_rollout_targets
                    if ready_mask.any():
                        ready_ids = ready_mask.nonzero(as_tuple=False).flatten().to(
                            device=self._device, dtype=torch.int64
                        )
                        # LR anneal before PPO (same schedule as monolithic path)
                        if self._config.lr_schedule_steps > 0:
                            frac = min(1.0, step_id / self._config.lr_schedule_steps)
                            lr_init = self._config.learning_rate
                            lr_final = self._config.learning_rate_final
                            cosine_lr = lr_final + 0.5 * (lr_init - lr_final) * (
                                1.0 + math.cos(math.pi * frac)
                            )
                            rnd_lr_init = self._config.rnd_learning_rate
                            rnd_lr_final = self._config.rnd_learning_rate_final
                            cosine_rnd_lr = rnd_lr_final + 0.5 * (rnd_lr_init - rnd_lr_final) * (
                                1.0 + math.cos(math.pi * frac)
                            )
                            self._learner.set_learning_rate(cosine_lr, rnd_lr=cosine_rnd_lr)
                        self._run_staggered_ppo_update(
                            actor_ids=ready_ids,
                            multi_buffer=self._multi_buffers,
                            obs_batch=current_obs_batch,
                            aux_batch=aux_states,
                            ppo_epochs=ep,
                            minibatch_size=mb,
                            seq_len=sl,
                        )
                        # After first PPO, all actors use full rollout length
                        actor_rollout_targets[ready_ids] = rl
                        if self._last_opt_metrics:
                            reward_ema_value = float(
                                r_ema.detach().to(device="cpu", dtype=torch.float32)
                            )
                            self._publish_update_telemetry(
                                step_id, reward_ema_value, self._last_opt_metrics
                            )
                        if _should_save_checkpoint(
                            step_id, last_checkpoint_step, checkpoint_every
                        ):
                            self.save_training_state(
                                Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt"
                            )
                            last_checkpoint_step = step_id

                if log_every > 0 and step_id % log_every < n:
                    now_perf = time.perf_counter()
                    sps = log_every / max(
                        0.001, now_perf - getattr(self, "_last_log_time", t_start)
                    )
                    self._last_log_time = now_perf
                    zw = self._sim_steps_during_opt / max(1, self._total_sim_steps)
                    reward_ema_value = float(r_ema.detach().to(device="cpu", dtype=torch.float32))
                    ep_cnt_val = int(ep_cnt.item())
                    # Sync instance state at coarse log cadence for checkpoint enrichment
                    self._reward_ema = reward_ema_value
                    self._total_episodes = ep_cnt_val
                    _LOGGER.info(
                        "[step %d] reward_ema=%.4f episodes=%d | sps=%.1f zw=%.1f%% | "
                        "fwd=%.1fms pack=%.1fms env=%.1fms mem=%.1fms trans=%.1fms "
                        "(shape=%.1fms madd=%.1fms buf=%.1fms host=%.1fms tele=%.1fms)",
                        step_id,
                        reward_ema_value,
                        ep_cnt_val,
                        sps,
                        zw * 100,
                        fwd_ms,
                        pack_ms,
                        step_ms,
                        mem_ms,
                        trans_ms,
                        shape_ms,
                        memory_add_ms,
                        buffer_ms,
                        host_extract_ms,
                        telemetry_ms,
                    )
                    self._publish_perf_telemetry(
                        step_id=step_id,
                        sps=sps,
                        forward_pass_ms=fwd_ms,
                        batch_step_ms=step_ms,
                        memory_query_ms=mem_ms,
                        transition_ms=trans_ms,
                        tick_total_ms=tick_ms,
                        zero_wait_ratio=zw,
                        ppo_update_ms=self._last_opt_duration_ms,
                        host_extract_ms=host_extract_ms,
                        telemetry_publish_ms=telemetry_ms,
                        n_actors=float(self._n_actors),
                    )
                    self._publish_runtime_perf(step_id=step_id)
                    self._publish_scene_telemetry(step_id=step_id)
                    if sps < _SOFT_WARN_MIN_SPS:
                        _LOGGER.warning(
                            "Soft stall monitor: SPS below target (%.1f < %.1f) at step=%d",
                            sps,
                            _SOFT_WARN_MIN_SPS,
                            step_id,
                        )
                    if zw > _SOFT_WARN_MAX_ZERO_WAIT:
                        _LOGGER.warning(
                            "Soft stall monitor: zero-wait ratio elevated (%.1f%% > %.1f%%) at step=%d",
                            zw * 100,
                            _SOFT_WARN_MAX_ZERO_WAIT * 100,
                            step_id,
                        )

            if rollout_actor_steps <= 0:
                break

            # --- Rollout Loop Finished ---
            if stagger_active:
                # Staggered PPO updates already ran inside the rollout loop.
                # Only handle checkpoint if not already saved by a stagger trigger.
                if _should_save_checkpoint(step_id, last_checkpoint_step, checkpoint_every):
                    self.save_training_state(Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt")
                    last_checkpoint_step = step_id
            else:
                # Monolithic PPO: Cosine LR annealing + single PPO update for all actors.
                if self._config.lr_schedule_steps > 0:
                    frac = min(1.0, step_id / self._config.lr_schedule_steps)
                    lr_init = self._config.learning_rate
                    lr_final = self._config.learning_rate_final
                    cosine_lr = lr_final + 0.5 * (lr_init - lr_final) * (1.0 + math.cos(math.pi * frac))
                    rnd_lr_init = self._config.rnd_learning_rate
                    rnd_lr_final = self._config.rnd_learning_rate_final
                    cosine_rnd_lr = rnd_lr_final + 0.5 * (rnd_lr_init - rnd_lr_final) * (1.0 + math.cos(math.pi * frac))
                    self._learner.set_learning_rate(cosine_lr, rnd_lr=cosine_rnd_lr)
                self._run_ppo_update(
                    multi_buffer=self._multi_buffers,
                    obs_batch=current_obs_batch,
                    aux_batch=aux_states,
                    ppo_epochs=ep,
                    minibatch_size=mb,
                    seq_len=sl,
                )
                if self._last_opt_metrics:
                    reward_ema_value = float(r_ema.detach().to(device="cpu", dtype=torch.float32))
                    self._publish_update_telemetry(step_id, reward_ema_value, self._last_opt_metrics)

                # Save when we've advanced at least one checkpoint interval since the last save.
                # This remains correct even when rollout boundaries do not land on exact modulo values.
                if _should_save_checkpoint(step_id, last_checkpoint_step, checkpoint_every):
                    self.save_training_state(Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt")
                    last_checkpoint_step = step_id

        # ── Stagger drain: flush any remaining per-actor data ──
        if stagger_active:
            drain_counts = self._multi_buffers.get_actor_step_counts()
            for count_val in drain_counts.unique().tolist():
                if count_val == 0:
                    continue
                actors_at_count = (drain_counts == count_val).nonzero(as_tuple=False).flatten().to(
                    device=self._device, dtype=torch.int64
                )
                if self._config.lr_schedule_steps > 0:
                    frac = min(1.0, step_id / self._config.lr_schedule_steps)
                    lr_init_d = self._config.learning_rate
                    lr_final_d = self._config.learning_rate_final
                    cosine_lr_d = lr_final_d + 0.5 * (lr_init_d - lr_final_d) * (
                        1.0 + math.cos(math.pi * frac)
                    )
                    rnd_lr_init_d = self._config.rnd_learning_rate
                    rnd_lr_final_d = self._config.rnd_learning_rate_final
                    cosine_rnd_lr_d = rnd_lr_final_d + 0.5 * (rnd_lr_init_d - rnd_lr_final_d) * (
                        1.0 + math.cos(math.pi * frac)
                    )
                    self._learner.set_learning_rate(cosine_lr_d, rnd_lr=cosine_rnd_lr_d)
                self._run_staggered_ppo_update(
                    actor_ids=actors_at_count,
                    multi_buffer=self._multi_buffers,
                    obs_batch=current_obs_batch,
                    aux_batch=aux_states,
                    ppo_epochs=ep,
                    minibatch_size=mb,
                    seq_len=sl,
                )

        reward_mean = float(reward_sum_tensor.item()) / max(1, step_id)
        reward_ema_value = float(r_ema.detach().to(device="cpu", dtype=torch.float32))
        wall_clock_seconds = time.perf_counter() - t_start
        # Final sync of instance state for checkpoint enrichment
        self._reward_ema = reward_ema_value
        self._total_episodes = int(ep_cnt.item())
        self._emit_metrics_record(
            "training_summary",
            build_phase_metrics_payload(
                "training_summary",
                elapsed_ms=wall_clock_seconds * 1000.0,
                step_id=step_id,
                cuda_device=self._device,
                include_resources=self._config.attach_resource_snapshots,
                metadata={
                    "total_steps": step_id,
                    "episodes": int(ep_cnt.item()),
                    "reward_mean": reward_mean,
                    "reward_ema": reward_ema_value,
                    "beta_final": self._reward_shaper.beta,
                    "sps_mean": step_id / max(0.001, wall_clock_seconds),
                    "ppo_update_ms_mean": self._opt_duration_acc
                    / max(1, self._opt_duration_count),
                },
            ),
        )
        return PpoTrainingMetrics(
            total_steps=step_id,
            episodes=int(ep_cnt.item()),
            reward_mean=reward_mean,
            reward_ema=reward_ema_value,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            rnd_loss=0.0,
            intrinsic_reward_mean=0.0,
            loop_detections=0,
            beta_final=self._reward_shaper.beta,
            sps_mean=(step_id / max(0.001, time.perf_counter() - t_start)),
            forward_pass_ms_mean=(acc_fwd / max(1, t_cnt)),
            action_pack_ms_mean=(acc_pack / max(1, t_cnt)),
            batch_step_ms_mean=(acc_step / max(1, t_cnt)),
            memory_query_ms_mean=(acc_mem / max(1, t_cnt)),
            transition_ms_mean=(acc_trans / max(1, t_cnt)),
            reward_shape_ms_mean=(acc_shape / max(1, t_cnt)),
            memory_add_ms_mean=(acc_madd / max(1, t_cnt)),
            buffer_append_ms_mean=(acc_buf / max(1, t_cnt)),
            host_extract_ms_mean=(acc_host / max(1, t_cnt)),
            telemetry_publish_ms_mean=(acc_tel / max(1, t_cnt)),
            tick_total_ms_mean=(acc_tick / max(1, t_cnt)),
            ppo_update_ms_mean=(self._opt_duration_acc / max(1, self._opt_duration_count)),
            ppo_inference_ms_mean=(self._opt_inference_acc / max(1, self._opt_duration_count)),
            ppo_optimization_ms_mean=(
                self._opt_optimization_acc / max(1, self._opt_duration_count)
            ),
            ppo_aux_ms_mean=(self._opt_aux_acc / max(1, self._opt_duration_count)),
            wall_clock_seconds=wall_clock_seconds,
        )
