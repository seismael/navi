"""Single canonical PPO trainer for direct in-process sdfdag training."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
import logging
import time
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
    TelemetryEvent,
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
        publish_actor_ids: tuple[int, ...] = (),
    ) -> tuple[SdfDagTensorStepBatch, tuple[Any, ...]]: ...

    def close(self) -> None: ...


__all__: list[str] = ["PpoTrainer", "PpoTrainingMetrics"]

_LOGGER = logging.getLogger(__name__)

_SOFT_WARN_MIN_SPS: float = 15.0
_SOFT_WARN_MAX_ZERO_WAIT: float = 0.20
_SOFT_WARN_MAX_OPT_MS: float = 30_000.0
_DASHBOARD_HEARTBEAT_SECONDS: float = 0.5


def _safe_pub_send(
    pub_socket: zmq.Socket[bytes] | None,
    *,
    topic: str,
    payload: bytes,
    label: str,
) -> None:
    """Best-effort telemetry publish for attribution-only diagnostics."""
    if pub_socket is None:
        return
    try:
        pub_socket.send_multipart([topic.encode("utf-8"), payload])
    except Exception as exc:  # pragma: no cover - transport/runtime defensive path
        _LOGGER.warning("Skipping %s publish after send failure: %s", label, exc)


def _should_save_checkpoint(step_id: int, last_checkpoint_step: int, checkpoint_every: int) -> bool:
    """Return True when a full checkpoint interval has elapsed.

    Rollout updates advance in chunks, so modulo-based checks can miss saves when
    boundaries do not align exactly with ``checkpoint_every``.
    """
    return checkpoint_every > 0 and (step_id - last_checkpoint_step) >= checkpoint_every


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
    wall_clock_seconds: float = 0.0


@dataclass(frozen=True)
class _RolloutHostBatch:
    """CPU mirrors for one rollout tick after a single packed device transfer."""

    embedding_tensor: torch.Tensor
    action_tensor: torch.Tensor
    aux_tensor: torch.Tensor
    log_prob_tensor: torch.Tensor
    value_tensor: torch.Tensor
    intrinsic_reward_tensor: torch.Tensor
    action_numpy: np.ndarray | None


@dataclass(frozen=True)
class _RolloutHostScalarBatch:
    """CPU mirrors for the small scalar subset still needed by Python bookkeeping."""

    shaped_reward_tensor: torch.Tensor
    intrinsic_reward_tensor: torch.Tensor | None
    loop_similarity_tensor: torch.Tensor | None


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
        self._config = config
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._pub_socket: zmq.Socket[bytes] | None = None
        self._runtime = runtime
        self._gmdag_file = gmdag_file
        self._scene_pool = scene_pool
        self._last_dashboard_observation: DistanceMatrix | None = None
        self._last_dashboard_heartbeat_at: float = 0.0

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
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)

        self._rollout_policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
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
        )

        # Shared learner
        self._learner = PpoLearner(
            gamma=config.gamma,
            clip_ratio=config.clip_ratio,
            entropy_coeff=config.entropy_coeff,
            value_coeff=config.value_coeff,
            learning_rate=config.learning_rate,
        )

        # Multi-Trajectory buffers
        self._multi_buffer_a = MultiTrajectoryBuffer(
            n_actors=self._n_actors,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            capacity=config.rollout_length,
        )
        self._multi_buffers = self._multi_buffer_a

        self._sim_steps_during_opt: int = 0
        self._total_sim_steps: int = 0
        self._last_opt_duration_ms: float = 0.0
        self._opt_duration_acc: float = 0.0
        self._opt_duration_count: int = 0
        self._last_opt_metrics: PpoMetrics | None = None

    def _sync_rollout_policy(self) -> None:
        """Copy weights from learner_policy to rollout_policy with thread safety."""
        self._rollout_policy.load_state_dict(self._learner_policy.state_dict())
        self._rollout_policy.eval()

    def _run_ppo_update(
        self,
        *,
        multi_buffer: MultiTrajectoryBuffer,
        obs_batch: torch.Tensor,
        hiddens: dict[int, torch.Tensor | None],
        aux_dict: dict[int, torch.Tensor],
        ppo_epochs: int,
        minibatch_size: int,
        seq_len: int,
    ) -> None:
        t_opt = time.perf_counter()
        with torch.no_grad():
            b_obs = obs_batch.to(self._device)
            b_hid = self._stack_hiddens({i: hiddens[i] for i in range(self._n_actors)}, self._n_actors)
            b_aux = torch.stack([aux_dict[i] for i in range(self._n_actors)]).to(self._device)

            self._learner_policy.eval()
            _, _, b_val, _, _ = self._learner_policy.forward(b_obs, b_hid, aux_tensor=b_aux)
            multi_buffer.compute_returns_and_advantages(last_values=b_val)

        self._learner_policy.train()
        self._last_opt_metrics = self._learner.train_ppo_epoch(
            self._learner_policy,
            multi_buffer,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            seq_len=seq_len,
            rnd=self._rnd,
            progress_callback=lambda: self._maybe_publish_dashboard_heartbeat(step_id=self._total_sim_steps),
        )
        self._sync_rollout_policy()

        ms = (time.perf_counter() - t_opt) * 1000
        self._last_opt_duration_ms = ms
        self._opt_duration_acc += ms
        self._opt_duration_count += 1
        _LOGGER.info("Inline PPO optimization completed in %.2fms. Weights synced.", ms)
        if ms > _SOFT_WARN_MAX_OPT_MS:
            _LOGGER.warning(
                "Soft stall monitor: optimizer wall-time high (%.1fms > %.1fms)",
                ms,
                _SOFT_WARN_MAX_OPT_MS,
            )
        multi_buffer.clear()

    def start(self) -> None:
        """Initialize trainer resources for the canonical runtime."""
        pub = self._ctx.socket(zmq.PUB)
        pub.bind(self._config.pub_address)
        self._pub_socket = pub

        if self._runtime is None:
            raise RuntimeError("Canonical sdfdag runtime is not configured")

        _LOGGER.info(
            "Canonical PPO trainer started: actors=%d gmdag=%s pub=%s",
            self._n_actors,
            self._gmdag_file or "<state-only>",
            self._config.pub_address,
        )

    def stop(self) -> None:
        """Close trainer transport resources."""
        if self._runtime is not None:
            with contextlib.suppress(Exception):
                self._runtime.close()
        self._runtime = None
        for sock in (self._pub_socket,):
            if sock:
                with contextlib.suppress(Exception):
                    sock.close()
        self._pub_socket = None
        with contextlib.suppress(Exception):
            self._ctx.term()

    def save_training_state(self, path: str | Path) -> None:
        """Save complete training state."""
        save_path = Path(path)
        _LOGGER.info("Saving training checkpoint to %s", save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 2,
            "policy_state_dict": self._learner_policy.state_dict(),
            "rnd_state_dict": self._rnd.state_dict(),
            "reward_shaper_step": self._reward_shaper._global_step,
        }
        if self._learner._optimizer:
            state["optimizer_state_dict"] = self._learner._optimizer.state_dict()
        if self._learner._rnd_optimizer:
            state["rnd_optimizer_state_dict"] = self._learner._rnd_optimizer.state_dict()
        torch.save(state, save_path)

    def load_training_state(self, path: str | Path) -> None:
        """Restore training state."""
        _LOGGER.info("Loading training checkpoint from %s", path)
        data = torch.load(path, weights_only=False, map_location="cpu")
        if not isinstance(data, dict) or data.get("version") != 2:
            raise RuntimeError(
                "Unsupported training checkpoint format: expected version=2 canonical state",
            )

        self._learner_policy.load_state_dict(data["policy_state_dict"])
        self._rnd.load_state_dict(data["rnd_state_dict"])
        self._reward_shaper._global_step = int(data.get("reward_shaper_step", 0))
        if "optimizer_state_dict" in data:
            self._learner._get_optimizer(self._learner_policy).load_state_dict(data["optimizer_state_dict"])
        if "rnd_optimizer_state_dict" in data:
            self._learner._get_rnd_optimizer(self._rnd).load_state_dict(data["rnd_optimizer_state_dict"])

        # Sync weights to rollout policy after loading
        self._sync_rollout_policy()

    def _request_batch_step_tensor_actions(
        self,
        action_tensor: torch.Tensor,
        step_id: int,
        *,
        publish_actor_ids: tuple[int, ...] = (),
    ) -> tuple[SdfDagTensorStepBatch, tuple[Any, ...]]:
        runtime = self._require_tensor_runtime()
        step_batch, results = runtime.batch_step_tensor_actions(
            action_tensor,
            step_id,
            publish_actor_ids=publish_actor_ids,
        )
        return step_batch, results

    def _load_initial_observation_batch(
        self,
    ) -> tuple[torch.Tensor, dict[int, DistanceMatrix]]:
        runtime = self._require_tensor_runtime()
        observation_tensors: list[torch.Tensor] = []
        obs_dict: dict[int, DistanceMatrix] = {}
        for actor_id in range(self._n_actors):
            materialize = self._should_publish_actor_telemetry(actor_id)
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
        if self._pub_socket:
            _safe_pub_send(
                self._pub_socket,
                topic=TOPIC_DISTANCE_MATRIX,
                payload=serialize(observation),
                label="observation",
            )

    def _should_publish_dashboard_observation(self) -> bool:
        """Return True when a fresh dashboard frame should be materialized."""
        if not self._config.emit_observation_stream:
            return False
        return (time.perf_counter() - self._last_dashboard_heartbeat_at) >= _DASHBOARD_HEARTBEAT_SECONDS

    def _maybe_publish_dashboard_heartbeat(self, *, step_id: int) -> None:
        """Re-publish the last observation while PPO optimization is running."""
        if not self._config.emit_observation_stream:
            return
        if not self._pub_socket:
            return
        observation = self._last_dashboard_observation
        if observation is None:
            return
        now_perf = time.perf_counter()
        if (now_perf - self._last_dashboard_heartbeat_at) < _DASHBOARD_HEARTBEAT_SECONDS:
            return

        heartbeat = replace(
            observation,
            step_id=step_id,
            timestamp=time.time(),
        )
        self._last_dashboard_observation = heartbeat
        self._last_dashboard_heartbeat_at = now_perf
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_DISTANCE_MATRIX,
            payload=serialize(heartbeat),
            label="dashboard heartbeat",
        )

    def _should_publish_actor_telemetry(self, actor_id: int) -> bool:
        if self._config.telemetry_all_actors:
            return True
        return actor_id == int(self._config.telemetry_actor_id)

    def _publish_step_telemetry(self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any) -> None:
        if not self._config.emit_training_telemetry:
            return
        if not self._pub_socket:
            return
        if not self._should_publish_actor_telemetry(actor_id):
            return
        p = np.array([kwargs.get(k, 0.0) for k in ["raw_reward", "shaped_reward", "intrinsic_reward",
                     "loop_similarity", "is_loop", "beta", "done", "forward_vel", "yaw_vel"]], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.step", episode_id=episode_id, env_id=actor_id,
                               step_id=step_id, payload=p, timestamp=time.time())
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="step telemetry",
        )

    def _publish_update_telemetry(self, step_id: int, reward_ema: float, metrics: PpoMetrics) -> None:
        if not self._config.emit_training_telemetry:
            return
        if not self._pub_socket:
            return
        p = np.array([reward_ema, metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.approx_kl,
                     metrics.clip_fraction, metrics.total_loss, metrics.rnd_loss, self._reward_shaper.beta], dtype=np.float32)
        event = TelemetryEvent(
            event_type="actor.training.ppo.update",
            episode_id=0,
            env_id=int(self._config.telemetry_actor_id),
            step_id=step_id,
            payload=p,
            timestamp=time.time(),
        )
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="update telemetry",
        )

    def _publish_episode_telemetry(self, *, step_id: int, episode_id: int, actor_id: int, **kwargs: Any) -> None:
        if not self._config.emit_training_telemetry:
            return
        if not self._pub_socket:
            return
        if not self._should_publish_actor_telemetry(actor_id):
            return
        p = np.array([kwargs["episode_return"], float(kwargs["episode_length"])], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.episode", episode_id=episode_id, env_id=actor_id,
                               step_id=step_id, payload=p, timestamp=time.time())
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="episode telemetry",
        )

    def _publish_perf_telemetry(self, *, step_id: int, **kwargs: Any) -> None:
        if not self._config.emit_perf_telemetry:
            return
        if not self._pub_socket:
            return
        p = np.array([kwargs[k] for k in ["sps", "forward_pass_ms", "batch_step_ms", "memory_query_ms",
                     "transition_ms", "tick_total_ms", "zero_wait_ratio", "ppo_update_ms",
                     "host_extract_ms", "telemetry_publish_ms"]], dtype=np.float32)
        event = TelemetryEvent(event_type="actor.training.ppo.perf", episode_id=0, env_id=int(self._config.telemetry_actor_id), step_id=step_id,
                               payload=p, timestamp=time.time())
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="perf telemetry",
        )

    def _publish_runtime_perf(self, *, step_id: int) -> None:
        """Publish coarse runtime perf from the canonical sdfdag backend."""
        if not self._config.emit_perf_telemetry:
            return
        if not self._pub_socket:
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
        _safe_pub_send(
            self._pub_socket,
            topic=TOPIC_TELEMETRY_EVENT,
            payload=serialize(event),
            label="runtime perf telemetry",
        )

    def _require_runtime(self) -> CanonicalRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Canonical sdfdag runtime is not configured")
        return runtime

    def _require_tensor_runtime(self) -> CanonicalRuntime:
        runtime = self._require_runtime()
        if not hasattr(runtime, "reset_tensor") or not hasattr(runtime, "batch_step_tensor_actions"):
            raise RuntimeError(
                "Canonical PPO training requires tensor-native runtime seams: reset_tensor() and batch_step_tensor_actions().",
            )
        return runtime

    def _stack_hiddens(self, hiddens: dict[int, torch.Tensor | None], n: int) -> torch.Tensor | None:
        if all(h is None for h in hiddens.values()):
            return None
        dim = self._rollout_policy.temporal_core.d_model
        parts = []
        for i in range(n):
            h_state = hiddens[i]
            if h_state is not None:
                parts.append(h_state.to(self._device))
            else:
                parts.append(torch.zeros(1, 1, dim, device=self._device))
        return torch.cat(parts, dim=1)

    def _extract_hidden(self, batched: torch.Tensor | None, actor_id: int) -> torch.Tensor | None:
        return batched[:, actor_id : actor_id + 1, :].clone() if batched is not None else None

    def _pack_rollout_host_batch(
        self,
        *,
        actions: torch.Tensor,
        embeddings: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        aux_batch: torch.Tensor,
        materialize_action_numpy: bool,
    ) -> _RolloutHostBatch:
        """Snapshot rollout tensors while keeping canonical training data on-device."""
        action_tensor = actions.detach().clone()
        embedding_tensor = embeddings.detach().clone()
        aux_tensor = aux_batch.detach().clone()
        action_numpy = None
        if materialize_action_numpy:
            action_numpy = action_tensor.to(device="cpu").numpy()

        return _RolloutHostBatch(
            embedding_tensor=embedding_tensor,
            action_tensor=action_tensor,
            aux_tensor=aux_tensor,
            log_prob_tensor=log_probs.detach().clone(),
            value_tensor=values.detach().clone(),
            intrinsic_reward_tensor=intrinsic_rewards.detach().clone(),
            action_numpy=action_numpy,
        )

    def _extract_host_rollout_scalars(
        self,
        *,
        shaped_rewards: torch.Tensor,
        intrinsic_rewards: torch.Tensor,
        loop_similarities: torch.Tensor,
        include_telemetry_scalars: bool,
    ) -> _RolloutHostScalarBatch:
        """Batch the unavoidable device-to-host scalar transfer for one rollout tick."""
        if include_telemetry_scalars:
            packed = torch.stack(
                (shaped_rewards, intrinsic_rewards, loop_similarities),
                dim=1,
            ).detach().to(device="cpu", dtype=torch.float32)
            return _RolloutHostScalarBatch(
                shaped_reward_tensor=packed[:, 0],
                intrinsic_reward_tensor=packed[:, 1],
                loop_similarity_tensor=packed[:, 2],
            )

        return _RolloutHostScalarBatch(
            shaped_reward_tensor=shaped_rewards.detach().to(device="cpu", dtype=torch.float32),
            intrinsic_reward_tensor=None,
            loop_similarity_tensor=None,
        )

    def train(self, total_steps: int, *, log_every: int = 100, checkpoint_every: int = 0, checkpoint_dir: str = "checkpoints") -> PpoTrainingMetrics:
        n = self._n_actors
        rl, ep, mb, sl = self._config.rollout_length, self._config.ppo_epochs, self._config.minibatch_size, self._config.bptt_len
        r_sum, r_ema, ep_cnt, step_id = 0.0, 0.0, 0, 0
        unbounded = total_steps <= 0
        r_acc = dict.fromkeys(range(n), 0.0)
        s_acc = dict.fromkeys(range(n), 0)
        hids = dict.fromkeys(range(n), None)
        aux_states = {i: torch.zeros(3, dtype=torch.float32, device=self._device) for i in range(n)}

        current_obs_batch, obs_dict = self._load_initial_observation_batch()
        telemetry_actor_id = int(self._config.telemetry_actor_id)
        initial_view = obs_dict.get(telemetry_actor_id)
        if initial_view is not None and self._should_publish_actor_telemetry(telemetry_actor_id):
            self._publish_observation(initial_view)

        t_start = time.perf_counter()
        acc_fwd, acc_pack, acc_step, acc_mem, acc_trans, acc_shape, acc_madd, acc_buf, acc_host, acc_tel, acc_tick, t_cnt = (
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

        while unbounded or step_id < total_steps:
            # Rollout Phase
            for _ in range(rl):
                if (not unbounded) and step_id >= total_steps:
                    break
                t_tick = time.perf_counter()
                o_batch = current_obs_batch
                h_batch = self._stack_hiddens(hids, n)
                aux_batch = torch.stack([aux_states[i] for i in range(n)])

                t_fwd_start = time.perf_counter()
                with torch.no_grad():
                    self._rollout_policy.eval()
                    a_t, lp_t, v_t, n_h_batch, z_t = self._rollout_policy.forward(o_batch, h_batch, aux_tensor=aux_batch)
                    if self._config.enable_reward_shaping:
                        intr_r = self._rnd.intrinsic_reward(z_t)
                    else:
                        intr_r = torch.zeros((z_t.shape[0],), dtype=torch.float32, device=z_t.device)
                fwd_ms = (time.perf_counter() - t_fwd_start) * 1000

                publish_actor_ids: tuple[int, ...] = ()
                publish_step_telemetry = bool(
                    self._config.emit_training_telemetry and log_every > 0 and step_id % log_every == 0
                )
                publish_observations = self._should_publish_dashboard_observation()
                if publish_observations:
                    publish_actor_ids = tuple(
                        actor_id
                        for actor_id in range(n)
                        if self._should_publish_actor_telemetry(actor_id)
                    )

                t_step_start = time.perf_counter()
                step_batch, res_results = self._request_batch_step_tensor_actions(
                    a_t.detach(),
                    step_id,
                    publish_actor_ids=publish_actor_ids,
                )
                next_obs_batch = step_batch.observation_tensor
                published_observations = step_batch.published_observations
                step_ms = (time.perf_counter() - t_step_start) * 1000

                t_pack_start = time.perf_counter()
                host_batch = self._pack_rollout_host_batch(
                    actions=a_t,
                    embeddings=z_t,
                    log_probs=lp_t,
                    values=v_t,
                    intrinsic_rewards=intr_r,
                    aux_batch=aux_batch,
                    materialize_action_numpy=publish_step_telemetry,
                )
                pack_ms = (time.perf_counter() - t_pack_start) * 1000

                t_mem_start = time.perf_counter()
                if self._config.enable_episodic_memory:
                    # Keep similarities on-device for shaping and batch-copy once for logging/accounting.
                    loop_similarities, _ = self._memory.query_batch_tensor(host_batch.embedding_tensor)
                else:
                    loop_similarities = torch.zeros(
                        (n,),
                        dtype=torch.float32,
                        device=host_batch.embedding_tensor.device,
                    )
                mem_ms = (time.perf_counter() - t_mem_start) * 1000

                t_shape_start = time.perf_counter()
                reward_device = host_batch.action_tensor.device
                raw_rewards = step_batch.reward_tensor.to(device=reward_device, dtype=torch.float32)
                dones = step_batch.done_tensor.to(device=reward_device, dtype=torch.bool)
                intrinsic_rewards = host_batch.intrinsic_reward_tensor.to(device=reward_device, dtype=torch.float32)
                if self._config.enable_reward_shaping:
                    shaped_rewards = self._reward_shaper.shape_batch(
                        raw_rewards=raw_rewards,
                        dones=dones,
                        forward_velocities=host_batch.action_tensor[:, 0],
                        angular_velocities=host_batch.action_tensor[:, 3],
                        intrinsic_rewards=intrinsic_rewards,
                        loop_similarities=loop_similarities,
                    )
                    self._reward_shaper.step_batch(n)
                else:
                    shaped_rewards = raw_rewards
                shape_ms = (time.perf_counter() - t_shape_start) * 1000

                t_host_start = time.perf_counter()
                host_scalar_batch = self._extract_host_rollout_scalars(
                    shaped_rewards=shaped_rewards,
                    intrinsic_rewards=host_batch.intrinsic_reward_tensor,
                    loop_similarities=loop_similarities,
                    include_telemetry_scalars=publish_step_telemetry,
                )
                host_extract_ms = (time.perf_counter() - t_host_start) * 1000

                t_trans_start = time.perf_counter()
                buffer_ms = 0.0
                telemetry_ms = 0.0
                truncateds = step_batch.truncated_tensor.to(device=reward_device, dtype=torch.bool)
                t_buffer_start = time.perf_counter()
                self._multi_buffers.append_batch(
                    observations=o_batch,
                    actions=host_batch.action_tensor,
                    log_probs=host_batch.log_prob_tensor,
                    values=host_batch.value_tensor,
                    rewards=shaped_rewards,
                    dones=dones,
                    truncateds=truncateds,
                    hidden_batch=h_batch,
                    aux_tensors=host_batch.aux_tensor,
                )
                buffer_ms += (time.perf_counter() - t_buffer_start) * 1000
                r_sum += float(raw_rewards.sum().item())
                for i in range(n):
                    d = bool(res_results[i].done)
                    tr = bool(res_results[i].truncated)
                    shaped_total = float(host_scalar_batch.shaped_reward_tensor[i])

                    # Periodic step telemetry keeps dashboard mode detection alive.
                    if publish_step_telemetry:
                        if host_batch.action_numpy is None:
                            raise RuntimeError("action_numpy is required for telemetry emission")
                        if host_scalar_batch.intrinsic_reward_tensor is None or host_scalar_batch.loop_similarity_tensor is None:
                            raise RuntimeError("telemetry scalar extraction is required for step telemetry emission")
                        r = float(raw_rewards[i].item())
                        intrinsic_reward = float(host_scalar_batch.intrinsic_reward_tensor[i])
                        loop_similarity = float(host_scalar_batch.loop_similarity_tensor[i])
                        t_tel_start = time.perf_counter()
                        self._publish_step_telemetry(step_id=step_id, episode_id=int(res_results[i].episode_id), actor_id=i, raw_reward=r, shaped_reward=shaped_total,
                                                     intrinsic_reward=intrinsic_reward, loop_similarity=loop_similarity, is_loop=(loop_similarity > self._config.loop_threshold),
                                                     beta=self._reward_shaper.beta, done=d or tr, forward_vel=float(host_batch.action_numpy[i, 0]), yaw_vel=float(host_batch.action_numpy[i, 3]))
                        telemetry_ms += (time.perf_counter() - t_tel_start) * 1000

                    r_ema = 0.01 * shaped_total + 0.99 * r_ema
                    r_acc[i] += shaped_total
                    s_acc[i] += 1
                    aux_state = aux_states[i]
                    aux_state[0] = shaped_rewards[i]
                    aux_state[1] = loop_similarities[i]
                    aux_state[2] = host_batch.intrinsic_reward_tensor[i]
                    if d or tr:
                        ep_cnt += 1
                        t_tel_start = time.perf_counter()
                        self._publish_episode_telemetry(step_id=step_id, episode_id=int(res_results[i].episode_id), actor_id=i, episode_return=r_acc[i], episode_length=s_acc[i])
                        telemetry_ms += (time.perf_counter() - t_tel_start) * 1000
                        r_acc[i] = 0.0
                        s_acc[i] = 0
                        hids[i] = None
                        aux_state.zero_()
                    else:
                        hids[i] = self._extract_hidden(n_h_batch, i)
                    if publish_observations and self._should_publish_actor_telemetry(i) and i in published_observations:
                        t_tel_start = time.perf_counter()
                        self._publish_observation(published_observations[i])
                        telemetry_ms += (time.perf_counter() - t_tel_start) * 1000
                if self._config.enable_episodic_memory:
                    t_memory_add_start = time.perf_counter()
                    self._memory.add_batch_tensor(host_batch.embedding_tensor)
                    memory_add_ms = (time.perf_counter() - t_memory_add_start) * 1000
                else:
                    memory_add_ms = 0.0
                trans_ms = (time.perf_counter() - t_trans_start) * 1000
                current_obs_batch = next_obs_batch

                step_id += n
                self._total_sim_steps += n

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

                if log_every > 0 and step_id % log_every < n:
                    now_perf = time.perf_counter()
                    sps = log_every / max(0.001, now_perf - getattr(self, '_last_log_time', t_start))
                    self._last_log_time = now_perf
                    zw = self._sim_steps_during_opt / max(1, self._total_sim_steps)
                    _LOGGER.info(
                        "[step %d] reward_ema=%.4f episodes=%d | sps=%.1f zw=%.1f%% | "
                        "fwd=%.1fms pack=%.1fms env=%.1fms mem=%.1fms trans=%.1fms "
                        "(shape=%.1fms madd=%.1fms buf=%.1fms host=%.1fms tele=%.1fms)",
                        step_id, r_ema, ep_cnt, sps, zw * 100,
                        fwd_ms, pack_ms, step_ms, mem_ms, trans_ms, shape_ms, memory_add_ms, buffer_ms, host_extract_ms, telemetry_ms,
                    )
                    self._publish_perf_telemetry(step_id=step_id, sps=sps, forward_pass_ms=fwd_ms, batch_step_ms=step_ms, memory_query_ms=mem_ms,
                                                 transition_ms=trans_ms, tick_total_ms=tick_ms, zero_wait_ratio=zw, ppo_update_ms=self._last_opt_duration_ms,
                                                 host_extract_ms=host_extract_ms, telemetry_publish_ms=telemetry_ms)
                    self._publish_runtime_perf(step_id=step_id)
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

            # --- Rollout Loop Finished ---
            self._run_ppo_update(
                multi_buffer=self._multi_buffers,
                obs_batch=current_obs_batch.detach().clone(),
                hiddens=dict(hids),
                aux_dict={k: v.clone() for k, v in aux_states.items()},
                ppo_epochs=ep,
                minibatch_size=mb,
                seq_len=sl,
            )
            if self._last_opt_metrics:
                self._publish_update_telemetry(step_id, r_ema, self._last_opt_metrics)

            # Save when we've advanced at least one checkpoint interval since the last save.
            # This remains correct even when rollout boundaries do not land on exact modulo values.
            if _should_save_checkpoint(step_id, last_checkpoint_step, checkpoint_every):
                self.save_training_state(Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt")
                last_checkpoint_step = step_id

        return PpoTrainingMetrics(total_steps=step_id, episodes=ep_cnt, reward_mean=r_sum/max(1, step_id), reward_ema=r_ema,
                                  policy_loss=0.0, value_loss=0.0, entropy=0.0, rnd_loss=0.0, intrinsic_reward_mean=0.0,
                                  loop_detections=0, beta_final=self._reward_shaper.beta,
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
                                  wall_clock_seconds=(time.perf_counter() - t_start))
