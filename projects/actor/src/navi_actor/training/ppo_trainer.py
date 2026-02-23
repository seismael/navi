"""ZMQ-based PPO training loop for the Cognitive Mamba Policy."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import ActorConfig
from navi_actor.learner_ppo import PpoLearner, PpoMetrics
from navi_actor.memory.episodic import EpisodicMemory
from navi_actor.reward_shaping import RewardShaper
from navi_actor.rnd import RNDModule
from navi_actor.rollout_buffer import PPOTransition, TrajectoryBuffer
from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    BatchStepRequest,
    BatchStepResult,
    DistanceMatrix,
    StepRequest,
    StepResult,
    TelemetryEvent,
    deserialize,
    serialize,
)

__all__: list[str] = ["PpoTrainer", "PpoTrainingMetrics"]

_LOGGER = logging.getLogger(__name__)


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


class PpoTrainer:
    """Train a CognitiveMambaPolicy using PPO with ZMQ step-mode environment.

    Collects rollouts of configurable length, computes GAE advantages,
    and runs multi-epoch minibatch PPO updates.
    """

    def __init__(self, config: ActorConfig) -> None:
        self._config = config
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sub_socket: zmq.Socket[bytes] | None = None
        self._step_socket: zmq.Socket[bytes] | None = None
        self._pub_socket: zmq.Socket[bytes] | None = None

        self._n_actors = config.n_actors

        # Compute device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared policy
        self._policy = CognitiveMambaPolicy(
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

        # Per-actor episodic memories
        self._memories: dict[int, EpisodicMemory] = {
            i: EpisodicMemory(
                embedding_dim=config.embedding_dim,
                capacity=config.memory_capacity,
                exclusion_window=config.memory_exclusion_window,
                similarity_threshold=config.loop_threshold,
            )
            for i in range(self._n_actors)
        }

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

        # Per-actor trajectory buffers — dual sets for async double-buffering.
        # The simulation thread writes into ``_active_buffers`` while the
        # optimisation thread processes the filled set.  Pointer swap
        # happens at rollout boundaries.
        self._buffers_a: dict[int, TrajectoryBuffer] = {
            i: TrajectoryBuffer(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            for i in range(self._n_actors)
        }
        self._buffers_b: dict[int, TrajectoryBuffer] = {
            i: TrajectoryBuffer(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
            )
            for i in range(self._n_actors)
        }
        # Active buffer set pointer (sim thread writes here)
        self._buffers: dict[int, TrajectoryBuffer] = self._buffers_a

        # ── Async double-buffering state ──
        self._opt_event = threading.Event()     # signal opt thread to run
        self._opt_done = threading.Event()      # opt thread signals completion
        self._opt_done.set()                    # initially "done" (no pending work)
        self._opt_lock = threading.Lock()       # guards weight updates
        self._opt_thread: threading.Thread | None = None
        self._opt_stop = threading.Event()      # signal opt thread to exit
        # Filled buffer set waiting for optimisation
        self._opt_buffers: dict[int, TrajectoryBuffer] | None = None
        self._opt_obs: dict[int, DistanceMatrix] | None = None
        self._opt_hiddens: dict[int, torch.Tensor | None] | None = None
        # Zero-wait metric counters
        self._sim_steps_during_opt: int = 0
        self._total_sim_steps: int = 0

        # Shared reference for opt thread to write metrics back
        self._last_opt_metrics: PpoMetrics | None = None

        # Pending observations received for the wrong actor (Fix 1)
        self._pending_obs: dict[int, DistanceMatrix] = {}

        # Thread pool for parallel episodic memory queries (FAISS
        # releases the GIL so threads achieve true concurrency).
        self._mem_pool: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=self._n_actors)
            if self._n_actors > 1
            else None
        )

    def start(self) -> None:
        """Open ZMQ sockets for training."""
        sub = self._ctx.socket(zmq.SUB)
        sub.connect(self._config.sub_address)
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
        self._sub_socket = sub

        req = self._ctx.socket(zmq.REQ)
        req.connect(self._config.step_endpoint)
        self._step_socket = req

        pub = self._ctx.socket(zmq.PUB)
        pub.bind(self._config.pub_address)
        self._pub_socket = pub

        _LOGGER.info(
            "PPO Trainer started: SUB=%s, REQ=%s, PUB=%s, device=%s",
            self._config.sub_address,
            self._config.step_endpoint,
            self._config.pub_address,
            self._device,
        )

    def stop(self) -> None:
        """Close sockets and terminate ZMQ context.

        Idempotent — safe to call more than once.
        """
        import contextlib

        for sock in (self._sub_socket, self._step_socket, self._pub_socket):
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close()
        self._sub_socket = None
        self._step_socket = None
        self._pub_socket = None
        with contextlib.suppress(Exception):
            self._ctx.term()

    # ── Full training-state persistence (knowledge accumulation) ──

    def save_training_state(self, path: str | Path) -> None:
        """Save complete training state for knowledge accumulation.

        Persists **all** learned state so that training can resume across
        different scenes without losing accumulated knowledge:

        - Policy network weights (encoder, temporal_core, actor_critic heads)
        - RND target + predictor network weights and running statistics
        - Adam optimizer state dicts (policy + RND)
        - RewardShaper global step (beta annealing position)
        - Accumulated training step counter

        The checkpoint is a single ``.pt`` file containing a dict with
        versioned keys.  Old-format checkpoints (plain state_dict) are
        detected automatically by :meth:`load_training_state`.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = {
            "version": 2,
            "policy_state_dict": self._policy.state_dict(),
            "rnd_state_dict": self._rnd.state_dict(),
            "reward_shaper_step": self._reward_shaper._global_step,
        }

        # Optimizer states (may not exist if no update has run yet)
        if self._learner._optimizer is not None:
            state["optimizer_state_dict"] = self._learner._optimizer.state_dict()
        if self._learner._value_optimizer is not None:
            state["value_optimizer_state_dict"] = (
                self._learner._value_optimizer.state_dict()
            )
        if self._learner._rnd_optimizer is not None:
            state["rnd_optimizer_state_dict"] = (
                self._learner._rnd_optimizer.state_dict()
            )

        torch.save(state, save_path)
        _LOGGER.info(
            "Training state saved: %s (step=%d, beta=%.4f)",
            save_path,
            self._reward_shaper._global_step,
            self._reward_shaper.beta,
        )

    def load_training_state(self, path: str | Path) -> None:
        """Restore complete training state from a checkpoint.

        Handles two checkpoint formats:

        - **v2** (dict with ``"version": 2``): full training state including
          RND, optimizers, and reward-shaper position.
        - **Legacy** (plain ``state_dict``): only policy weights are restored;
          all other components start fresh (backward compatible).
        """
        load_path = Path(path)
        data: Any = torch.load(load_path, weights_only=False, map_location="cpu")

        if isinstance(data, dict) and data.get("version") == 2:
            # ── Full training state ──
            self._policy.load_state_dict(data["policy_state_dict"])
            self._policy = self._policy.to(self._device)

            self._rnd.load_state_dict(data["rnd_state_dict"])
            self._rnd = self._rnd.to(self._device)

            self._reward_shaper._global_step = int(
                data.get("reward_shaper_step", 0)
            )

            # Restore optimizer states (lazy — create optimizers first)
            if "optimizer_state_dict" in data:
                opt = self._learner._get_optimizer(self._policy)
                opt.load_state_dict(data["optimizer_state_dict"])
            if "value_optimizer_state_dict" in data:
                val_opt = self._learner._get_value_optimizer(self._policy)
                val_opt.load_state_dict(data["value_optimizer_state_dict"])
            if "rnd_optimizer_state_dict" in data:
                rnd_opt = self._learner._get_rnd_optimizer(self._rnd)
                rnd_opt.load_state_dict(data["rnd_optimizer_state_dict"])

            _LOGGER.info(
                "Full training state restored from %s "
                "(step=%d, beta=%.4f, optimizers=%s)",
                load_path,
                self._reward_shaper._global_step,
                self._reward_shaper.beta,
                "yes"
                if "optimizer_state_dict" in data
                else "no",
            )
        else:
            # ── Legacy checkpoint (plain state_dict) ──
            self._policy.load_state_dict(data)
            self._policy = self._policy.to(self._device)
            _LOGGER.info(
                "Legacy checkpoint loaded (policy weights only): %s",
                load_path,
            )

    def _recv_matrix(
        self,
        timeout_ms: int = 3000,
        *,
        expected_actor_id: int | None = None,
    ) -> DistanceMatrix | None:
        """Receive a DistanceMatrix, optionally filtered by actor_id.

        When *expected_actor_id* is set, observations for other actors
        are stashed in ``_pending_obs`` and the method keeps receiving
        until the correct actor's observation arrives (or timeout).
        """
        if self._sub_socket is None:
            return None

        # Check pending buffer first
        if expected_actor_id is not None and expected_actor_id in self._pending_obs:
            return self._pending_obs.pop(expected_actor_id)

        self._sub_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        deadline = time.monotonic() + timeout_ms / 1000.0

        while True:
            remaining_ms = int((deadline - time.monotonic()) * 1000)
            if remaining_ms <= 0:
                return None
            self._sub_socket.setsockopt(zmq.RCVTIMEO, remaining_ms)
            try:
                parts: list[bytes] = self._sub_socket.recv_multipart()
                if len(parts) < 2:
                    continue
                msg = deserialize(parts[1])
                assert isinstance(msg, DistanceMatrix)

                if expected_actor_id is None:
                    return msg

                obs_actor = int(msg.env_ids[0]) if len(msg.env_ids) > 0 else 0
                if obs_actor == expected_actor_id:
                    return msg

                # Stash for the correct actor to pick up later
                self._pending_obs[obs_actor] = msg
            except zmq.Again:
                return None

    def _request_step(self, action: Action, step_id: int) -> StepResult:
        """Send step request and receive result."""
        if self._step_socket is None:
            msg = "Step socket not initialised."
            raise RuntimeError(msg)
        req = StepRequest(action=action, step_id=step_id, timestamp=time.time())
        self._step_socket.send(serialize(req))
        reply = self._step_socket.recv()
        result = deserialize(reply)
        assert isinstance(result, StepResult)
        return result

    def _request_batch_step(
        self,
        actions: tuple[Action, ...],
        step_id: int,
    ) -> BatchStepResult:
        """Send batched step request for all actors and receive results.

        A single REQ/REP round-trip replaces N sequential ones, cutting
        ZMQ latency from O(N) to O(1) per rollout tick.
        """
        if self._step_socket is None:
            msg = "Step socket not initialised."
            raise RuntimeError(msg)
        req = BatchStepRequest(
            actions=actions,
            step_id=step_id,
            timestamp=time.time(),
        )
        self._step_socket.send(serialize(req))
        reply = self._step_socket.recv()
        result = deserialize(reply)
        assert isinstance(result, BatchStepResult)
        return result

    def _publish_action(self, action: Action) -> None:
        """Publish action to PUB socket."""
        if self._pub_socket is not None:
            self._pub_socket.send_multipart([
                TOPIC_ACTION.encode("utf-8"),
                serialize(action),
            ])

    def _publish_step_telemetry(
        self,
        *,
        step_id: int,
        raw_reward: float,
        shaped_reward: float,
        intrinsic_reward: float,
        loop_similarity: float,
        is_loop: bool,
        beta: float,
        done: bool,
        forward_vel: float,
        yaw_vel: float,
        actor_id: int = 0,
    ) -> None:
        """Publish per-step rollout telemetry via ZMQ."""
        if self._pub_socket is None:
            return

        payload = np.array([
            raw_reward,          # [0] extrinsic reward
            shaped_reward,       # [1] total shaped reward
            intrinsic_reward,    # [2] RND intrinsic reward
            loop_similarity,     # [3] episodic memory similarity
            float(is_loop),      # [4] loop detected flag
            beta,                # [5] intrinsic annealing coefficient
            float(done),         # [6] episode done flag
            forward_vel,         # [7] forward velocity command
            yaw_vel,             # [8] yaw velocity command
        ], dtype=np.float32)

        event = TelemetryEvent(
            event_type="actor.training.ppo.step",
            episode_id=0,
            env_id=actor_id,
            step_id=step_id,
            payload=payload,
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart([
            TOPIC_TELEMETRY_EVENT.encode("utf-8"),
            serialize(event),
        ])

    def _publish_update_telemetry(
        self,
        step_id: int,
        reward_ema: float,
        metrics: PpoMetrics,
    ) -> None:
        """Publish per-PPO-update telemetry via ZMQ."""
        if self._pub_socket is None:
            return

        payload = np.array([
            reward_ema,              # [0] reward EMA
            metrics.policy_loss,     # [1] policy loss
            metrics.value_loss,      # [2] value loss
            metrics.entropy,         # [3] entropy
            metrics.approx_kl,       # [4] approx KL
            metrics.clip_fraction,   # [5] clip fraction
            metrics.total_loss,      # [6] total loss
            metrics.rnd_loss,        # [7] RND distillation loss
            self._reward_shaper.beta,  # [8] current beta
        ], dtype=np.float32)

        event = TelemetryEvent(
            event_type="actor.training.ppo.update",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=payload,
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart([
            TOPIC_TELEMETRY_EVENT.encode("utf-8"),
            serialize(event),
        ])

    def _publish_episode_telemetry(
        self,
        *,
        step_id: int,
        episode_return: float,
        episode_length: int,
        actor_id: int = 0,
    ) -> None:
        """Publish end-of-episode summary telemetry via ZMQ."""
        if self._pub_socket is None:
            return

        payload = np.array([
            episode_return,         # [0] total shaped reward for episode
            float(episode_length),  # [1] number of steps in episode
        ], dtype=np.float32)

        event = TelemetryEvent(
            event_type="actor.training.ppo.episode",
            episode_id=0,
            env_id=actor_id,
            step_id=step_id,
            payload=payload,
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart([
            TOPIC_TELEMETRY_EVENT.encode("utf-8"),
            serialize(event),
        ])

    def _obs_to_tensor(self, obs: DistanceMatrix) -> torch.Tensor:
        """Convert DistanceMatrix to (2, Az, El) float32 tensor.

        Depth and semantic may have a leading env dimension (n_envs, Az, El).
        We select env index 0 to get (Az, El).
        """
        depth = np.asarray(obs.depth, dtype=np.float32)
        semantic = np.asarray(obs.semantic, dtype=np.float32)
        # Strip leading env dimension if present
        if depth.ndim == 3:
            depth = depth[0]
        if semantic.ndim == 3:
            semantic = semantic[0]
        stacked = np.stack([depth, semantic])
        return torch.from_numpy(stacked)

    def _stack_hiddens(
        self,
        hiddens: dict[int, torch.Tensor | None],
        n: int,
    ) -> torch.Tensor | None:
        """Stack per-actor hidden states into a batched tensor.

        For the GRU fallback the hidden shape is ``(1, B, D)``.
        We stack N per-actor hiddens along B to get ``(1, N, D)``.
        If **all** actors have ``None`` hidden (e.g. first step),
        we return ``None``.

        For the Mamba2 core (stateless training), hidden is always
        ``None``.
        """
        if all(h is None for h in hiddens.values()):
            return None
        # Determine D from the first non-None hidden
        d_model = self._policy.encoder.embedding_dim
        parts: list[torch.Tensor] = []
        for i in range(n):
            h = hiddens[i]
            if h is None:
                # Zero hidden for actors at episode start
                parts.append(torch.zeros(1, 1, d_model, device=self._device))
            else:
                # h is (1, 1, D) from when this actor was last stepped
                parts.append(h.to(self._device))
        return torch.cat(parts, dim=1)  # (1, N, D)

    @staticmethod
    def _extract_hidden(
        batched: torch.Tensor | None,
        actor_id: int,
    ) -> torch.Tensor | None:
        """Extract a single actor's hidden from a batched tensor.

        Returns a ``(1, 1, D)`` slice for the given actor index.
        """
        if batched is None:
            return None
        return batched[:, actor_id : actor_id + 1, :].clone()

    # ── Async optimisation thread ────────────────────────────────

    def _optimisation_worker(
        self,
        ppo_epochs: int,
        minibatch_size: int,
        seq_len: int,
        step_ref: list[int],
        reward_ema_ref: list[float],
    ) -> None:
        """Background thread: runs PPO updates on filled buffers.

        Waits for ``_opt_event``, processes the filled buffer set,
        then signals ``_opt_done``.  The simulation thread continues
        stepping into the alternate buffer set with zero latency.

        Weight updates are guarded by ``_opt_lock`` to prevent tearing
        during the brief ``optimizer.step()`` window.  PyTorch inference
        (``torch.no_grad()`` forward) is thread-safe and needs no lock.
        """
        while not self._opt_stop.is_set():
            # Wait for sim thread to signal a filled buffer
            self._opt_event.wait(timeout=0.5)
            if self._opt_stop.is_set():
                break
            if not self._opt_event.is_set():
                continue
            self._opt_event.clear()

            buffers = self._opt_buffers
            obs_per_actor = self._opt_obs
            hiddens = self._opt_hiddens
            if buffers is None or obs_per_actor is None or hiddens is None:
                self._opt_done.set()
                continue

            n = self._n_actors

            try:
                # ── Bootstrap values (batched) ──
                self._policy.eval()
                active_ids = [
                    aid for aid in range(n) if len(buffers[aid]) > 0
                ]
                if active_ids:
                    with torch.no_grad():
                        boot_obs = torch.stack([
                            self._obs_to_tensor(obs_per_actor[aid])
                            for aid in active_ids
                        ]).to(self._device)
                        boot_hiddens = self._stack_hiddens(
                            {i: hiddens[active_ids[i]]
                             for i in range(len(active_ids))},
                            len(active_ids),
                        )
                        _, _, boot_vals, _, _ = self._policy.forward(
                            boot_obs, boot_hiddens,
                        )
                    for k, aid in enumerate(active_ids):
                        buffers[aid].compute_returns_and_advantages(
                            last_value=boot_vals[k].item(),
                        )

                # ── Merge buffers ──
                merged: TrajectoryBuffer | None = None
                for actor_id in range(n):
                    buf = buffers[actor_id]
                    if len(buf) == 0:
                        continue
                    if merged is None:
                        merged = buf
                    else:
                        merged.extend_from(buf)

                # ── PPO update (under lock) ──
                if merged is not None and len(merged) > 0:
                    with self._opt_lock:
                        self._policy.train()
                        metrics = self._learner.train_ppo_epoch(
                            self._policy,
                            merged,
                            ppo_epochs=ppo_epochs,
                            minibatch_size=minibatch_size,
                            seq_len=seq_len,
                            rnd=self._rnd,
                        )
                    self._last_opt_metrics = metrics
                    _LOGGER.info(
                        "[PPO update @ step %d] policy_loss=%.4f "
                        "value_loss=%.4f entropy=%.4f kl=%.4f "
                        "clip=%.2f rnd=%.4f",
                        step_ref[0],
                        metrics.policy_loss,
                        metrics.value_loss,
                        metrics.entropy,
                        metrics.approx_kl,
                        metrics.clip_fraction,
                        metrics.rnd_loss,
                    )

                # Clear the processed buffers
                for buf in buffers.values():
                    buf.clear()

            except Exception:
                _LOGGER.exception("Optimisation thread error")
            finally:
                self._opt_done.set()

    def train(
        self,
        total_steps: int,
        *,
        log_every: int = 100,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "checkpoints",
    ) -> PpoTrainingMetrics:
        """Run PPO training loop with N actors in parallel.

        Each rollout tick performs **one** batched policy forward pass for
        all actors, sends **one** ``BatchStepRequest``, and receives
        **one** ``BatchStepResult``.  This replaces the old sequential
        round-robin loop and cuts ZMQ latency from O(N) to O(1) per tick.

        Per-actor buffers are concatenated for the PPO update so the
        shared policy learns from every actor's experience.

        Args:
            total_steps: total environment steps to collect (across all actors).
            log_every: steps between log messages.
            checkpoint_every: steps between checkpoints (0 = disabled).
            checkpoint_dir: directory for checkpoint files.

        Returns:
            Summary training metrics.

        """
        if self._sub_socket is None or self._step_socket is None:
            msg = "Sockets not initialised. Call start() first."
            raise RuntimeError(msg)

        n = self._n_actors
        rollout_length = self._config.rollout_length
        ppo_epochs = self._config.ppo_epochs
        minibatch_size = self._config.minibatch_size
        seq_len = self._config.bptt_len

        reward_sum = 0.0
        reward_ema_raw = 0.0
        ema_alpha = 0.01
        ema_step = 0
        episode_count = 0
        intrinsic_sum = 0.0
        loop_detections = 0
        last_metrics = PpoMetrics(
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clip_fraction=0.0,
            total_loss=0.0,
            rnd_loss=0.0,
        )

        # Per-actor state
        episode_reward_acc: dict[int, float] = {i: 0.0 for i in range(n)}
        episode_step_acc: dict[int, int] = {i: 0 for i in range(n)}
        hiddens: dict[int, torch.Tensor | None] = {i: None for i in range(n)}

        # Get initial observations via PUB/SUB (environment publishes on start)
        obs_per_actor: dict[int, DistanceMatrix] = {}
        for i in range(n):
            obs = self._recv_matrix(timeout_ms=5000, expected_actor_id=i)
            if obs is None:
                msg = f"No initial observation received for actor {i}."
                raise RuntimeError(msg)
            obs_per_actor[i] = obs

        reward_ema = 0.0
        step_id = 0

        # Mutable references for cross-thread communication
        step_ref: list[int] = [0]
        reward_ema_ref: list[float] = [0.0]

        # Start async optimisation thread (No-Stall protocol §6.2)
        self._opt_stop.clear()
        self._opt_done.set()
        self._sim_steps_during_opt = 0
        self._total_sim_steps = 0
        self._opt_thread = threading.Thread(
            target=self._optimisation_worker,
            args=(
                ppo_epochs, minibatch_size, seq_len,
                step_ref, reward_ema_ref,
            ),
            daemon=True,
        )
        self._opt_thread.start()

        while step_id < total_steps:
            # ── Collect rollout (batched across all actors) ──
            self._policy.eval()

            for _r in range(rollout_length):
                if step_id >= total_steps:
                    break

                # ── 1. Stack observations from all actors ──
                obs_tensors: list[torch.Tensor] = []
                for actor_id in range(n):
                    obs_tensors.append(self._obs_to_tensor(obs_per_actor[actor_id]))
                obs_batch = torch.stack(obs_tensors).to(self._device)  # (N, 2, Az, El)

                # ── 2. Stack hidden states for batched temporal core ──
                hidden_batch = self._stack_hiddens(hiddens, n)

                # ── 3. Single batched forward pass ──
                with torch.no_grad():
                    actions_t, log_probs_t, values_t, new_hidden_batch, z_t = (
                        self._policy.forward(obs_batch, hidden_batch)
                    )
                    # actions_t: (N, 4), log_probs_t: (N,), values_t: (N,)
                    # z_t: (N, D), new_hidden_batch: (1, N, D) or None

                    # RND intrinsic rewards for all actors at once
                    intrinsic_rewards = self._rnd.intrinsic_reward(z_t)  # (N,)

                # ── 4. Build per-actor Action objects ──
                actions_np = actions_t.cpu().numpy()  # (N, 4)
                z_np_all = z_t.cpu().numpy()  # (N, D)
                now = time.time()

                actor_actions: list[Action] = []
                for actor_id in range(n):
                    a = actions_np[actor_id]
                    actor_actions.append(Action(
                        env_ids=np.array([actor_id], dtype=np.int32),
                        linear_velocity=np.array(
                            [[a[0], a[1], a[2]]], dtype=np.float32,
                        ),
                        angular_velocity=np.array(
                            [[0.0, 0.0, a[3]]], dtype=np.float32,
                        ),
                        policy_id="cognitive-mamba-ppo",
                        step_id=step_id,
                        timestamp=now,
                    ))
                    # Publish action on PUB socket (for auditor)
                    self._publish_action(actor_actions[-1])

                # Episodic memory queries — threaded (FAISS releases GIL)
                loop_sims: list[float] = [0.0] * n
                if n > 1 and self._mem_pool is not None:
                    def _query_mem(
                        aid: int,
                        _z: np.ndarray = z_np_all,
                        _sims: list[float] = loop_sims,
                    ) -> None:
                        sim, _, _ = self._memories[aid].query(_z[aid])
                        _sims[aid] = sim
                    list(self._mem_pool.map(_query_mem, range(n)))
                else:
                    for actor_id in range(n):
                        sim, _, _ = self._memories[actor_id].query(
                            z_np_all[actor_id],
                        )
                        loop_sims[actor_id] = sim

                # ── 5. Single batched step request ──
                batch_result = self._request_batch_step(
                    tuple(actor_actions), step_id,
                )

                # ── 6. Per-actor transition processing ──
                for actor_id in range(n):
                    result = batch_result.results[actor_id]
                    next_obs = batch_result.observations[actor_id]
                    action_np = actions_np[actor_id]
                    intrinsic_reward = intrinsic_rewards[actor_id].item()
                    loop_sim = loop_sims[actor_id]

                    raw_reward = float(result.reward)
                    done = bool(result.done)
                    truncated = bool(result.truncated)

                    shaped = self._reward_shaper.shape(
                        raw_reward=raw_reward,
                        done=done,
                        forward_velocity=float(action_np[0]),
                        angular_velocity=float(action_np[3]),
                        intrinsic_reward=intrinsic_reward,
                        loop_similarity=loop_sim,
                    )
                    self._reward_shaper.step()

                    intrinsic_sum += intrinsic_reward
                    is_loop = loop_sim > self._config.loop_threshold
                    if is_loop:
                        loop_detections += 1

                    self._publish_step_telemetry(
                        step_id=step_id,
                        raw_reward=raw_reward,
                        shaped_reward=shaped.total,
                        intrinsic_reward=intrinsic_reward,
                        loop_similarity=loop_sim,
                        is_loop=is_loop,
                        beta=self._reward_shaper.beta,
                        done=done or truncated,
                        forward_vel=float(action_np[0]),
                        yaw_vel=float(action_np[3]),
                        actor_id=actor_id,
                    )

                    self._memories[actor_id].add(z_np_all[actor_id])

                    transition = PPOTransition(
                        observation=obs_tensors[actor_id],
                        action=actions_t[actor_id].cpu(),
                        log_prob=log_probs_t[actor_id].item(),
                        value=values_t[actor_id].item(),
                        reward=shaped.total,
                        done=done,
                        truncated=truncated,
                        hidden_state=self._extract_hidden(hidden_batch, actor_id),
                    )
                    self._buffers[actor_id].append(transition)

                    reward_sum += raw_reward
                    ema_step += 1
                    reward_ema_raw = (
                        ema_alpha * shaped.total
                        + (1.0 - ema_alpha) * reward_ema_raw
                    )
                    reward_ema = reward_ema_raw / (
                        1.0 - (1.0 - ema_alpha) ** ema_step
                    )
                    episode_reward_acc[actor_id] += shaped.total
                    episode_step_acc[actor_id] += 1

                    if done or truncated:
                        episode_count += 1
                        self._publish_episode_telemetry(
                            step_id=step_id,
                            episode_return=episode_reward_acc[actor_id],
                            episode_length=episode_step_acc[actor_id],
                            actor_id=actor_id,
                        )
                        episode_reward_acc[actor_id] = 0.0
                        episode_step_acc[actor_id] = 0
                        _LOGGER.info(
                            "[step %d] actor %d episode ended "
                            "(done=%s truncated=%s) "
                            "reward_ema=%.4f episodes=%d",
                            step_id, actor_id,
                            done, truncated,
                            reward_ema, episode_count,
                        )
                        hiddens[actor_id] = None
                        self._memories[actor_id].reset()
                    else:
                        hiddens[actor_id] = self._extract_hidden(
                            new_hidden_batch, actor_id,
                        )

                    obs_per_actor[actor_id] = next_obs

                # All N actors stepped in this tick
                step_id += n
                step_ref[0] = step_id
                reward_ema_ref[0] = reward_ema

                # Zero-wait tracking (§6.2)
                self._total_sim_steps += n
                if not self._opt_done.is_set():
                    self._sim_steps_during_opt += n

                if log_every > 0 and step_id % log_every < n:
                    _LOGGER.info(
                        "[step %d] reward_ema=%.4f episodes=%d beta=%.4f",
                        step_id, reward_ema, episode_count,
                        self._reward_shaper.beta,
                    )

            # ── Async buffer hand-off (No-Stall §6.2) ──
            # Wait for any previous optimisation to complete
            self._opt_done.wait()

            # Publish deferred telemetry from completed optimisation
            if self._last_opt_metrics is not None:
                last_metrics = self._last_opt_metrics
                self._publish_update_telemetry(
                    step_id, reward_ema, last_metrics,
                )

            # Snapshot observations & hidden states for bootstrap
            opt_obs = dict(obs_per_actor)
            opt_hiddens = dict(hiddens)

            # Swap active buffer pointer
            filled = self._buffers
            self._buffers = (
                self._buffers_b if filled is self._buffers_a
                else self._buffers_a
            )

            # Hand off filled buffers to optimisation thread
            self._opt_buffers = filled
            self._opt_obs = opt_obs
            self._opt_hiddens = opt_hiddens
            self._opt_done.clear()
            self._opt_event.set()

            # ── Checkpoint (requires opt to finish) ──
            if checkpoint_every > 0 and step_id % checkpoint_every < n:
                self._opt_done.wait()
                if self._last_opt_metrics is not None:
                    last_metrics = self._last_opt_metrics
                with self._opt_lock:
                    ckpt_path = (
                        Path(checkpoint_dir) / f"policy_step_{step_id}.pt"
                    )
                    self.save_training_state(ckpt_path)
                    _LOGGER.info("Checkpoint saved: %s", ckpt_path)

        # ── Shutdown optimisation thread ──
        self._opt_stop.set()
        self._opt_event.set()
        if self._opt_thread is not None:
            self._opt_thread.join(timeout=30.0)

        # Wait for final optimisation to complete
        self._opt_done.wait()
        if self._last_opt_metrics is not None:
            last_metrics = self._last_opt_metrics
            self._publish_update_telemetry(
                step_id, reward_ema, last_metrics,
            )

        # Compute zero-wait ratio
        zero_wait = (
            self._sim_steps_during_opt / max(1, self._total_sim_steps)
        )
        _LOGGER.info(
            "Zero-wait ratio: %.1f%% (%d/%d sim steps during opt)",
            zero_wait * 100,
            self._sim_steps_during_opt,
            self._total_sim_steps,
        )

        # ── Final checkpoint (always saved) ──
        final_ckpt = Path(checkpoint_dir) / f"policy_step_{step_id:07d}.pt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.save_training_state(final_ckpt)
        _LOGGER.info("Final checkpoint saved: %s", final_ckpt)

        return PpoTrainingMetrics(
            total_steps=step_id,
            episodes=episode_count,
            reward_mean=reward_sum / max(1, step_id),
            reward_ema=reward_ema,
            policy_loss=last_metrics.policy_loss,
            value_loss=last_metrics.value_loss,
            entropy=last_metrics.entropy,
            rnd_loss=last_metrics.rnd_loss,
            intrinsic_reward_mean=intrinsic_sum / max(1, step_id),
            loop_detections=loop_detections,
            beta_final=self._reward_shaper.beta,
            zero_wait_ratio=zero_wait,
        )
