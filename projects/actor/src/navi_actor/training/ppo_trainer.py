"""ZMQ-based PPO training loop for the Cognitive Mamba Policy."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

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

        # Compute device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy
        self._policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        ).to(self._device)

        # RND curiosity module
        self._rnd = RNDModule(
            input_dim=config.embedding_dim,
        ).to(self._device)

        # Episodic memory (non-parametric, CPU-based)
        self._memory = EpisodicMemory(
            embedding_dim=config.embedding_dim,
            capacity=config.memory_capacity,
            exclusion_window=config.memory_exclusion_window,
            similarity_threshold=config.loop_threshold,
        )

        # Reward shaper with annealing
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

        # Learner
        self._learner = PpoLearner(
            gamma=config.gamma,
            clip_ratio=config.clip_ratio,
            entropy_coeff=config.entropy_coeff,
            value_coeff=config.value_coeff,
            learning_rate=config.learning_rate,
        )

        # Trajectory buffer
        self._buffer = TrajectoryBuffer(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
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
        """Close sockets and terminate ZMQ context."""
        for sock in (self._sub_socket, self._step_socket, self._pub_socket):
            if sock is not None:
                sock.close()
        self._sub_socket = None
        self._step_socket = None
        self._pub_socket = None
        self._ctx.term()

    def _recv_matrix(self, timeout_ms: int = 3000) -> DistanceMatrix | None:
        """Receive the latest DistanceMatrix from SUB socket."""
        if self._sub_socket is None:
            return None
        self._sub_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            parts: list[bytes] = self._sub_socket.recv_multipart()
            if len(parts) >= 2:
                msg = deserialize(parts[1])
                assert isinstance(msg, DistanceMatrix)
                return msg
        except zmq.Again:
            pass
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
            env_id=0,
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

    def train(
        self,
        total_steps: int,
        *,
        log_every: int = 100,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "checkpoints",
    ) -> PpoTrainingMetrics:
        """Run PPO training loop.

        Collects rollout_length transitions, then runs PPO update.
        Repeats until total_steps transitions have been collected.

        Args:
            total_steps: total environment steps to collect.
            log_every: steps between log messages.
            checkpoint_every: steps between checkpoints (0 = disabled).
            checkpoint_dir: directory for checkpoint files.

        Returns:
            Summary training metrics.

        """
        if self._sub_socket is None or self._step_socket is None:
            msg = "Sockets not initialised. Call start() first."
            raise RuntimeError(msg)

        rollout_length = self._config.rollout_length
        ppo_epochs = self._config.ppo_epochs
        minibatch_size = self._config.minibatch_size
        seq_len = self._config.bptt_len

        reward_sum = 0.0
        reward_ema = 0.0
        ema_alpha = 0.01
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

        # Get initial observation
        obs = self._recv_matrix(timeout_ms=5000)
        if obs is None:
            msg = "No initial observation received."
            raise RuntimeError(msg)

        hidden: torch.Tensor | None = None
        step_id = 0

        while step_id < total_steps:
            # ── Collect rollout ──
            self._buffer.clear()
            self._policy.eval()

            for _r in range(rollout_length):
                if step_id >= total_steps:
                    break

                obs_tensor = self._obs_to_tensor(obs)

                with torch.no_grad():
                    obs_batch = obs_tensor.unsqueeze(0).to(self._device)
                    actions, log_probs, values, new_hidden = self._policy.forward(
                        obs_batch, hidden
                    )

                    # Extract spatial embedding for RND + episodic memory
                    z_t = self._policy.encode(obs_batch)
                    z_np = z_t.squeeze(0).cpu().numpy()

                    # RND intrinsic reward
                    intrinsic_reward = self._rnd.intrinsic_reward(z_t).item()

                    # Episodic memory: query for loop then store
                    loop_sim, _, _ = self._memory.query(z_np)

                action_np = actions.squeeze(0).cpu().numpy()
                action = Action(
                    env_ids=np.asarray(obs.env_ids, dtype=np.int32),
                    linear_velocity=np.array(
                        [[action_np[0], action_np[1], action_np[2]]],
                        dtype=np.float32,
                    ),
                    angular_velocity=np.array(
                        [[0.0, 0.0, action_np[3]]], dtype=np.float32
                    ),
                    policy_id="cognitive-mamba-ppo",
                    step_id=step_id,
                    timestamp=time.time(),
                )
                self._publish_action(action)

                result = self._request_step(action, step_id)
                next_obs = self._recv_matrix(timeout_ms=1000)
                if next_obs is None:
                    next_obs = obs

                raw_reward = float(result.reward)
                done = bool(getattr(result, "done", False))

                # Reward shaping: combine extrinsic + intrinsic + loop penalty
                shaped = self._reward_shaper.shape(
                    raw_reward=raw_reward,
                    done=done,
                    forward_velocity=float(action_np[0]),
                    angular_velocity=float(action_np[3]),
                    intrinsic_reward=intrinsic_reward,
                    loop_similarity=loop_sim,
                )
                self._reward_shaper.step()

                # Track intrinsic reward and loop detections
                intrinsic_sum += intrinsic_reward
                is_loop = loop_sim > self._config.loop_threshold
                if is_loop:
                    loop_detections += 1

                # Publish per-step telemetry
                self._publish_step_telemetry(
                    step_id=step_id,
                    raw_reward=raw_reward,
                    shaped_reward=shaped.total,
                    intrinsic_reward=intrinsic_reward,
                    loop_similarity=loop_sim,
                    is_loop=is_loop,
                    beta=self._reward_shaper.beta,
                    done=done,
                    forward_vel=float(action_np[0]),
                    yaw_vel=float(action_np[3]),
                )

                # Store embedding in episodic memory
                self._memory.add(z_np)

                transition = PPOTransition(
                    observation=obs_tensor,
                    action=actions.squeeze(0).cpu(),
                    log_prob=log_probs.item(),
                    value=values.item(),
                    reward=shaped.total,
                    done=done,
                )
                self._buffer.append(transition)

                reward_sum += raw_reward
                reward_ema = ema_alpha * raw_reward + (1.0 - ema_alpha) * reward_ema
                if done:
                    episode_count += 1
                    hidden = None
                    self._memory.reset()
                else:
                    hidden = new_hidden

                obs = next_obs
                step_id += 1

                if log_every > 0 and step_id % log_every == 0:
                    _LOGGER.info(
                        "[step %d] reward_ema=%.4f episodes=%d beta=%.4f",
                        step_id,
                        reward_ema,
                        episode_count,
                        self._reward_shaper.beta,
                    )

            # ── Bootstrap value for last state ──
            with torch.no_grad():
                last_obs = self._obs_to_tensor(obs).unsqueeze(0).to(self._device)
                _, _, last_val, _ = self._policy.forward(last_obs, hidden)
                last_value = last_val.item()

            self._buffer.compute_returns_and_advantages(last_value=last_value)

            # ── PPO update ──
            last_metrics = self._learner.train_ppo_epoch(
                self._policy,
                self._buffer,
                ppo_epochs=ppo_epochs,
                minibatch_size=minibatch_size,
                seq_len=seq_len,
                rnd=self._rnd,
            )

            _LOGGER.info(
                "[PPO update @ step %d] policy_loss=%.4f value_loss=%.4f "
                "entropy=%.4f kl=%.4f clip=%.2f rnd=%.4f",
                step_id,
                last_metrics.policy_loss,
                last_metrics.value_loss,
                last_metrics.entropy,
                last_metrics.approx_kl,
                last_metrics.clip_fraction,
                last_metrics.rnd_loss,
            )

            self._publish_update_telemetry(step_id, reward_ema, last_metrics)

            # ── Checkpoint ──
            if checkpoint_every > 0 and step_id % checkpoint_every == 0:
                ckpt_path = Path(checkpoint_dir) / f"policy_step_{step_id}.pt"
                self._policy.save_checkpoint(ckpt_path)
                _LOGGER.info("Checkpoint saved: %s", ckpt_path)

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
        )
