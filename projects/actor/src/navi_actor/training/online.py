"""Online spherical-view trainer for collision-aware navigation."""

from __future__ import annotations

import csv
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zmq

from navi_actor.policy import LearnedSphericalPolicy, PolicyCheckpoint
from navi_actor.spherical_features import extract_spherical_features
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

__all__: list[str] = [
    "EvaluationMetrics",
    "EvaluationPoint",
    "OnlineTrainingMetrics",
    "OnlineSphericalTrainer",
]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnlineTrainingMetrics:
    """Summary metrics for one training run."""

    steps: int
    reward_mean: float
    reward_ema: float
    collision_rate: float
    forward_mean: float
    yaw_abs_mean: float
    novelty_rate: float
    visited_cells: int
    eval_reward_mean: float
    eval_collision_rate: float
    eval_novelty_rate: float
    eval_coverage_mean: float
    eval_history: tuple[EvaluationPoint, ...]


@dataclass(frozen=True)
class EvaluationMetrics:
    """Evaluation metrics over deterministic policy episodes."""

    episodes: int
    horizon: int
    reward_mean: float
    collision_rate: float
    novelty_rate: float
    coverage_mean: float


@dataclass(frozen=True)
class EvaluationPoint:
    """One periodic evaluation record tied to a training step."""

    step: int
    reward_mean: float
    collision_rate: float
    novelty_rate: float
    coverage_mean: float


class OnlineSphericalTrainer:
    """Train a lightweight policy from full spherical DistanceMatrix input."""

    def __init__(
        self,
        sub_address: str,
        step_endpoint: str,
        pub_address: str = "",
        learning_rate: float = 3e-3,
        sigma_forward: float = 0.12,
        sigma_yaw: float = 0.16,
        max_forward: float = 1.2,
        max_yaw: float = 1.2,
    ) -> None:
        self._sub_address = sub_address
        self._step_endpoint = step_endpoint
        self._pub_address = pub_address
        self._learning_rate = float(max(1e-6, learning_rate))
        self._sigma_forward = float(max(1e-3, sigma_forward))
        self._sigma_yaw = float(max(1e-3, sigma_yaw))
        self._max_forward = float(max(0.1, max_forward))
        self._max_yaw = float(max(0.1, max_yaw))

        self._feature_dim = 13
        self._rng = np.random.default_rng(42)

        self._w_forward = self._rng.normal(0.0, 0.1, size=(self._feature_dim,)).astype(np.float32)
        self._b_forward = np.float32(0.0)
        self._w_yaw = self._rng.normal(0.0, 0.1, size=(self._feature_dim,)).astype(np.float32)
        self._b_yaw = np.float32(0.0)

        self._reward_baseline = 0.0
        self._visited_cells: set[tuple[int, int]] = set()
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sub_socket: zmq.Socket[bytes] | None = None
        self._step_socket: zmq.Socket[bytes] | None = None
        self._pub_socket: zmq.Socket[bytes] | None = None

    def start(self) -> None:
        """Open training sockets."""
        sub = self._ctx.socket(zmq.SUB)
        sub.connect(self._sub_address)
        sub.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
        self._sub_socket = sub

        req = self._ctx.socket(zmq.REQ)
        req.connect(self._step_endpoint)
        self._step_socket = req

        if self._pub_address:
            pub = self._ctx.socket(zmq.PUB)
            pub.bind(self._pub_address)
            self._pub_socket = pub

    def stop(self) -> None:
        """Close training sockets."""
        if self._sub_socket is not None:
            self._sub_socket.close()
            self._sub_socket = None
        if self._step_socket is not None:
            self._step_socket.close()
            self._step_socket = None
        if self._pub_socket is not None:
            self._pub_socket.close()
            self._pub_socket = None
        self._ctx.term()

    def train(
        self,
        steps: int,
        log_every: int = 100,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "",
        checkpoint_prefix: str = "policy_step",
        eval_every: int = 0,
        eval_episodes: int = 0,
        eval_horizon: int = 100,
    ) -> OnlineTrainingMetrics:
        """Run online REINFORCE-style updates against step-mode simulation."""
        if self._sub_socket is None or self._step_socket is None:
            msg = "Trainer sockets are not initialised. Call start() first."
            raise RuntimeError(msg)

        total_steps = max(1, int(steps))
        log_stride = max(1, int(log_every))

        reward_acc = 0.0
        collision_count = 0
        novelty_count = 0
        forward_acc = 0.0
        yaw_abs_acc = 0.0
        latest_eval = EvaluationMetrics(
            episodes=0,
            horizon=max(1, int(eval_horizon)),
            reward_mean=0.0,
            collision_rate=0.0,
            novelty_rate=0.0,
            coverage_mean=0.0,
        )
        eval_history: list[EvaluationPoint] = []

        latest = self._recv_latest_matrix(timeout_ms=3000)
        if latest is None:
            msg = "Did not receive initial distance_matrix_v2 message."
            raise RuntimeError(msg)
        self._visited_cells = {self._pose_cell(latest)}

        for step_id in range(total_steps):
            obs = latest
            features = self._build_features(obs)
            forward_cmd, yaw_cmd, forward_mean, yaw_mean = self._sample_action(features)

            action = Action(
                env_ids=obs.env_ids.astype(np.int32, copy=False),
                linear_velocity=np.array([[forward_cmd, 0.0, 0.0]], dtype=np.float32),
                angular_velocity=np.array([[0.0, 0.0, yaw_cmd]], dtype=np.float32),
                policy_id="brain-v2-train-spherical",
                step_id=step_id,
                timestamp=time.time(),
            )
            self._publish_action(action)

            result = self._request_step(action, step_id)
            next_obs = self._recv_latest_matrix(timeout_ms=1000)
            if next_obs is None:
                next_obs = obs

            reward, collided, is_novel = self._compute_training_reward(obs, next_obs, forward_cmd, yaw_cmd)
            reward += 0.2 * float(result.reward)

            advantage = float(reward - self._reward_baseline)
            self._update_policy(features, forward_cmd, yaw_cmd, forward_mean, yaw_mean, reward)

            # Publish per-step training telemetry
            self._publish_training_step(
                step_id=step_id,
                reward=reward,
                advantage=advantage,
                collided=collided,
                is_novel=is_novel,
                forward_cmd=forward_cmd,
                yaw_cmd=yaw_cmd,
            )

            reward_acc += reward
            collision_count += int(collided)
            novelty_count += int(is_novel)
            forward_acc += float(forward_cmd)
            yaw_abs_acc += abs(float(yaw_cmd))
            self._reward_baseline = 0.98 * self._reward_baseline + 0.02 * reward

            current_step = step_id + 1

            if checkpoint_every > 0 and checkpoint_dir and current_step % checkpoint_every == 0:
                checkpoint_path = self._periodic_checkpoint_path(
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_prefix=checkpoint_prefix,
                    step=current_step,
                )
                self.save_checkpoint(checkpoint_path)
                _LOGGER.info("[train] checkpoint_saved=%s", checkpoint_path)

            if eval_every > 0 and eval_episodes > 0 and current_step % eval_every == 0:
                latest_eval = self.evaluate(
                    episodes=eval_episodes,
                    horizon=eval_horizon,
                )
                eval_history.append(
                    EvaluationPoint(
                        step=current_step,
                        reward_mean=latest_eval.reward_mean,
                        collision_rate=latest_eval.collision_rate,
                        novelty_rate=latest_eval.novelty_rate,
                        coverage_mean=latest_eval.coverage_mean,
                    )
                )
                self._publish_eval_telemetry(
                    step_id=current_step,
                    metrics=latest_eval,
                )
                _LOGGER.info(
                    "[eval] step=%d reward_mean=%.4f collision_rate=%.3f novelty_rate=%.3f coverage_mean=%.3f",
                    current_step,
                    latest_eval.reward_mean,
                    latest_eval.collision_rate,
                    latest_eval.novelty_rate,
                    latest_eval.coverage_mean,
                )

            if current_step % log_stride == 0:
                mean_reward = reward_acc / float(current_step)
                collision_rate = collision_count / float(current_step)
                novelty_rate = novelty_count / float(current_step)
                _LOGGER.info(
                    "[train] "
                    f"step={current_step} "
                    f"reward_mean={mean_reward:.4f} "
                    f"reward_ema={self._reward_baseline:.4f} "
                    f"collision_rate={collision_rate:.3f} "
                    f"novelty_rate={novelty_rate:.3f}"
                )

            latest = next_obs

        return OnlineTrainingMetrics(
            steps=total_steps,
            reward_mean=reward_acc / float(total_steps),
            reward_ema=float(self._reward_baseline),
            collision_rate=collision_count / float(total_steps),
            forward_mean=forward_acc / float(total_steps),
            yaw_abs_mean=yaw_abs_acc / float(total_steps),
            novelty_rate=novelty_count / float(total_steps),
            visited_cells=len(self._visited_cells),
            eval_reward_mean=latest_eval.reward_mean,
            eval_collision_rate=latest_eval.collision_rate,
            eval_novelty_rate=latest_eval.novelty_rate,
            eval_coverage_mean=latest_eval.coverage_mean,
            eval_history=tuple(eval_history),
        )

    def evaluate(self, episodes: int, horizon: int = 100) -> EvaluationMetrics:
        """Run deterministic evaluation episodes and return exploration metrics."""
        if self._sub_socket is None or self._step_socket is None:
            msg = "Trainer sockets are not initialised. Call start() first."
            raise RuntimeError(msg)

        eval_episodes = max(1, int(episodes))
        eval_horizon = max(1, int(horizon))

        total_reward = 0.0
        total_collisions = 0
        total_novel_steps = 0
        total_coverage = 0.0
        eval_step_id = 10_000_000

        latest = self._recv_latest_matrix(timeout_ms=3000)
        if latest is None:
            msg = "Did not receive distance_matrix_v2 for evaluation."
            raise RuntimeError(msg)

        for _episode in range(eval_episodes):
            episode_obs = latest
            episode_cells: set[tuple[int, int]] = {self._pose_cell(episode_obs)}
            episode_reward = 0.0
            episode_collisions = 0
            episode_novel_steps = 0

            for _t in range(eval_horizon):
                features = self._build_features(episode_obs)
                forward_cmd, yaw_cmd = self._deterministic_action(features)

                action = Action(
                    env_ids=episode_obs.env_ids.astype(np.int32, copy=False),
                    linear_velocity=np.array([[forward_cmd, 0.0, 0.0]], dtype=np.float32),
                    angular_velocity=np.array([[0.0, 0.0, yaw_cmd]], dtype=np.float32),
                    policy_id="brain-v2-eval-spherical",
                    step_id=eval_step_id,
                    timestamp=time.time(),
                )
                eval_step_id += 1

                result = self._request_step(action, eval_step_id)
                next_obs = self._recv_latest_matrix(timeout_ms=1000)
                if next_obs is None:
                    next_obs = episode_obs

                reward, collided, is_novel = self._compute_eval_reward(
                    obs=episode_obs,
                    next_obs=next_obs,
                    forward_cmd=forward_cmd,
                    yaw_cmd=yaw_cmd,
                    episode_cells=episode_cells,
                )
                reward += 0.2 * float(result.reward)

                episode_reward += reward
                episode_collisions += int(collided)
                episode_novel_steps += int(is_novel)
                episode_cells.add(self._pose_cell(next_obs))
                episode_obs = next_obs

            total_reward += episode_reward / float(eval_horizon)
            total_collisions += episode_collisions
            total_novel_steps += episode_novel_steps
            total_coverage += len(episode_cells) / float(eval_horizon)
            latest = episode_obs

        denom_steps = float(eval_episodes * eval_horizon)
        return EvaluationMetrics(
            episodes=eval_episodes,
            horizon=eval_horizon,
            reward_mean=total_reward / float(eval_episodes),
            collision_rate=total_collisions / denom_steps,
            novelty_rate=total_novel_steps / denom_steps,
            coverage_mean=total_coverage / float(eval_episodes),
        )

    def _recv_latest_matrix(self, timeout_ms: int) -> DistanceMatrix | None:
        if self._sub_socket is None:
            return None

        poller = zmq.Poller()
        poller.register(self._sub_socket, zmq.POLLIN)
        events = dict(poller.poll(timeout=max(1, timeout_ms)))
        if self._sub_socket not in events:
            return None

        latest: DistanceMatrix | None = None
        while True:
            try:
                _topic, data = self._sub_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            msg = deserialize(data)
            if isinstance(msg, DistanceMatrix):
                latest = msg
        return latest

    def _request_step(self, action: Action, step_id: int) -> StepResult:
        if self._step_socket is None:
            msg = "Step socket is not initialised."
            raise RuntimeError(msg)

        req = StepRequest(action=action, step_id=step_id, timestamp=time.time())
        self._step_socket.send(serialize(req))
        reply = deserialize(self._step_socket.recv())
        if not isinstance(reply, StepResult):
            msg = f"Expected StepResult, got {type(reply).__name__}"
            raise TypeError(msg)
        return reply

    def _build_features(self, obs: DistanceMatrix) -> np.ndarray:
        return extract_spherical_features(obs)

    def _sample_action(self, features: np.ndarray) -> tuple[float, float, float, float]:
        f_pre = float(np.dot(self._w_forward, features) + self._b_forward)
        y_pre = float(np.dot(self._w_yaw, features) + self._b_yaw)

        f_sig = 1.0 / (1.0 + np.exp(-f_pre))
        y_tanh = np.tanh(y_pre)
        forward_mean = float(self._max_forward * f_sig)
        yaw_mean = float(self._max_yaw * y_tanh)

        forward = float(np.clip(forward_mean + self._sigma_forward * self._rng.standard_normal(), 0.0, self._max_forward))
        yaw = float(np.clip(yaw_mean + self._sigma_yaw * self._rng.standard_normal(), -self._max_yaw, self._max_yaw))
        return forward, yaw, forward_mean, yaw_mean

    def _deterministic_action(self, features: np.ndarray) -> tuple[float, float]:
        """Infer deterministic action means for evaluation episodes."""
        f_pre = float(np.dot(self._w_forward, features) + self._b_forward)
        y_pre = float(np.dot(self._w_yaw, features) + self._b_yaw)

        forward = float(self._max_forward / (1.0 + np.exp(-f_pre)))
        yaw = float(self._max_yaw * np.tanh(y_pre))
        return forward, yaw

    def _compute_training_reward(
        self,
        obs: DistanceMatrix,
        next_obs: DistanceMatrix,
        forward_cmd: float,
        yaw_cmd: float,
    ) -> tuple[float, bool, bool]:
        feats = self._build_features(next_obs)
        front_min = float(feats[0])
        front_mean = float(feats[1])

        dx = float(next_obs.robot_pose.x - obs.robot_pose.x)
        dz = float(next_obs.robot_pose.z - obs.robot_pose.z)
        progress = float(np.sqrt(dx * dx + dz * dz))

        expected = max(1e-3, float(max(forward_cmd, 0.0)))
        progress_ratio = progress / expected
        collided = forward_cmd > 0.2 and progress_ratio < 0.15

        new_cell = self._pose_cell(next_obs)
        is_novel = new_cell not in self._visited_cells
        if is_novel:
            self._visited_cells.add(new_cell)

        reward = 1.2 * progress
        reward += 0.35 * front_min
        reward += 0.15 * front_mean
        reward += 0.45 if is_novel else -0.05
        reward += 0.05 if abs(yaw_cmd) > 0.1 and front_mean > 0.12 else 0.0
        reward -= 0.03 * abs(yaw_cmd)
        if collided:
            reward -= 1.5
        if front_min < 0.06:
            reward -= 0.8
        return reward, collided, is_novel

    def _compute_eval_reward(
        self,
        obs: DistanceMatrix,
        next_obs: DistanceMatrix,
        forward_cmd: float,
        yaw_cmd: float,
        episode_cells: set[tuple[int, int]],
    ) -> tuple[float, bool, bool]:
        feats = self._build_features(next_obs)
        front_min = float(feats[0])
        front_mean = float(feats[1])

        dx = float(next_obs.robot_pose.x - obs.robot_pose.x)
        dz = float(next_obs.robot_pose.z - obs.robot_pose.z)
        progress = float(np.sqrt(dx * dx + dz * dz))

        expected = max(1e-3, float(max(forward_cmd, 0.0)))
        progress_ratio = progress / expected
        collided = forward_cmd > 0.2 and progress_ratio < 0.15

        next_cell = self._pose_cell(next_obs)
        is_novel = next_cell not in episode_cells

        reward = 1.2 * progress
        reward += 0.35 * front_min
        reward += 0.15 * front_mean
        reward += 0.45 if is_novel else -0.05
        reward += 0.05 if abs(yaw_cmd) > 0.1 and front_mean > 0.12 else 0.0
        reward -= 0.03 * abs(yaw_cmd)
        if collided:
            reward -= 1.5
        if front_min < 0.06:
            reward -= 0.8
        return reward, collided, is_novel

    @staticmethod
    def _pose_cell(obs: DistanceMatrix) -> tuple[int, int]:
        x_cell = int(np.floor(obs.robot_pose.x / 2.0))
        z_cell = int(np.floor(obs.robot_pose.z / 2.0))
        return x_cell, z_cell

    def to_checkpoint(self) -> PolicyCheckpoint:
        """Export current parameters as a runtime-loadable policy checkpoint."""
        return PolicyCheckpoint(
            w_forward=self._w_forward.copy(),
            b_forward=float(self._b_forward),
            w_yaw=self._w_yaw.copy(),
            b_yaw=float(self._b_yaw),
            max_forward=self._max_forward,
            max_yaw=self._max_yaw,
        )

    def save_checkpoint(self, path: str) -> None:
        """Persist trainer policy for use by runtime actor."""
        LearnedSphericalPolicy.save_checkpoint(path, self.to_checkpoint())

    @staticmethod
    def save_eval_csv(path: str, history: Sequence[EvaluationPoint]) -> None:
        """Write evaluation history rows to CSV for offline analysis."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["step", "reward_mean", "collision_rate", "novelty_rate", "coverage_mean"])
            for item in history:
                writer.writerow([
                    item.step,
                    f"{item.reward_mean:.6f}",
                    f"{item.collision_rate:.6f}",
                    f"{item.novelty_rate:.6f}",
                    f"{item.coverage_mean:.6f}",
                ])

    @staticmethod
    def plot_eval_progress(path: str, history: Sequence[EvaluationPoint]) -> None:
        """Render exploration progress plot from periodic evaluation history."""
        if len(history) == 0:
            msg = "No evaluation history available to plot. Enable eval windows first."
            raise RuntimeError(msg)

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            msg = "matplotlib is required for plotting. Install it in projects/actor environment."
            raise RuntimeError(msg) from exc

        steps = np.array([h.step for h in history], dtype=np.int32)
        reward = np.array([h.reward_mean for h in history], dtype=np.float32)
        collisions = np.array([h.collision_rate for h in history], dtype=np.float32)
        novelty = np.array([h.novelty_rate for h in history], dtype=np.float32)
        coverage = np.array([h.coverage_mean for h in history], dtype=np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        ax_reward, ax_collision = axes[0]
        ax_novelty, ax_coverage = axes[1]

        ax_reward.plot(steps, reward, color="#2e86de", linewidth=2.0)
        ax_reward.set_title("Eval Reward")
        ax_reward.grid(True, alpha=0.3)

        ax_collision.plot(steps, collisions, color="#e74c3c", linewidth=2.0)
        ax_collision.set_title("Collision Rate")
        ax_collision.grid(True, alpha=0.3)

        ax_novelty.plot(steps, novelty, color="#27ae60", linewidth=2.0)
        ax_novelty.set_title("Novelty Rate")
        ax_novelty.grid(True, alpha=0.3)
        ax_novelty.set_xlabel("Training Step")

        ax_coverage.plot(steps, coverage, color="#8e44ad", linewidth=2.0)
        ax_coverage.set_title("Coverage Mean")
        ax_coverage.grid(True, alpha=0.3)
        ax_coverage.set_xlabel("Training Step")

        fig.suptitle("Navi Exploration Progress", fontsize=13)
        fig.tight_layout()

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    # ── ZMQ telemetry helpers ──────────────────────────────────────────

    def _publish_action(self, action: Action) -> None:
        """Publish action_v2 on the PUB socket so the dashboard can see it."""
        if self._pub_socket is None:
            return
        self._pub_socket.send_multipart([
            TOPIC_ACTION.encode("utf-8"),
            serialize(action),
        ])

    def _publish_training_step(
        self,
        *,
        step_id: int,
        reward: float,
        advantage: float,
        collided: bool,
        is_novel: bool,
        forward_cmd: float,
        yaw_cmd: float,
    ) -> None:
        """Publish per-step training telemetry for dashboard curves."""
        if self._pub_socket is None:
            return
        grad_norm_fwd = float(np.linalg.norm(self._w_forward))
        grad_norm_yaw = float(np.linalg.norm(self._w_yaw))
        event = TelemetryEvent(
            event_type="actor.training.step",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=np.array([
                reward,
                advantage,
                float(collided),
                float(is_novel),
                forward_cmd,
                yaw_cmd,
                grad_norm_fwd,
                grad_norm_yaw,
            ], dtype=np.float32),
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart([
            TOPIC_TELEMETRY_EVENT.encode("utf-8"),
            serialize(event),
        ])

    def _publish_eval_telemetry(
        self,
        *,
        step_id: int,
        metrics: EvaluationMetrics,
    ) -> None:
        """Publish evaluation window summary for dashboard plots."""
        if self._pub_socket is None:
            return
        event = TelemetryEvent(
            event_type="actor.training.eval",
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=np.array([
                metrics.reward_mean,
                metrics.collision_rate,
                metrics.novelty_rate,
                metrics.coverage_mean,
            ], dtype=np.float32),
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart([
            TOPIC_TELEMETRY_EVENT.encode("utf-8"),
            serialize(event),
        ])

    @staticmethod
    def _periodic_checkpoint_path(checkpoint_dir: str, checkpoint_prefix: str, step: int) -> str:
        out_dir = Path(checkpoint_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{checkpoint_prefix}_{step:07d}.npz"
        return str(out_dir / filename)

    def _update_policy(
        self,
        features: np.ndarray,
        forward_cmd: float,
        yaw_cmd: float,
        forward_mean: float,
        yaw_mean: float,
        reward: float,
    ) -> None:
        advantage = float(reward - self._reward_baseline)

        f_sig = np.clip(forward_mean / self._max_forward, 1e-5, 1.0 - 1e-5)
        y_tanh = np.clip(yaw_mean / self._max_yaw, -0.999, 0.999)

        dmean_dfpre = self._max_forward * f_sig * (1.0 - f_sig)
        dmean_dypre = self._max_yaw * (1.0 - y_tanh * y_tanh)

        dlogp_dmean_f = (forward_cmd - forward_mean) / (self._sigma_forward * self._sigma_forward)
        dlogp_dmean_y = (yaw_cmd - yaw_mean) / (self._sigma_yaw * self._sigma_yaw)

        grad_f = advantage * dlogp_dmean_f * dmean_dfpre
        grad_y = advantage * dlogp_dmean_y * dmean_dypre

        lr = self._learning_rate
        self._w_forward = (self._w_forward + lr * grad_f * features).astype(np.float32)
        self._b_forward = np.float32(self._b_forward + lr * grad_f)
        self._w_yaw = (self._w_yaw + lr * grad_y * features).astype(np.float32)
        self._b_yaw = np.float32(self._b_yaw + lr * grad_y)
