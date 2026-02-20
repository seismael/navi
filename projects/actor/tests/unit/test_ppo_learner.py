"""Tests for PpoLearner (full PPO training)."""

from __future__ import annotations

import torch

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.learner_ppo import PpoLearner, PpoMetrics
from navi_actor.rollout_buffer import PPOTransition, TrajectoryBuffer


def _fill_buffer(n: int = 64) -> TrajectoryBuffer:
    """Create a trajectory buffer with n dummy transitions."""
    buf = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    for _ in range(n):
        buf.append(PPOTransition(
            observation=torch.randn(2, 64, 32),
            action=torch.randn(4),
            log_prob=-0.5,
            value=0.5,
            reward=1.0,
            done=False,
        ))
    buf.compute_returns_and_advantages(last_value=0.0)
    return buf


def test_legacy_train_epoch() -> None:
    """Legacy train_epoch should still work with RolloutBuffer."""
    import numpy as np

    from navi_actor.rollout_buffer import RolloutBuffer, Transition
    from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult

    learner = PpoLearner(gamma=0.99)
    buf = RolloutBuffer(capacity=8)
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    obs = DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(8, 4),
        depth=np.ones((1, 8, 4), dtype=np.float32),
        delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
        semantic=np.zeros((1, 8, 4), dtype=np.int32),
        valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
        overhead=np.zeros((256, 256, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=0,
        timestamp=1.0,
    )
    action = Action(
        env_ids=np.array([0], dtype=np.int32),
        linear_velocity=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
        angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        policy_id="test",
        step_id=0,
        timestamp=1.0,
    )
    result = StepResult(step_id=0, done=False, truncated=False, reward=1.0, episode_return=1.0, timestamp=1.0)
    buf.append(Transition(observation=obs, action=action, result=result))
    metrics = learner.train_epoch(buf)
    assert metrics["reward_mean"] == 1.0


def test_ppo_epoch_returns_metrics() -> None:
    """train_ppo_epoch should return PpoMetrics."""
    policy = CognitiveMambaPolicy(embedding_dim=128)
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(64)

    metrics = learner.train_ppo_epoch(
        policy, buf, ppo_epochs=1, minibatch_size=32, seq_len=0,
    )
    assert isinstance(metrics, PpoMetrics)
    assert metrics.policy_loss != 0.0 or metrics.value_loss != 0.0


def test_ppo_epoch_improves_loss() -> None:
    """Multiple PPO epochs should change the loss."""
    policy = CognitiveMambaPolicy(embedding_dim=128)
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(128)

    m1 = learner.train_ppo_epoch(policy, buf, ppo_epochs=1, minibatch_size=64, seq_len=0)
    # Rebuild buffer for second pass (same data)
    buf2 = _fill_buffer(128)
    m2 = learner.train_ppo_epoch(policy, buf2, ppo_epochs=1, minibatch_size=64, seq_len=0)
    # Losses should differ between calls (policy updated)
    assert m1.total_loss != m2.total_loss or m1.policy_loss != m2.policy_loss


def test_ppo_clip_fraction_bounded() -> None:
    """Clip fraction should be between 0 and 1."""
    policy = CognitiveMambaPolicy(embedding_dim=128)
    learner = PpoLearner(clip_ratio=0.2)
    buf = _fill_buffer(64)
    metrics = learner.train_ppo_epoch(
        policy, buf, ppo_epochs=2, minibatch_size=32, seq_len=0,
    )
    assert 0.0 <= metrics.clip_fraction <= 1.0


def test_empty_buffer_returns_zeros() -> None:
    """An empty buffer should return zero metrics."""
    policy = CognitiveMambaPolicy(embedding_dim=128)
    learner = PpoLearner()
    buf = TrajectoryBuffer()
    buf.compute_returns_and_advantages(last_value=0.0)
    metrics = learner.train_ppo_epoch(policy, buf, ppo_epochs=1, minibatch_size=32, seq_len=0)
    assert metrics.total_loss == 0.0
