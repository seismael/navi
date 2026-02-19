"""Unit tests for online spherical trainer."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from navi_actor.training.online import EvaluationMetrics, EvaluationPoint, OnlineSphericalTrainer
from navi_contracts import DistanceMatrix, RobotPose, StepResult


def _obs(step_id: int, depth_value: float = 0.8, x: float = 0.0, z: float = 0.0) -> DistanceMatrix:
    pose = RobotPose(x=x, y=0.0, z=z, roll=0.0, pitch=0.0, yaw=0.0, timestamp=float(step_id))
    depth = np.full((1, 32, 16), depth_value, dtype=np.float32)
    valid = np.ones((1, 32, 16), dtype=np.bool_)
    semantic = np.zeros((1, 32, 16), dtype=np.int32)
    return DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(32, 16),
        depth=depth,
        delta_depth=np.zeros_like(depth),
        semantic=semantic,
        valid_mask=valid,
        overhead=np.zeros((128, 128, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=step_id,
        timestamp=float(step_id),
    )


def test_build_features_uses_full_spherical_bins() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    features = trainer._build_features(_obs(step_id=0))
    assert features.shape == (13,)
    assert np.all(np.isfinite(features))
    assert np.all((features >= 0.0) & (features <= 1.0))


def test_policy_update_changes_weights() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    features = trainer._build_features(_obs(step_id=0))
    before = trainer._w_forward.copy()

    trainer._update_policy(
        features=features,
        forward_cmd=0.7,
        yaw_cmd=0.1,
        forward_mean=0.4,
        yaw_mean=0.0,
        reward=1.0,
    )

    assert not np.allclose(before, trainer._w_forward)


def test_collision_reward_penalizes_blocked_motion() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    obs = _obs(step_id=1, depth_value=0.7, x=0.0, z=0.0)
    next_obs = _obs(step_id=2, depth_value=0.02, x=0.0, z=0.0)

    reward, collided, _is_novel = trainer._compute_training_reward(obs, next_obs, forward_cmd=0.8, yaw_cmd=0.0)
    assert collided
    assert reward < 0.0


def test_novelty_reward_is_higher_for_unseen_cell() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    obs = _obs(step_id=1, depth_value=0.7, x=0.0, z=0.0)
    next_obs = _obs(step_id=2, depth_value=0.7, x=2.2, z=0.0)
    trainer._visited_cells = {trainer._pose_cell(obs)}

    reward_first, _collided_first, is_novel_first = trainer._compute_training_reward(
        obs,
        next_obs,
        forward_cmd=0.2,
        yaw_cmd=0.0,
    )
    reward_second, _collided_second, is_novel_second = trainer._compute_training_reward(
        obs,
        next_obs,
        forward_cmd=0.2,
        yaw_cmd=0.0,
    )

    assert is_novel_first
    assert not is_novel_second
    assert reward_first > reward_second


def test_periodic_checkpoint_path_creates_directory() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    out_dir = Path(".tmp_test_outputs") / f"ckpt_{uuid4().hex}"

    ckpt_path = trainer._periodic_checkpoint_path(
        checkpoint_dir=str(out_dir),
        checkpoint_prefix="policy",
        step=123,
    )

    assert out_dir.exists()
    assert ckpt_path.endswith("policy_0000123.npz")


def test_train_triggers_periodic_checkpoint_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")

    trainer._sub_socket = object()  # type: ignore[assignment]
    trainer._step_socket = object()  # type: ignore[assignment]

    obs_a = _obs(step_id=1, depth_value=0.7, x=0.0, z=0.0)
    obs_b = _obs(step_id=2, depth_value=0.7, x=0.2, z=0.0)
    obs_cycle = [obs_a, obs_b]
    state = {"i": 0}

    def fake_recv_latest_matrix(timeout_ms: int) -> DistanceMatrix | None:
        del timeout_ms
        idx = state["i"]
        state["i"] = (idx + 1) % len(obs_cycle)
        return obs_cycle[idx]

    def fake_request_step(_action: object, step_id: int) -> StepResult:
        return StepResult(
            step_id=step_id,
            done=False,
            truncated=False,
            reward=0.0,
            episode_return=0.0,
            timestamp=0.0,
        )

    saved_paths: list[str] = []

    def fake_save_checkpoint(path: str) -> None:
        saved_paths.append(path)

    def fake_eval(episodes: int, horizon: int) -> EvaluationMetrics:
        assert episodes == 1
        assert horizon == 5
        return EvaluationMetrics(
            episodes=episodes,
            horizon=horizon,
            reward_mean=0.25,
            collision_rate=0.1,
            novelty_rate=0.6,
            coverage_mean=0.4,
        )

    monkeypatch.setattr(trainer, "_recv_latest_matrix", fake_recv_latest_matrix)
    monkeypatch.setattr(trainer, "_request_step", fake_request_step)
    monkeypatch.setattr(trainer, "_update_policy", lambda *args: None)
    monkeypatch.setattr(trainer, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(trainer, "evaluate", fake_eval)

    metrics = trainer.train(
        steps=6,
        checkpoint_every=3,
        checkpoint_dir=".tmp_test_outputs/train_ckpts",
        checkpoint_prefix="policy",
        eval_every=3,
        eval_episodes=1,
        eval_horizon=5,
    )

    assert len(saved_paths) == 2
    assert saved_paths[0].endswith("policy_0000003.npz")
    assert saved_paths[1].endswith("policy_0000006.npz")
    assert np.isclose(metrics.eval_reward_mean, 0.25)
    assert np.isclose(metrics.eval_novelty_rate, 0.6)
    assert len(metrics.eval_history) == 2


def test_save_eval_csv_writes_rows() -> None:
    trainer = OnlineSphericalTrainer(sub_address="tcp://localhost:1", step_endpoint="tcp://localhost:2")
    csv_path = Path(".tmp_test_outputs") / f"eval_{uuid4().hex}.csv"
    history = [
        EvaluationPoint(
            step=20,
            reward_mean=0.5,
            collision_rate=0.2,
            novelty_rate=0.7,
            coverage_mean=0.4,
        ),
        EvaluationPoint(
            step=40,
            reward_mean=0.8,
            collision_rate=0.1,
            novelty_rate=0.8,
            coverage_mean=0.5,
        ),
    ]

    trainer.save_eval_csv(str(csv_path), history)

    text = csv_path.read_text(encoding="utf-8")
    assert "step,reward_mean,collision_rate,novelty_rate,coverage_mean" in text
    assert "20,0.500000,0.200000,0.700000,0.400000" in text
    assert "40,0.800000,0.100000,0.800000,0.500000" in text
