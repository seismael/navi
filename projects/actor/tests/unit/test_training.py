"""Tests for rollout buffer, PPO learner scaffold, and training loop."""

from __future__ import annotations

import numpy as np

from navi_actor.learner_ppo import PpoLearner
from navi_actor.rollout_buffer import RolloutBuffer, Transition
from navi_actor.training.callbacks import PrintCallback
from navi_actor.training.loop import TrainingLoop
from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult


def _transition(step_id: int, reward: float) -> Transition:
    pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)
    depth = np.ones((1, 8, 4), dtype=np.float32)
    obs = DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(8, 4),
        depth=depth,
        delta_depth=np.zeros_like(depth),
        semantic=np.zeros((1, 8, 4), dtype=np.int32),
        valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
        overhead=np.zeros((256, 256, 3), dtype=np.uint8),
        robot_pose=pose,
        step_id=step_id,
        timestamp=1.0,
    )
    action = Action(
        env_ids=np.array([0], dtype=np.int32),
        linear_velocity=np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
        angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        policy_id="test",
        step_id=step_id,
        timestamp=1.0,
    )
    result = StepResult(
        step_id=step_id,
        done=False,
        truncated=False,
        reward=reward,
        episode_return=reward,
        timestamp=1.0,
    )
    return Transition(observation=obs, action=action, result=result)


def test_learner_discounted_return() -> None:
    learner = PpoLearner(gamma=0.9)
    value = learner.discounted_return(np.array([1.0, 1.0], dtype=np.float32))
    assert value > 1.8


def test_training_loop_clears_buffer() -> None:
    learner = PpoLearner()
    buffer = RolloutBuffer(capacity=8)
    callback = PrintCallback()
    loop = TrainingLoop(learner=learner, buffer=buffer, callbacks=[callback])

    loop.push(_transition(step_id=1, reward=0.5))
    loop.push(_transition(step_id=2, reward=1.0))

    metrics = loop.train_epoch()
    assert metrics["episodes"] == 2.0
    assert metrics["reward_mean"] > 0.0
    assert len(buffer) == 0
