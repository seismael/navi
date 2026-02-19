"""Minimal PPO-style learner scaffold for Ghost-Matrix actor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from navi_actor.rollout_buffer import RolloutBuffer

__all__: list[str] = ["PpoLearner"]


class PpoLearner:
    """Learner scaffold that computes simple rollout metrics."""

    def __init__(self, gamma: float = 0.99) -> None:
        self._gamma = gamma

    def discounted_return(self, rewards: np.ndarray) -> float:
        """Compute scalar discounted return for one reward vector."""
        running = 0.0
        for reward in rewards[::-1]:
            running = float(reward) + self._gamma * running
        return running

    def train_epoch(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Compute basic epoch metrics from buffered transitions."""
        rewards = buffer.rewards()
        if rewards.size == 0:
            return {"episodes": 0.0, "reward_mean": 0.0, "discounted_return": 0.0}

        return {
            "episodes": float(rewards.size),
            "reward_mean": float(rewards.mean()),
            "discounted_return": self.discounted_return(rewards),
        }
