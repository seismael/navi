"""Ghost-Matrix training loop built on rollout buffer + PPO learner scaffold."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from navi_actor.learner_ppo import PpoLearner
    from navi_actor.rollout_buffer import RolloutBuffer, Transition
    from navi_actor.training.callbacks import AbstractCallback

__all__: list[str] = ["TrainingLoop"]


class TrainingLoop:
    """Collect transitions and run learner epochs with callback hooks."""

    def __init__(
        self,
        learner: PpoLearner,
        buffer: RolloutBuffer,
        callbacks: Sequence[AbstractCallback] = (),
    ) -> None:
        self._learner = learner
        self._buffer = buffer
        self._callbacks = list(callbacks)

    def push(self, transition: Transition) -> None:
        """Push one transition into the rollout buffer."""
        self._buffer.append(transition)

    def train_epoch(self) -> dict[str, float]:
        """Run one learner epoch and notify callbacks."""
        metrics = self._learner.train_epoch(self._buffer)
        for cb in self._callbacks:
            cb.on_epoch_end(float(metrics["reward_mean"]))
        self._buffer.clear()
        return metrics
