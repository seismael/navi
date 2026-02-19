"""Training callbacks for logging and checkpointing."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__: list[str] = ["AbstractCallback", "PrintCallback"]


class AbstractCallback(ABC):
    """Base class for training callbacks."""

    @abstractmethod
    def on_step_end(self, loss: float) -> None:
        """Called after each training step."""

    @abstractmethod
    def on_epoch_end(self, avg_loss: float) -> None:
        """Called after each epoch."""


class PrintCallback(AbstractCallback):
    """Simple callback that prints loss to stdout."""

    def __init__(self) -> None:
        self._step = 0

    def on_step_end(self, loss: float) -> None:
        """Print step loss."""
        self._step += 1

    def on_epoch_end(self, avg_loss: float) -> None:
        """Print epoch average loss."""
