"""Rollout buffer primitives for policy optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from navi_contracts import Action, DistanceMatrix, StepResult

__all__: list[str] = ["RolloutBuffer", "Transition"]


@dataclass(frozen=True)
class Transition:
    """Single transition used by PPO-style learners."""

    observation: DistanceMatrix
    action: Action
    result: StepResult


class RolloutBuffer:
    """In-memory transition buffer with fixed capacity."""

    def __init__(self, capacity: int = 4096) -> None:
        self._capacity = max(1, capacity)
        self._items: list[Transition] = []

    def append(self, item: Transition) -> None:
        """Append one transition, dropping oldest when full."""
        self._items.append(item)
        if len(self._items) > self._capacity:
            self._items.pop(0)

    def clear(self) -> None:
        """Clear all buffered transitions."""
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def rewards(self) -> np.ndarray:
        """Return rewards as float32 array."""
        if not self._items:
            return np.zeros((0,), dtype=np.float32)
        return np.array([item.result.reward for item in self._items], dtype=np.float32)
