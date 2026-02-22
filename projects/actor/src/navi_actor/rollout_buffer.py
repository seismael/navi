"""Rollout buffer primitives for policy optimization."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from navi_contracts import Action, DistanceMatrix, StepResult

__all__: list[str] = [
    "PPOTransition",
    "RolloutBuffer",
    "TrajectoryBuffer",
    "Transition",
]


# ---------------------------------------------------------------------------
# Legacy types (kept as fallback for REINFORCE trainer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """Single transition used by REINFORCE-style learners (legacy)."""

    observation: DistanceMatrix
    action: Action
    result: StepResult


class RolloutBuffer:
    """In-memory transition buffer with fixed capacity (legacy)."""

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


# ---------------------------------------------------------------------------
# PPO trajectory types
# ---------------------------------------------------------------------------


@dataclass
class PPOTransition:
    """Single PPO transition with tensors for on-policy training."""

    observation: Tensor  # (2, Az, El)
    action: Tensor  # (4,)
    log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool = False
    hidden_state: Tensor | None = None


class TrajectoryBuffer:
    """Trajectory-aware buffer for PPO with GAE computation.

    Stores a contiguous rollout of PPOTransitions and provides:
    - GAE(λ) advantage/return computation
    - Minibatch sampling with optional sequential (BPTT) chunks
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._transitions: list[PPOTransition] = []
        self._advantages: Tensor = torch.zeros(0)
        self._returns: Tensor = torch.zeros(0)
        self._finalized: bool = False

    def append(self, transition: PPOTransition) -> None:
        """Append one transition. Resets finalized state."""
        self._transitions.append(transition)
        self._finalized = False

    def clear(self) -> None:
        """Clear all buffered data."""
        self._transitions.clear()
        self._advantages = torch.zeros(0)
        self._returns = torch.zeros(0)
        self._finalized = False

    def __len__(self) -> int:
        return len(self._transitions)

    def compute_returns_and_advantages(
        self,
        last_value: float = 0.0,
    ) -> None:
        """Compute GAE(λ) advantages and discounted returns.

        Args:
            last_value: bootstrap value for the state after the last transition
                        (0.0 if episode ended, V(s_T) otherwise).

        """
        n = len(self._transitions)
        if n == 0:
            self._finalized = True
            return

        advantages = torch.zeros(n, dtype=torch.float32)
        last_gae = 0.0
        prev_value = last_value

        for t in reversed(range(n)):
            tr = self._transitions[t]

            if tr.truncated and not tr.done:
                # Time-limit: episode was artificially ended.  Bootstrap
                # from V(s_T) as a proxy for V(s_{T+1}).  The GAE trace
                # is cut so advantages from the next episode don't leak.
                delta = tr.reward + self.gamma * tr.value - tr.value
                last_gae = delta
            elif tr.done:
                # True termination (collision): future value is 0.
                delta = tr.reward - tr.value
                last_gae = delta
            else:
                # Normal mid-episode step: full GAE recursion.
                delta = tr.reward + self.gamma * prev_value - tr.value
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae
            prev_value = tr.value

        values = torch.tensor(
            [tr.value for tr in self._transitions], dtype=torch.float32
        )
        self._advantages = advantages
        self._returns = advantages + values
        self._finalized = True

    @dataclass
    class MiniBatch:
        """A minibatch of PPO training data."""

        observations: Tensor  # (B, 2, Az, El)
        actions: Tensor  # (B, 4)
        old_log_probs: Tensor  # (B,)
        old_values: Tensor  # (B,)
        advantages: Tensor  # (B,)
        returns: Tensor  # (B,)
        hidden_states: list[Tensor | None] = field(default_factory=list)
        dones: Tensor | None = None  # (n_seqs, seq_len) for BPTT

    def sample_minibatches(
        self,
        batch_size: int = 64,
        seq_len: int = 32,
    ) -> Generator[MiniBatch, None, None]:
        """Yield minibatches for PPO training epochs.

        If seq_len > 0, samples contiguous chunks of length seq_len
        (for BPTT with recurrent policies). Otherwise, samples random indices.

        Args:
            batch_size: number of transitions per minibatch.
            seq_len: sequence length for BPTT (0 for random shuffle).

        Yields:
            MiniBatch instances.

        """
        if not self._finalized:
            msg = "Must call compute_returns_and_advantages() before sampling."
            raise RuntimeError(msg)

        n = len(self._transitions)
        if n == 0:
            return

        # Build flat tensors
        obs = torch.stack([tr.observation for tr in self._transitions])
        acts = torch.stack([tr.action for tr in self._transitions])
        old_lp = torch.tensor(
            [tr.log_prob for tr in self._transitions], dtype=torch.float32
        )
        old_v = torch.tensor(
            [tr.value for tr in self._transitions], dtype=torch.float32
        )
        dones_flat = torch.tensor(
            [tr.done for tr in self._transitions], dtype=torch.bool
        )
        advs = self._advantages
        rets = self._returns

        # Normalize advantages
        if advs.numel() > 1:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        if seq_len > 0 and n >= seq_len:
            # Sequential chunks for BPTT
            starts = list(range(0, n - seq_len + 1, seq_len))
            perm = torch.randperm(len(starts)).tolist()
            for i in range(0, len(perm), max(1, batch_size // seq_len)):
                selected: list[int] = perm[
                    i : i + max(1, batch_size // seq_len)
                ]
                chunk_indices_list: list[int] = []
                chunk_hidden: list[Tensor | None] = []
                chunk_dones: list[Tensor] = []
                for j in selected:
                    s = starts[j]
                    chunk_indices_list.extend(range(s, s + seq_len))
                    # Hidden state at the START of this chunk
                    chunk_hidden.append(
                        self._transitions[s].hidden_state
                    )
                    # Done flags for each step in this chunk
                    chunk_dones.append(dones_flat[s : s + seq_len])
                idx = torch.tensor(chunk_indices_list, dtype=torch.long)
                yield TrajectoryBuffer.MiniBatch(
                    observations=obs[idx],
                    actions=acts[idx],
                    old_log_probs=old_lp[idx],
                    old_values=old_v[idx],
                    advantages=advs[idx],
                    returns=rets[idx],
                    hidden_states=chunk_hidden,
                    dones=torch.stack(chunk_dones),  # (n_seqs, seq_len)
                )
        else:
            # Random shuffle
            indices = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = indices[start : start + batch_size]
                yield TrajectoryBuffer.MiniBatch(
                    observations=obs[idx],
                    actions=acts[idx],
                    old_log_probs=old_lp[idx],
                    old_values=old_v[idx],
                    advantages=advs[idx],
                    returns=rets[idx],
                )
