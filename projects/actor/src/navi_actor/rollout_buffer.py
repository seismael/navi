"""Rollout buffer primitives for policy optimization."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__: list[str] = [
    "PPOTransition",
    "TrajectoryBuffer",
]


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

    Tensor data is cached lazily on the first call to
    :meth:`sample_minibatches` and reused across subsequent PPO
    epochs.  The cache is invalidated automatically on mutation
    (``append``, ``clear``, ``extend_from``).
    """

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._transitions: list[PPOTransition] = []
        self._advantages: Tensor = torch.zeros(0)
        self._returns: Tensor = torch.zeros(0)
        self._finalized: bool = False

        # ── Tensor cache (eliminates repeated torch.stack) ──
        self._t_obs: Tensor | None = None
        self._t_actions: Tensor | None = None
        self._t_log_probs: Tensor | None = None
        self._t_values: Tensor | None = None
        self._t_dones: Tensor | None = None
        self._t_cached: bool = False

    # ── Mutation (invalidates cache) ─────────────────────────────

    def append(self, transition: PPOTransition) -> None:
        """Append one transition. Resets finalized state."""
        self._transitions.append(transition)
        self._finalized = False
        self._t_cached = False

    def clear(self) -> None:
        """Clear all buffered data."""
        self._transitions.clear()
        self._advantages = torch.zeros(0)
        self._returns = torch.zeros(0)
        self._finalized = False
        self._t_cached = False
        self._t_obs = None
        self._t_actions = None
        self._t_log_probs = None
        self._t_values = None
        self._t_dones = None

    def extend_from(self, other: TrajectoryBuffer) -> None:
        """Append all transitions from *other*, inserting a done boundary.

        Marks the last transition of ``self`` as ``done=True`` so that
        BPTT chunks never straddle two different actors' trajectories.
        Also concatenates pre-computed advantages and returns.
        """
        if len(other) == 0:
            return

        # Insert synthetic done boundary
        if len(self._transitions) > 0:
            tail = self._transitions[-1]
            if not tail.done:
                self._transitions[-1] = PPOTransition(
                    observation=tail.observation,
                    action=tail.action,
                    log_prob=tail.log_prob,
                    value=tail.value,
                    reward=tail.reward,
                    done=True,
                    truncated=tail.truncated,
                    hidden_state=tail.hidden_state,
                )

        self._transitions.extend(other._transitions)
        self._advantages = torch.cat([self._advantages, other._advantages])
        self._returns = torch.cat([self._returns, other._returns])
        self._t_cached = False

    def __len__(self) -> int:
        return len(self._transitions)

    # ── Tensor cache ─────────────────────────────────────────────

    def _build_tensor_cache(self) -> None:
        """Build flat tensors from transition list (once per PPO update).

        Called lazily from :meth:`sample_minibatches`.  Reused across
        all PPO epochs without redundant ``torch.stack`` calls.
        """
        if self._t_cached:
            return
        n = len(self._transitions)
        if n == 0:
            return
        self._t_obs = torch.stack(
            [tr.observation for tr in self._transitions],
        )
        self._t_actions = torch.stack(
            [tr.action for tr in self._transitions],
        )
        self._t_log_probs = torch.tensor(
            [tr.log_prob for tr in self._transitions], dtype=torch.float32,
        )
        self._t_values = torch.tensor(
            [tr.value for tr in self._transitions], dtype=torch.float32,
        )
        self._t_dones = torch.tensor(
            [tr.done for tr in self._transitions], dtype=torch.bool,
        )
        self._t_cached = True

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

        # Build / reuse tensor cache (no repeated torch.stack)
        self._build_tensor_cache()
        assert self._t_obs is not None
        assert self._t_actions is not None
        assert self._t_log_probs is not None
        assert self._t_values is not None
        assert self._t_dones is not None

        obs = self._t_obs
        acts = self._t_actions
        old_lp = self._t_log_probs
        old_v = self._t_values
        dones_flat = self._t_dones
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
