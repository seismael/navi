"""Rollout buffer primitives for policy optimization."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import cast

import torch
from torch import Tensor

__all__: list[str] = [
    "MultiTrajectoryBuffer",
    "PPOTransition",
    "TrajectoryBuffer",
]


# ---------------------------------------------------------------------------
# PPO trajectory types
# ---------------------------------------------------------------------------


class MultiTrajectoryBuffer:
    """Buffer that manages multiple independent actor trajectories.

    Prevents 'Cross-Actor Bleed' by keeping actor rollout sequences separate
    during BPTT sampling.
    """

    def __init__(
        self,
        n_actors: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        capacity: int | None = None,
    ) -> None:
        self._buffers = {
            i: TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda, capacity=capacity)
            for i in range(n_actors)
        }
        self._n_actors = n_actors

    def append(self, actor_id: int, transition: PPOTransition) -> None:
        self._buffers[actor_id].append(transition)

    def append_batch(
        self,
        *,
        observations: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncateds: Tensor,
        hidden_batch: Tensor | None,
        aux_tensors: Tensor | None,
    ) -> None:
        """Append one rollout step for every actor without Python transition objects."""
        for actor_id in range(self._n_actors):
            hidden_state = None
            if hidden_batch is not None:
                hidden_state = hidden_batch[:, actor_id : actor_id + 1, :].detach().clone()
            aux_tensor = None if aux_tensors is None else aux_tensors[actor_id].detach().clone()
            self._buffers[actor_id].append_fields(
                observation=observations[actor_id].detach().clone(),
                action=actions[actor_id].detach().clone(),
                log_prob=log_probs[actor_id].detach(),
                value=values[actor_id].detach(),
                reward=rewards[actor_id].detach(),
                done=dones[actor_id].detach(),
                truncated=truncateds[actor_id].detach(),
                hidden_state=hidden_state,
                aux_tensor=aux_tensor,
            )

    def compute_returns_and_advantages(self, last_values: Tensor) -> None:
        """Compute GAE for all buffers using per-actor bootstrap values."""
        for i in range(self._n_actors):
            self._buffers[i].compute_returns_and_advantages(last_value=last_values[i].detach())

    def clear(self) -> None:
        for buf in self._buffers.values():
            buf.clear()

    def sample_minibatches(
        self, batch_size: int = 64, seq_len: int = 32
    ) -> Generator[TrajectoryBuffer.MiniBatch, None, None]:
        """Sample minibatches across all actors while preserving sequences."""
        # 1. Build tensor cache for each actor
        for buf in self._buffers.values():
            buf._build_tensor_cache()

        # 2. Stack into (Actor, Time, ...) tensors
        # Assumes all actors have SAME rollout length!
        all_obs = torch.stack([cast("Tensor", b._t_obs) for b in self._buffers.values()]) # (N, L, 3, Az, El)
        all_acts = torch.stack([cast("Tensor", b._t_actions) for b in self._buffers.values()]) # (N, L, 4)
        all_lps = torch.stack([cast("Tensor", b._t_log_probs) for b in self._buffers.values()]) # (N, L)
        all_vals = torch.stack([cast("Tensor", b._t_values) for b in self._buffers.values()]) # (N, L)
        all_advs = torch.stack([b._advantages for b in self._buffers.values()]) # (N, L)
        all_rets = torch.stack([b._returns for b in self._buffers.values()]) # (N, L)
        all_dones = torch.stack([cast("Tensor", b._t_dones) for b in self._buffers.values()]) # (N, L)

        n_actors, rollout_len = all_obs.shape[:2]

        if seq_len > 0:
            # BPTT sampling: (N, L) -> (N * L/seq_len, seq_len) sequences
            n_seqs_per_actor = rollout_len // seq_len
            total_seqs = n_actors * n_seqs_per_actor

            obs_seqs = all_obs[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len, *all_obs.shape[2:])
            acts_seqs = all_acts[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len, -1)
            lps_seqs = all_lps[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len)
            vals_seqs = all_vals[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len)
            advs_seqs = all_advs[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len)
            rets_seqs = all_rets[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len)
            dones_seqs = all_dones[:, :n_seqs_per_actor * seq_len].reshape(total_seqs, seq_len)

            # Shuffle sequences
            perm = torch.randperm(total_seqs)

            seqs_per_minibatch = max(1, batch_size // seq_len)
            for i in range(0, total_seqs, seqs_per_minibatch):
                idx = perm[i : i + seqs_per_minibatch]

                # Hidden states at start of sequences
                # We need to map linear index back to (actor, time)
                h0 = []
                for j in idx:
                    actor_idx = j // n_seqs_per_actor
                    seq_idx = j % n_seqs_per_actor
                    h0.append(self._buffers[int(actor_idx)].hidden_state_at(int(seq_idx * seq_len)))

                yield TrajectoryBuffer.MiniBatch(
                    observations=obs_seqs[idx].reshape(-1, *obs_seqs.shape[2:]),
                    actions=acts_seqs[idx].reshape(-1, acts_seqs.shape[-1]),
                    old_log_probs=lps_seqs[idx].flatten(),
                    old_values=vals_seqs[idx].flatten(),
                    advantages=advs_seqs[idx].flatten(),
                    returns=rets_seqs[idx].flatten(),
                    hidden_states=h0,
                    dones=dones_seqs[idx], # (n_seqs, seq_len)
                )
        else:
            # Flat random shuffle
            flat_indices = torch.randperm(n_actors * rollout_len)
            obs_flat = all_obs.view(-1, *all_obs.shape[2:])
            acts_flat = all_acts.view(-1, all_acts.shape[-1])
            lps_flat = all_lps.flatten()
            vals_flat = all_vals.flatten()
            advs_flat = all_advs.flatten()
            rets_flat = all_rets.flatten()

            for i in range(0, len(flat_indices), batch_size):
                idx = flat_indices[i : i + batch_size]
                yield TrajectoryBuffer.MiniBatch(
                    observations=obs_flat[idx],
                    actions=acts_flat[idx],
                    old_log_probs=lps_flat[idx],
                    old_values=vals_flat[idx],
                    advantages=advs_flat[idx],
                    returns=rets_flat[idx],
                )

    def __len__(self) -> int:
        return sum(len(b) for b in self._buffers.values())


@dataclass
class PPOTransition:
    """Single PPO transition with tensors for on-policy training."""

    observation: Tensor  # (3, Az, El)
    action: Tensor  # (4,)
    log_prob: float
    value: float
    reward: float
    done: bool
    truncated: bool = False
    hidden_state: Tensor | None = None
    aux_tensor: Tensor | None = None


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

    def __init__(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        capacity: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._capacity = capacity
        self._transitions: list[PPOTransition] = []
        self._hidden_states: list[Tensor | None] = []
        self._size = 0
        self._advantages: Tensor = torch.zeros(0)
        self._returns: Tensor = torch.zeros(0)
        self._finalized: bool = False

        # ── Tensor cache (eliminates repeated torch.stack) ──
        self._t_obs: Tensor | None = None
        self._t_actions: Tensor | None = None
        self._t_log_probs: Tensor | None = None
        self._t_values: Tensor | None = None
        self._t_dones: Tensor | None = None
        self._t_aux: Tensor | None = None
        self._t_rewards: Tensor | None = None
        self._t_truncated: Tensor | None = None
        self._t_cached: bool = False

    def _uses_tensor_storage(self) -> bool:
        return self._capacity is not None

    def _allocate_storage(
        self,
        *,
        observation: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        truncated: Tensor,
        aux_tensor: Tensor | None,
    ) -> None:
        if self._capacity is None:
            return
        if self._t_obs is None:
            self._t_obs = torch.empty((self._capacity, *observation.shape), dtype=observation.dtype, device=observation.device)
            self._t_actions = torch.empty((self._capacity, *action.shape), dtype=action.dtype, device=action.device)
            self._t_log_probs = torch.empty((self._capacity,), dtype=log_prob.dtype, device=log_prob.device)
            self._t_values = torch.empty((self._capacity,), dtype=value.dtype, device=value.device)
            self._t_rewards = torch.empty((self._capacity,), dtype=reward.dtype, device=reward.device)
            self._t_dones = torch.empty((self._capacity,), dtype=done.dtype, device=done.device)
            self._t_truncated = torch.empty((self._capacity,), dtype=truncated.dtype, device=truncated.device)
            if aux_tensor is not None:
                self._t_aux = torch.empty((self._capacity, *aux_tensor.shape), dtype=aux_tensor.dtype, device=aux_tensor.device)

    def append_fields(
        self,
        *,
        observation: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        truncated: Tensor,
        hidden_state: Tensor | None,
        aux_tensor: Tensor | None,
    ) -> None:
        """Append transition fields directly into tensor storage."""
        if not self._uses_tensor_storage():
            self.append(
                PPOTransition(
                    observation=observation,
                    action=action,
                    log_prob=float(log_prob.detach().to(device="cpu")),
                    value=float(value.detach().to(device="cpu")),
                    reward=float(reward.detach().to(device="cpu")),
                    done=bool(done.detach().to(device="cpu")),
                    truncated=bool(truncated.detach().to(device="cpu")),
                    hidden_state=hidden_state,
                    aux_tensor=aux_tensor,
                )
            )
            return

        if self._capacity is None:
            raise RuntimeError("Tensor storage requires a configured capacity")
        if self._size >= self._capacity:
            raise RuntimeError("TrajectoryBuffer capacity exceeded")
        self._allocate_storage(
            observation=observation,
            action=action,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done,
            truncated=truncated,
            aux_tensor=aux_tensor,
        )
        assert self._t_obs is not None
        assert self._t_actions is not None
        assert self._t_log_probs is not None
        assert self._t_values is not None
        assert self._t_rewards is not None
        assert self._t_dones is not None
        assert self._t_truncated is not None

        idx = self._size
        self._t_obs[idx].copy_(observation)
        self._t_actions[idx].copy_(action)
        self._t_log_probs[idx].copy_(log_prob.reshape(()))
        self._t_values[idx].copy_(value.reshape(()))
        self._t_rewards[idx].copy_(reward.reshape(()))
        self._t_dones[idx].copy_(done.reshape(()))
        self._t_truncated[idx].copy_(truncated.reshape(()))
        if aux_tensor is not None:
            if self._t_aux is None:
                self._t_aux = torch.empty((self._capacity, *aux_tensor.shape), dtype=aux_tensor.dtype, device=aux_tensor.device)
            self._t_aux[idx].copy_(aux_tensor)
        self._hidden_states.append(hidden_state)
        self._size += 1
        self._finalized = False
        self._t_cached = False

    # ── Mutation (invalidates cache) ─────────────────────────────

    def append(self, transition: PPOTransition) -> None:
        """Append one transition. Resets finalized state."""
        if self._uses_tensor_storage():
            self.append_fields(
                observation=transition.observation,
                action=transition.action,
                log_prob=torch.tensor(transition.log_prob, dtype=torch.float32, device=transition.action.device),
                value=torch.tensor(transition.value, dtype=torch.float32, device=transition.action.device),
                reward=torch.tensor(transition.reward, dtype=torch.float32, device=transition.action.device),
                done=torch.tensor(transition.done, dtype=torch.bool, device=transition.action.device),
                truncated=torch.tensor(transition.truncated, dtype=torch.bool, device=transition.action.device),
                hidden_state=transition.hidden_state,
                aux_tensor=transition.aux_tensor,
            )
            return
        self._transitions.append(transition)
        self._hidden_states.append(transition.hidden_state)
        self._size += 1
        self._finalized = False
        self._t_cached = False

    def clear(self) -> None:
        """Clear all buffered data."""
        self._transitions.clear()
        self._hidden_states.clear()
        self._size = 0
        self._advantages = torch.zeros(0)
        self._returns = torch.zeros(0)
        self._finalized = False
        self._t_cached = False
        self._t_obs = None
        self._t_actions = None
        self._t_log_probs = None
        self._t_values = None
        self._t_dones = None
        self._t_aux = None
        self._t_rewards = None
        self._t_truncated = None

    def extend_from(self, other: TrajectoryBuffer) -> None:
        """Append all transitions from *other*, inserting a done boundary.

        Marks the last transition of ``self`` as ``done=True`` so that
        BPTT chunks never straddle two different actors' trajectories.
        Also concatenates pre-computed advantages and returns.
        """
        if len(other) == 0:
            return

        if self._uses_tensor_storage() or other._uses_tensor_storage():
            if len(self) > 0 and self._t_dones is not None and len(self) > 0:
                self._t_dones[len(self) - 1] = True
            for idx in range(len(other)):
                self.append_fields(
                    observation=other.observation_at(idx),
                    action=other.action_at(idx),
                    log_prob=other.log_prob_at(idx),
                    value=other.value_at(idx),
                    reward=other.reward_at(idx),
                    done=other.done_at(idx),
                    truncated=other.truncated_at(idx),
                    hidden_state=other.hidden_state_at(idx),
                    aux_tensor=other.aux_tensor_at(idx),
                )
            self._advantages = torch.cat([self._advantages, other._advantages])
            self._returns = torch.cat([self._returns, other._returns])
            self._t_cached = False
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
                    aux_tensor=tail.aux_tensor,
                )

        self._transitions.extend(other._transitions)
        self._hidden_states.extend(other._hidden_states)
        self._size += len(other)
        self._advantages = torch.cat([self._advantages, other._advantages])
        self._returns = torch.cat([self._returns, other._returns])
        self._t_cached = False

    def __len__(self) -> int:
        return self._size

    def hidden_state_at(self, index: int) -> Tensor | None:
        return self._hidden_states[index]

    def observation_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_obs is not None
            return self._t_obs[index]
        return self._transitions[index].observation

    def action_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_actions is not None
            return self._t_actions[index]
        return self._transitions[index].action

    def log_prob_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_log_probs is not None
            return self._t_log_probs[index]
        return torch.tensor(self._transitions[index].log_prob, dtype=torch.float32)

    def value_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_values is not None
            return self._t_values[index]
        return torch.tensor(self._transitions[index].value, dtype=torch.float32)

    def reward_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_rewards is not None
            return self._t_rewards[index]
        return torch.tensor(self._transitions[index].reward, dtype=torch.float32)

    def done_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_dones is not None
            return self._t_dones[index]
        return torch.tensor(self._transitions[index].done, dtype=torch.bool)

    def truncated_at(self, index: int) -> Tensor:
        if self._uses_tensor_storage():
            assert self._t_truncated is not None
            return self._t_truncated[index]
        return torch.tensor(self._transitions[index].truncated, dtype=torch.bool)

    def aux_tensor_at(self, index: int) -> Tensor | None:
        if self._uses_tensor_storage():
            if self._t_aux is None:
                return None
            return self._t_aux[index]
        return self._transitions[index].aux_tensor

    # ── Tensor cache ─────────────────────────────────────────────

    def _build_tensor_cache(self) -> None:
        """Build flat tensors from transition list (once per PPO update).

        Called lazily from :meth:`sample_minibatches`.  Reused across
        all PPO epochs without redundant ``torch.stack`` calls.
        """
        if self._t_cached:
            return
        n = len(self)
        if n == 0:
            return
        if self._uses_tensor_storage():
            assert self._t_obs is not None
            assert self._t_actions is not None
            assert self._t_log_probs is not None
            assert self._t_values is not None
            assert self._t_dones is not None
            self._t_obs = self._t_obs[:n]
            self._t_actions = self._t_actions[:n]
            self._t_log_probs = self._t_log_probs[:n]
            self._t_values = self._t_values[:n]
            self._t_dones = self._t_dones[:n]
            if self._t_aux is not None:
                self._t_aux = self._t_aux[:n]
            self._t_cached = True
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
        self._t_aux = (
            torch.stack([
                cast("Tensor", tr.aux_tensor) for tr in self._transitions
            ])
            if self._transitions[0].aux_tensor is not None
            else None
        )
        self._t_cached = True

    def compute_returns_and_advantages(
        self,
        last_value: float | Tensor = 0.0,
    ) -> None:
        """Compute GAE(λ) advantages and discounted returns.

        Args:
            last_value: bootstrap value for the state after the last transition
                        (0.0 if episode ended, V(s_T) otherwise).

        """
        n = len(self)
        if n == 0:
            self._finalized = True
            return

        value_device = self.value_at(0).device
        bootstrap = torch.as_tensor(last_value, dtype=torch.float32, device=value_device).reshape(())
        advantages = torch.zeros(n, dtype=torch.float32, device=value_device)
        last_gae = torch.zeros((), dtype=torch.float32, device=value_device)
        if self._uses_tensor_storage():
            assert self._t_rewards is not None
            assert self._t_values is not None
            assert self._t_dones is not None
            assert self._t_truncated is not None
            rewards = self._t_rewards[:n].to(device=value_device, dtype=torch.float32)
            values = self._t_values[:n].to(device=value_device, dtype=torch.float32)
            done_flags = self._t_dones[:n].to(device=value_device)
            truncated_flags = self._t_truncated[:n].to(device=value_device)
        else:
            rewards = torch.tensor(
                [tr.reward for tr in self._transitions],
                dtype=torch.float32,
                device=value_device,
            )
            values = torch.tensor(
                [tr.value for tr in self._transitions],
                dtype=torch.float32,
                device=value_device,
            )
            done_flags = torch.tensor(
                [tr.done for tr in self._transitions],
                dtype=torch.bool,
                device=value_device,
            )
            truncated_flags = torch.tensor(
                [tr.truncated for tr in self._transitions],
                dtype=torch.bool,
                device=value_device,
            )

        prev_value = bootstrap
        continue_mask = (~done_flags & ~truncated_flags).to(dtype=torch.float32)
        truncation_mask = (truncated_flags & ~done_flags).to(dtype=torch.float32)
        done_mask = done_flags.to(dtype=torch.float32)

        for t in reversed(range(n)):
            reward_t = rewards[t]
            value_t = values[t]
            delta = reward_t + self.gamma * prev_value - value_t
            terminal_delta = reward_t - value_t
            continued_gae = delta + self.gamma * self.gae_lambda * last_gae
            last_gae = (
                truncation_mask[t] * delta
                + done_mask[t] * terminal_delta
                + continue_mask[t] * continued_gae
            )
            advantages[t] = last_gae
            prev_value = value_t
        self._advantages = advantages
        self._returns = advantages + values
        self._finalized = True

    @dataclass
    class MiniBatch:
        """A minibatch of PPO training data."""

        observations: Tensor  # (B, 3, Az, El)
        actions: Tensor  # (B, 4)
        old_log_probs: Tensor  # (B,)
        old_values: Tensor  # (B,)
        advantages: Tensor  # (B,)
        returns: Tensor  # (B,)
        hidden_states: list[Tensor | None] = field(default_factory=list)
        dones: Tensor | None = None  # (n_seqs, seq_len) for BPTT
        aux_tensors: Tensor | None = None  # (B, 3) or (n_seqs, seq_len, 3)

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

        n = len(self)
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
        aux_flat = self._t_aux
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
                        self.hidden_state_at(s)
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
                    aux_tensors=aux_flat[idx] if aux_flat is not None else None,
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
                    aux_tensors=aux_flat[idx] if aux_flat is not None else None,
                )
