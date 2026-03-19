"""Rollout buffer primitives for policy optimization."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

__all__: list[str] = [
    "MultiTrajectoryBuffer",
    "PPOTransition",
    "TrajectoryBuffer",
]


def _normalize_advantages_once(advantages: Tensor) -> Tensor:
    """Return a cached normalized advantage tensor for PPO sampling reuse."""
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


# ---------------------------------------------------------------------------
# PPO trajectory types
# ---------------------------------------------------------------------------


class MultiTrajectoryBuffer:
    """Canonical batched PPO buffer for multiple actor trajectories."""

    def __init__(
        self,
        n_actors: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        capacity: int | None = None,
    ) -> None:
        if capacity is None:
            raise ValueError(
                "MultiTrajectoryBuffer requires capacity on the canonical batched path"
            )
        self._capacity = capacity
        self._n_actors = n_actors
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._cache_valid = False
        self._all_obs: Tensor | None = None
        self._all_actions: Tensor | None = None
        self._all_log_probs: Tensor | None = None
        self._all_values: Tensor | None = None
        self._all_advantages: Tensor | None = None
        self._all_advantages_normalized: Tensor | None = None
        self._all_returns: Tensor | None = None
        self._all_dones: Tensor | None = None
        self._all_aux: Tensor | None = None
        self._sequence_view_seq_len: int | None = None
        self._sequence_views: dict[str, Tensor] = {}
        self._batched_actor_step_counts: Tensor | None = None
        self._has_aux_for_rollout = False
        self._batch_obs: Tensor | None = None
        self._batch_actions: Tensor | None = None
        self._batch_log_probs: Tensor | None = None
        self._batch_values: Tensor | None = None
        self._batch_rewards: Tensor | None = None
        self._batch_dones: Tensor | None = None
        self._batch_truncateds: Tensor | None = None
        self._batch_aux: Tensor | None = None
        self._batch_advantages: Tensor | None = None
        self._batch_returns: Tensor | None = None

    def _invalidate_views(self) -> None:
        self._cache_valid = False
        self._sequence_view_seq_len = None
        self._sequence_views = {}

    def _allocate_batched_storage(
        self,
        *,
        observations: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncateds: Tensor,
        aux_tensors: Tensor | None,
    ) -> None:
        if self._batch_obs is None:
            self._batch_obs = torch.empty(
                (self._n_actors, self._capacity, *observations.shape[1:]),
                dtype=observations.dtype,
                device=observations.device,
            )
            self._batch_actions = torch.empty(
                (self._n_actors, self._capacity, *actions.shape[1:]),
                dtype=actions.dtype,
                device=actions.device,
            )
            self._batch_log_probs = torch.empty(
                (self._n_actors, self._capacity),
                dtype=log_probs.dtype,
                device=log_probs.device,
            )
            self._batch_values = torch.empty(
                (self._n_actors, self._capacity),
                dtype=values.dtype,
                device=values.device,
            )
            self._batch_rewards = torch.empty(
                (self._n_actors, self._capacity),
                dtype=rewards.dtype,
                device=rewards.device,
            )
            self._batch_dones = torch.empty(
                (self._n_actors, self._capacity),
                dtype=dones.dtype,
                device=dones.device,
            )
            self._batch_truncateds = torch.empty(
                (self._n_actors, self._capacity),
                dtype=truncateds.dtype,
                device=truncateds.device,
            )
        if aux_tensors is not None and self._batch_aux is None:
            self._batch_aux = torch.zeros(
                (self._n_actors, self._capacity, *aux_tensors.shape[1:]),
                dtype=aux_tensors.dtype,
                device=aux_tensors.device,
            )
        if self._batched_actor_step_counts is None:
            self._batched_actor_step_counts = torch.zeros(
                (self._n_actors,),
                dtype=torch.int64,
                device=observations.device,
            )

    def _ensure_batched_advantage_storage(self, device: torch.device) -> None:
        if self._batch_advantages is None:
            self._batch_advantages = torch.empty(
                (self._n_actors, self._capacity),
                dtype=torch.float32,
                device=device,
            )
            self._batch_returns = torch.empty(
                (self._n_actors, self._capacity),
                dtype=torch.float32,
                device=device,
            )

    def _coerce_actor_indices(
        self, actor_indices: Tensor | None, batch_size: int, *, device: torch.device
    ) -> Tensor:
        if actor_indices is None:
            if batch_size != self._n_actors:
                raise RuntimeError(
                    f"append_batch without actor_indices requires batch size {self._n_actors}; got {batch_size}",
                )
            return torch.arange(self._n_actors, device=device, dtype=torch.int64)

        indices = actor_indices.to(device=device, dtype=torch.int64).reshape(-1)
        if int(indices.numel()) != batch_size:
            raise RuntimeError(
                f"actor_indices length {int(indices.numel())} does not match batch size {batch_size}",
            )
        if batch_size == 0:
            return indices
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= self._n_actors:
            raise RuntimeError(f"actor_indices must stay within [0, {self._n_actors - 1}]")
        if int(torch.unique(indices).numel()) != batch_size:
            raise RuntimeError("actor_indices must be unique within one append_batch call")
        return indices

    def _require_rollout_tensor_device_alignment(
        self,
        *,
        observations: Tensor,
        actions: Tensor,
        log_probs: Tensor,
        values: Tensor,
        rewards: Tensor,
        dones: Tensor,
        truncateds: Tensor,
        aux_tensors: Tensor | None,
    ) -> None:
        device = observations.device
        tensor_fields: tuple[tuple[str, Tensor], ...] = (
            ("actions", actions),
            ("log_probs", log_probs),
            ("values", values),
            ("rewards", rewards),
            ("dones", dones),
            ("truncateds", truncateds),
        )
        if aux_tensors is not None:
            tensor_fields += (("aux_tensors", aux_tensors),)

        for field_name, tensor in tensor_fields:
            if tensor.device != device:
                raise RuntimeError(
                    "MultiTrajectoryBuffer.append_batch requires all rollout tensors on one device; "
                    f"observations are on {device}, but {field_name} is on {tensor.device}",
                )

        if self._batch_obs is not None and self._batch_obs.device != device:
            raise RuntimeError(
                "MultiTrajectoryBuffer batched storage device is fixed for the active rollout; "
                f"storage is on {self._batch_obs.device}, incoming observations are on {device}",
            )

    def _required_batched_rollout_len(self) -> int:
        if self._batched_actor_step_counts is None:
            return 0
        if int(self._batched_actor_step_counts.numel()) == 0:
            return 0
        min_count = int(self._batched_actor_step_counts.min().item())
        max_count = int(self._batched_actor_step_counts.max().item())
        if min_count != max_count:
            raise RuntimeError(
                "MultiTrajectoryBuffer batched storage requires equal per-actor rollout lengths before PPO update",
            )
        return max_count

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
        aux_tensors: Tensor | None,
        actor_indices: Tensor | None = None,
    ) -> None:
        """Append one rollout step for every actor without Python transition objects."""
        self._require_rollout_tensor_device_alignment(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds,
            aux_tensors=aux_tensors,
        )
        self._allocate_batched_storage(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds,
            aux_tensors=aux_tensors,
        )
        assert self._batch_obs is not None
        assert self._batch_actions is not None
        assert self._batch_log_probs is not None
        assert self._batch_values is not None
        assert self._batch_rewards is not None
        assert self._batch_dones is not None
        assert self._batch_truncateds is not None
        assert self._batched_actor_step_counts is not None

        actor_indices_t = self._coerce_actor_indices(
            actor_indices, observations.shape[0], device=observations.device
        )
        step_indices = self._batched_actor_step_counts.index_select(0, actor_indices_t)
        if bool((step_indices >= self._capacity).any()):
            raise RuntimeError("MultiTrajectoryBuffer capacity exceeded")

        self._batch_obs[actor_indices_t, step_indices] = observations.detach()
        self._batch_actions[actor_indices_t, step_indices] = actions.detach()
        self._batch_log_probs[actor_indices_t, step_indices] = log_probs.detach()
        self._batch_values[actor_indices_t, step_indices] = values.detach()
        self._batch_rewards[actor_indices_t, step_indices] = rewards.detach()
        self._batch_dones[actor_indices_t, step_indices] = dones.detach()
        self._batch_truncateds[actor_indices_t, step_indices] = truncateds.detach()

        if aux_tensors is not None:
            assert self._batch_aux is not None
            self._batch_aux[actor_indices_t, step_indices] = aux_tensors.detach()
            self._has_aux_for_rollout = True
        elif self._batch_aux is not None:
            self._batch_aux[actor_indices_t, step_indices] = 0

        self._batched_actor_step_counts[actor_indices_t] = step_indices + 1
        self._invalidate_views()

    def compute_returns_and_advantages(self, last_values: Tensor) -> None:
        """Compute GAE for all buffers using per-actor bootstrap values."""
        assert self._batch_rewards is not None
        assert self._batch_values is not None
        assert self._batch_dones is not None
        assert self._batch_truncateds is not None
        rollout_len = self._required_batched_rollout_len()
        if rollout_len == 0:
            self._invalidate_views()
            return
        device = self._batch_values.device
        rewards = self._batch_rewards[:, :rollout_len].to(device=device, dtype=torch.float32)
        values = self._batch_values[:, :rollout_len].to(device=device, dtype=torch.float32)
        done_flags = self._batch_dones[:, :rollout_len].to(device=device)
        truncated_flags = self._batch_truncateds[:, :rollout_len].to(device=device)
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros((self._n_actors,), dtype=torch.float32, device=device)
        prev_value = (
            last_values.detach().to(device=device, dtype=torch.float32).reshape(self._n_actors)
        )
        continue_mask = (~done_flags & ~truncated_flags).to(dtype=torch.float32)
        truncation_mask = (truncated_flags & ~done_flags).to(dtype=torch.float32)
        done_mask = done_flags.to(dtype=torch.float32)

        for t in reversed(range(rollout_len)):
            reward_t = rewards[:, t]
            value_t = values[:, t]
            delta = reward_t + self._gamma * prev_value - value_t
            terminal_delta = reward_t - value_t
            continued_gae = delta + self._gamma * self._gae_lambda * last_gae
            last_gae = (
                truncation_mask[:, t] * delta
                + done_mask[:, t] * terminal_delta
                + continue_mask[:, t] * continued_gae
            )
            advantages[:, t] = last_gae
            prev_value = value_t

        self._ensure_batched_advantage_storage(device)
        assert self._batch_advantages is not None
        assert self._batch_returns is not None
        self._batch_advantages[:, :rollout_len].copy_(advantages)
        self._batch_returns[:, :rollout_len].copy_(advantages + values)
        self._invalidate_views()

    def clear(self) -> None:
        if self._batched_actor_step_counts is not None:
            self._batched_actor_step_counts.zero_()
        self._has_aux_for_rollout = False
        self._cache_valid = False
        self._all_obs = None
        self._all_actions = None
        self._all_log_probs = None
        self._all_values = None
        self._all_advantages = None
        self._all_returns = None
        self._all_advantages_normalized = None
        self._all_dones = None
        self._all_aux = None
        self._sequence_view_seq_len = None
        self._sequence_views = {}

    def _ensure_batch_cache(self) -> None:
        """Build stacked actor tensors once per PPO update and reuse them across epochs."""
        if self._cache_valid:
            return

        assert self._batch_obs is not None
        assert self._batch_actions is not None
        assert self._batch_log_probs is not None
        assert self._batch_values is not None
        assert self._batch_dones is not None
        assert self._batch_advantages is not None
        assert self._batch_returns is not None
        rollout_len = self._required_batched_rollout_len()
        if rollout_len == 0:
            self._all_obs = None
            self._all_actions = None
            self._all_log_probs = None
            self._all_values = None
            self._all_advantages = None
            self._all_returns = None
            self._all_advantages_normalized = None
            self._all_dones = None
            self._all_aux = None
            self._cache_valid = True
            return
        self._all_obs = self._batch_obs[:, :rollout_len]
        self._all_actions = self._batch_actions[:, :rollout_len]
        self._all_log_probs = self._batch_log_probs[:, :rollout_len]
        self._all_values = self._batch_values[:, :rollout_len]
        self._all_advantages = self._batch_advantages[:, :rollout_len]
        self._all_returns = self._batch_returns[:, :rollout_len]
        self._all_advantages_normalized = _normalize_advantages_once(self._all_advantages)
        self._all_dones = self._batch_dones[:, :rollout_len]
        self._all_aux = (
            self._batch_aux[:, :rollout_len]
            if self._has_aux_for_rollout and self._batch_aux is not None
            else None
        )
        self._cache_valid = True

    def _get_sequence_views(self, seq_len: int) -> dict[str, Tensor]:
        """Cache reshaped sequence views once per PPO update for the active BPTT length."""
        if self._sequence_view_seq_len == seq_len and self._sequence_views:
            return self._sequence_views

        assert self._all_obs is not None
        assert self._all_actions is not None
        assert self._all_log_probs is not None
        assert self._all_values is not None
        assert self._all_advantages_normalized is not None
        assert self._all_returns is not None
        assert self._all_dones is not None

        n_actors, rollout_len = self._all_obs.shape[:2]
        n_seqs_per_actor = rollout_len // seq_len
        usable_steps = n_seqs_per_actor * seq_len
        total_seqs = n_actors * n_seqs_per_actor

        if n_seqs_per_actor == 0:
            self._sequence_view_seq_len = seq_len
            self._sequence_views = {}
            return self._sequence_views

        sequence_views: dict[str, Tensor] = {
            "obs": self._all_obs[:, :usable_steps].reshape(
                total_seqs, seq_len, *self._all_obs.shape[2:]
            ),
            "acts": self._all_actions[:, :usable_steps].reshape(total_seqs, seq_len, -1),
            "lps": self._all_log_probs[:, :usable_steps].reshape(total_seqs, seq_len),
            "vals": self._all_values[:, :usable_steps].reshape(total_seqs, seq_len),
            "advs": self._all_advantages_normalized[:, :usable_steps].reshape(total_seqs, seq_len),
            "rets": self._all_returns[:, :usable_steps].reshape(total_seqs, seq_len),
        }

        if self._all_aux is not None:
            sequence_views["aux"] = self._all_aux[:, :usable_steps].reshape(
                total_seqs, seq_len, -1
            )

        self._sequence_view_seq_len = seq_len
        self._sequence_views = sequence_views
        return sequence_views

    def sample_minibatches(
        self, batch_size: int = 64, seq_len: int = 32
    ) -> Generator[TrajectoryBuffer.MiniBatch, None, None]:
        """Sample minibatches across all actors while preserving sequences."""
        self._ensure_batch_cache()
        assert self._all_obs is not None
        assert self._all_actions is not None
        assert self._all_log_probs is not None
        assert self._all_values is not None
        assert self._all_advantages_normalized is not None
        assert self._all_returns is not None
        assert self._all_dones is not None

        all_obs = self._all_obs
        all_acts = self._all_actions
        all_lps = self._all_log_probs
        all_vals = self._all_values
        all_advs = self._all_advantages_normalized
        all_rets = self._all_returns
        all_aux = self._all_aux

        n_actors, rollout_len = all_obs.shape[:2]
        index_device = all_obs.device

        if seq_len > 0:
            # BPTT sampling: (N, L) -> (N * L/seq_len, seq_len) sequences
            n_seqs_per_actor = rollout_len // seq_len
            if n_seqs_per_actor == 0:
                return
            seq_views = self._get_sequence_views(seq_len)
            obs_seqs = seq_views["obs"]
            acts_seqs = seq_views["acts"]
            lps_seqs = seq_views["lps"]
            vals_seqs = seq_views["vals"]
            advs_seqs = seq_views["advs"]
            rets_seqs = seq_views["rets"]
            aux_seqs = seq_views.get("aux")
            total_seqs = obs_seqs.shape[0]

            # Shuffle sequence tensors once per epoch, then slice contiguous
            # minibatches to avoid repeated per-tensor index_select kernels.
            perm = torch.randperm(total_seqs, device=index_device)
            obs_seqs = obs_seqs.index_select(0, perm)
            acts_seqs = acts_seqs.index_select(0, perm)
            lps_seqs = lps_seqs.index_select(0, perm)
            vals_seqs = vals_seqs.index_select(0, perm)
            advs_seqs = advs_seqs.index_select(0, perm)
            rets_seqs = rets_seqs.index_select(0, perm)
            if aux_seqs is not None:
                aux_seqs = aux_seqs.index_select(0, perm)

            seqs_per_minibatch = max(1, batch_size // seq_len)
            for i in range(0, total_seqs, seqs_per_minibatch):
                batch_slice = slice(i, i + seqs_per_minibatch)
                mb_obs_seqs = obs_seqs[batch_slice]
                mb_acts_seqs = acts_seqs[batch_slice]
                mb_lps_seqs = lps_seqs[batch_slice]
                mb_vals_seqs = vals_seqs[batch_slice]
                mb_advs_seqs = advs_seqs[batch_slice]
                mb_rets_seqs = rets_seqs[batch_slice]
                mb_aux_seqs = aux_seqs[batch_slice] if aux_seqs is not None else None
                yield TrajectoryBuffer.MiniBatch(
                    observations=mb_obs_seqs.flatten(0, 1),
                    actions=mb_acts_seqs.flatten(0, 1),
                    old_log_probs=mb_lps_seqs.flatten(),
                    old_values=mb_vals_seqs.flatten(),
                    advantages=mb_advs_seqs.flatten(),
                    returns=mb_rets_seqs.flatten(),
                    aux_tensors=(
                        mb_aux_seqs.flatten(0, 1)
                        if mb_aux_seqs is not None and all_aux is not None
                        else None
                    ),
                    sequence_observations=mb_obs_seqs,
                    sequence_actions=mb_acts_seqs,
                    sequence_aux_tensors=mb_aux_seqs,
                )
        else:
            # Flat random shuffle
            flat_indices = torch.randperm(n_actors * rollout_len, device=index_device)
            obs_flat = all_obs.view(-1, *all_obs.shape[2:])
            acts_flat = all_acts.view(-1, all_acts.shape[-1])
            lps_flat = all_lps.flatten()
            vals_flat = all_vals.flatten()
            advs_flat = all_advs.flatten()
            rets_flat = all_rets.flatten()
            aux_flat = all_aux.view(-1, all_aux.shape[-1]) if all_aux is not None else None

            for i in range(0, len(flat_indices), batch_size):
                idx = flat_indices[i : i + batch_size]
                yield TrajectoryBuffer.MiniBatch(
                    observations=obs_flat[idx],
                    actions=acts_flat[idx],
                    old_log_probs=lps_flat[idx],
                    old_values=vals_flat[idx],
                    advantages=advs_flat[idx],
                    returns=rets_flat[idx],
                    aux_tensors=aux_flat[idx] if aux_flat is not None else None,
                )

    def __len__(self) -> int:
        if self._batched_actor_step_counts is None:
            return 0
        return int(self._batched_actor_step_counts.sum().item())


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
        self._normalized_advantages: Tensor = torch.zeros(0)
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
            self._t_obs = torch.empty(
                (self._capacity, *observation.shape),
                dtype=observation.dtype,
                device=observation.device,
            )
            self._t_actions = torch.empty(
                (self._capacity, *action.shape), dtype=action.dtype, device=action.device
            )
            self._t_log_probs = torch.empty(
                (self._capacity,), dtype=log_prob.dtype, device=log_prob.device
            )
            self._t_values = torch.empty((self._capacity,), dtype=value.dtype, device=value.device)
            self._t_rewards = torch.empty(
                (self._capacity,), dtype=reward.dtype, device=reward.device
            )
            self._t_dones = torch.empty((self._capacity,), dtype=done.dtype, device=done.device)
            self._t_truncated = torch.empty(
                (self._capacity,), dtype=truncated.dtype, device=truncated.device
            )
            if aux_tensor is not None:
                self._t_aux = torch.empty(
                    (self._capacity, *aux_tensor.shape),
                    dtype=aux_tensor.dtype,
                    device=aux_tensor.device,
                )

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
                self._t_aux = torch.empty(
                    (self._capacity, *aux_tensor.shape),
                    dtype=aux_tensor.dtype,
                    device=aux_tensor.device,
                )
            self._t_aux[idx].copy_(aux_tensor)
        self._hidden_states.append(hidden_state)
        self._size += 1
        self._finalized = False
        self._t_cached = False
        self._normalized_advantages = torch.zeros(0)

    # ── Mutation (invalidates cache) ─────────────────────────────

    def append(self, transition: PPOTransition) -> None:
        """Append one transition. Resets finalized state."""
        if self._uses_tensor_storage():
            self.append_fields(
                observation=transition.observation,
                action=transition.action,
                log_prob=torch.tensor(
                    transition.log_prob, dtype=torch.float32, device=transition.action.device
                ),
                value=torch.tensor(
                    transition.value, dtype=torch.float32, device=transition.action.device
                ),
                reward=torch.tensor(
                    transition.reward, dtype=torch.float32, device=transition.action.device
                ),
                done=torch.tensor(
                    transition.done, dtype=torch.bool, device=transition.action.device
                ),
                truncated=torch.tensor(
                    transition.truncated, dtype=torch.bool, device=transition.action.device
                ),
                hidden_state=transition.hidden_state,
                aux_tensor=transition.aux_tensor,
            )
            return
        self._transitions.append(transition)
        self._hidden_states.append(transition.hidden_state)
        self._size += 1
        self._finalized = False
        self._t_cached = False
        self._normalized_advantages = torch.zeros(0)

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
            self._normalized_advantages = torch.zeros(0)
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
        self._normalized_advantages = torch.zeros(0)
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
            [tr.log_prob for tr in self._transitions],
            dtype=torch.float32,
        )
        self._t_values = torch.tensor(
            [tr.value for tr in self._transitions],
            dtype=torch.float32,
        )
        self._t_dones = torch.tensor(
            [tr.done for tr in self._transitions],
            dtype=torch.bool,
        )
        self._t_aux = (
            torch.stack([cast("Tensor", tr.aux_tensor) for tr in self._transitions])
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
        bootstrap = torch.as_tensor(last_value, dtype=torch.float32, device=value_device).reshape(
            ()
        )
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
        self._normalized_advantages = _normalize_advantages_once(advantages)
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
        sequence_observations: Tensor | None = None  # (n_seqs, seq_len, 3, Az, El)
        sequence_actions: Tensor | None = None  # (n_seqs, seq_len, 4)
        dones: Tensor | None = None  # (n_seqs, seq_len) for BPTT
        aux_tensors: Tensor | None = None  # (B, 3) or (n_seqs, seq_len, 3)
        sequence_aux_tensors: Tensor | None = None  # (n_seqs, seq_len, 3) for BPTT

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
        advs = self._normalized_advantages
        rets = self._returns

        if seq_len > 0 and n >= seq_len:
            # Sequential chunks for BPTT
            starts = list(range(0, n - seq_len + 1, seq_len))
            perm = torch.randperm(len(starts), device=obs.device)
            start_tensor = torch.arange(
                0, n - seq_len + 1, seq_len, device=obs.device, dtype=torch.long
            )
            offsets = torch.arange(seq_len, device=obs.device, dtype=torch.long)
            for i in range(0, len(perm), max(1, batch_size // seq_len)):
                seq_indices = perm[i : i + max(1, batch_size // seq_len)]
                start_indices = start_tensor.index_select(0, seq_indices)
                chunk_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
                seq_obs = obs[chunk_indices]
                seq_acts = acts[chunk_indices]
                seq_old_lp = old_lp[chunk_indices]
                seq_old_v = old_v[chunk_indices]
                seq_advs = advs[chunk_indices]
                seq_rets = rets[chunk_indices]
                seq_aux = aux_flat[chunk_indices] if aux_flat is not None else None

                yield TrajectoryBuffer.MiniBatch(
                    observations=seq_obs.flatten(0, 1),
                    actions=seq_acts.flatten(0, 1),
                    old_log_probs=seq_old_lp.flatten(),
                    old_values=seq_old_v.flatten(),
                    advantages=seq_advs.flatten(),
                    returns=seq_rets.flatten(),
                    dones=dones_flat[chunk_indices],  # (n_seqs, seq_len)
                    aux_tensors=seq_aux.flatten(0, 1) if seq_aux is not None else None,
                    sequence_observations=seq_obs,
                    sequence_actions=seq_acts,
                    sequence_aux_tensors=seq_aux,
                )
        else:
            # Random shuffle
            indices = torch.randperm(n, device=obs.device)
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
