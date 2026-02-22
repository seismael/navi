"""PPO learner with clipped surrogate objective and GAE for Ghost-Matrix actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from navi_actor.cognitive_policy import CognitiveMambaPolicy
    from navi_actor.rnd import RNDModule
    from navi_actor.rollout_buffer import RolloutBuffer, TrajectoryBuffer

__all__: list[str] = ["PpoLearner", "PpoMetrics", "compute_vtrace"]


def compute_vtrace(
    log_probs_current: Tensor,
    log_probs_behaviour: Tensor,
    rewards: Tensor,
    values: Tensor,
    bootstrap_value: float,
    dones: Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Compute V-trace targets and advantages for off-policy correction.

    Implements the IMPALA V-trace algorithm (Espeholt et al., 2018) which
    corrects for the policy lag between rollout workers and the central
    learner.

    Args:
        log_probs_current: (T,) log-probabilities under current learner policy.
        log_probs_behaviour: (T,) log-probabilities under worker's behaviour policy.
        rewards: (T,) rewards at each step.
        values: (T,) value estimates from the *current* policy for each state.
        bootstrap_value: V(s_{T+1}) value estimate for the state after the
            last transition (0.0 if episode ended).
        dones: (T,) done flags (1.0 = terminal, 0.0 = continuing).
        gamma: discount factor.
        rho_bar: truncation level for importance sampling ratio rho_bar.
        c_bar: truncation level for trace-cutting coefficient c_bar.

    Returns:
        vtrace_targets: (T,) corrected value targets.
        advantages: (T,) policy gradient advantages (rho-weighted TD errors).

    """
    t_len = rewards.shape[0]
    device = rewards.device

    # Importance sampling ratios
    log_rhos = log_probs_current - log_probs_behaviour
    rhos = torch.exp(log_rhos).clamp(max=rho_bar)  # rho_t = min(rho_bar, pi/mu)
    cs = torch.exp(log_rhos).clamp(max=c_bar)  # c_t = min(c_bar, pi/mu)

    masks = 1.0 - dones  # 0 on terminal, 1 on continuing

    # V-trace targets (backward scan)
    vtrace_targets = torch.zeros(t_len, device=device)
    last_v = bootstrap_value

    for t in reversed(range(t_len)):
        delta = rhos[t] * (rewards[t] + gamma * masks[t] * last_v - values[t])
        # v_s = V(s) + sum gamma^t c_{s:t-1} delta_t V  (incremental form)
        last_v = values[t] + delta + gamma * masks[t] * cs[t] * (last_v - values[t] - delta / rhos[t].clamp(min=1e-8) * rhos[t])
        vtrace_targets[t] = last_v

    # Policy gradient advantages: rho_t * (r_t + gamma * v_{t+1} - V(s_t))
    next_values = torch.cat([values[1:], torch.tensor([bootstrap_value], device=device)])
    advantages = rhos * (rewards + gamma * masks * next_values - values)

    return vtrace_targets, advantages


@dataclass(frozen=True)
class PpoMetrics:
    """Metrics from a single PPO training epoch."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    total_loss: float
    rnd_loss: float


class PpoLearner:
    """PPO learner with clipped surrogate, value clipping, and entropy bonus.

    Supports both:
    - Legacy mode: ``train_epoch(RolloutBuffer)`` → simple reward metrics
    - Full PPO mode: ``train_ppo_epoch(policy, buffer)`` → PpoMetrics
    """

    def __init__(
        self,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        learning_rate: float = 3e-4,
        rnd_learning_rate: float = 3e-5,
    ) -> None:
        self._gamma = gamma
        self._clip_ratio = clip_ratio
        self._entropy_coeff = entropy_coeff
        self._value_coeff = value_coeff
        self._max_grad_norm = max_grad_norm
        self._learning_rate = learning_rate
        self._rnd_learning_rate = rnd_learning_rate
        self._optimizer: torch.optim.Adam | None = None
        self._rnd_optimizer: torch.optim.Adam | None = None

    def _get_optimizer(self, policy: CognitiveMambaPolicy) -> torch.optim.Adam:
        """Lazily create or return the Adam optimizer."""
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                policy.parameters(), lr=self._learning_rate
            )
        return self._optimizer

    def _get_rnd_optimizer(self, rnd: RNDModule) -> torch.optim.Adam:
        """Lazily create or return the RND predictor optimizer."""
        if self._rnd_optimizer is None:
            self._rnd_optimizer = torch.optim.Adam(
                rnd.predictor.parameters(), lr=self._rnd_learning_rate
            )
        return self._rnd_optimizer

    def discounted_return(self, rewards: np.ndarray) -> float:
        """Compute scalar discounted return for one reward vector."""
        running = 0.0
        for reward in rewards[::-1]:
            running = float(reward) + self._gamma * running
        return running

    def train_epoch(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Compute basic epoch metrics from buffered transitions (legacy)."""
        rewards = buffer.rewards()
        if rewards.size == 0:
            return {"episodes": 0.0, "reward_mean": 0.0, "discounted_return": 0.0}

        return {
            "episodes": float(rewards.size),
            "reward_mean": float(rewards.mean()),
            "discounted_return": self.discounted_return(rewards),
        }

    def train_ppo_epoch(
        self,
        policy: CognitiveMambaPolicy,
        buffer: TrajectoryBuffer,
        *,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        seq_len: int = 32,
        rnd: RNDModule | None = None,
    ) -> PpoMetrics:
        """Run multiple PPO mini-batch updates on a filled trajectory buffer.

        Args:
            policy: the neural policy to train.
            buffer: trajectory buffer with computed advantages/returns.
            ppo_epochs: number of passes over the data.
            minibatch_size: transitions per minibatch.
            seq_len: BPTT window length (0 for random shuffle).
            rnd: optional RND module for distillation loss.

        Returns:
            Aggregated PpoMetrics from the epoch.

        """
        optimizer = self._get_optimizer(policy)
        rnd_optimizer = self._get_rnd_optimizer(rnd) if rnd is not None else None
        device = policy.device
        policy.train()

        running_policy_loss = 0.0
        running_value_loss = 0.0
        running_entropy = 0.0
        running_kl = 0.0
        running_clip = 0.0
        running_total = 0.0
        running_rnd_loss = 0.0
        n_updates = 0

        for _epoch in range(ppo_epochs):
            for mb in buffer.sample_minibatches(minibatch_size, seq_len):
                obs = mb.observations.to(device)
                acts = mb.actions.to(device)
                old_lp = mb.old_log_probs.to(device)
                adv = mb.advantages.to(device)
                ret = mb.returns.to(device)
                old_vals = mb.old_values.to(device)

                # Forward pass — evaluate current policy
                if seq_len > 0 and obs.shape[0] >= seq_len:
                    # Reshape into (num_seqs, seq_len, ...) for BPTT
                    total = obs.shape[0]
                    n_seqs = total // seq_len
                    usable = n_seqs * seq_len
                    obs_seq = obs[:usable].reshape(
                        n_seqs, seq_len, *obs.shape[1:]
                    )
                    acts_seq = acts[:usable].reshape(n_seqs, seq_len, -1)
                    new_lp, new_vals, ent, _, z_mb = policy.evaluate_sequence(
                        obs_seq, acts_seq
                    )
                    # Handle remainder (if any) with single-step evaluate
                    if usable < total:
                        rem_lp, rem_v, rem_e, _, rem_z = policy.evaluate(
                            obs[usable:], acts[usable:]
                        )
                        new_lp = torch.cat([new_lp, rem_lp])
                        new_vals = torch.cat([new_vals, rem_v])
                        ent = (ent * usable + rem_e * (total - usable)) / total
                        z_mb = torch.cat([z_mb, rem_z])
                else:
                    new_lp, new_vals, ent, _, z_mb = policy.evaluate(obs, acts)

                # Policy (actor) loss — clipped surrogate
                ratio = (new_lp - old_lp).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio
                ) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value (critic) loss — clipped value
                value_pred_clipped = old_vals + torch.clamp(
                    new_vals - old_vals, -self._clip_ratio, self._clip_ratio
                )
                vf_loss1 = (new_vals - ret) ** 2
                vf_loss2 = (value_pred_clipped - ret) ** 2
                value_loss: Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                # Entropy bonus
                entropy_loss = -ent

                # Total loss
                total_loss: Tensor = (
                    policy_loss
                    + self._value_coeff * value_loss
                    + self._entropy_coeff * entropy_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), self._max_grad_norm
                )
                optimizer.step()

                # RND distillation loss (separate optimizer)
                rnd_loss_val = 0.0
                if rnd is not None and rnd_optimizer is not None:
                    # Use z_t from evaluate/evaluate_sequence (no extra CNN pass)
                    with torch.no_grad():
                        z_rnd = z_mb.detach()
                    rnd_loss = rnd.distillation_loss(z_rnd)
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()
                    rnd_loss_val = rnd_loss.item()

                # Approximate KL divergence (always non-negative)
                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - (ratio.log())).mean().item()
                    )
                    clip_frac = (
                        ((ratio - 1.0).abs() > self._clip_ratio)
                        .float()
                        .mean()
                        .item()
                    )

                running_policy_loss += policy_loss.item()
                running_value_loss += value_loss.item()
                running_entropy += ent.item()
                running_kl += approx_kl
                running_clip += clip_frac
                running_total += total_loss.item()
                running_rnd_loss += rnd_loss_val
                n_updates += 1

        if n_updates == 0:
            return PpoMetrics(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                total_loss=0.0,
                rnd_loss=0.0,
            )

        return PpoMetrics(
            policy_loss=running_policy_loss / n_updates,
            value_loss=running_value_loss / n_updates,
            entropy=running_entropy / n_updates,
            approx_kl=running_kl / n_updates,
            clip_fraction=running_clip / n_updates,
            total_loss=running_total / n_updates,
            rnd_loss=running_rnd_loss / n_updates,
        )

    def train_offpolicy_batch(
        self,
        policy: CognitiveMambaPolicy,
        *,
        observations: Tensor,
        actions: Tensor,
        behaviour_log_probs: Tensor,
        rewards: Tensor,
        dones: Tensor,
        values: Tensor,
        bootstrap_value: float,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        rnd: RNDModule | None = None,
        gamma: float = 0.99,
    ) -> PpoMetrics:
        """Train on an off-policy trajectory batch using V-trace corrections.

        Used by the central learner in parallel training to process
        trajectory batches collected by remote rollout workers under a
        stale copy of the policy.

        Args:
            policy: the central learner's current policy.
            observations: (T, 2, Az, El) observation tensors.
            actions: (T, 4) action tensors.
            behaviour_log_probs: (T,) log-probs under the worker's policy.
            rewards: (T,) shaped rewards.
            dones: (T,) done flags.
            values: (T,) value estimates from the worker's policy.
            bootstrap_value: V(s_{T+1}) for last state.
            ppo_epochs: number of PPO passes over the batch.
            minibatch_size: transitions per minibatch.
            rnd: optional RND module for distillation loss.
            gamma: discount factor.

        Returns:
            Aggregated PpoMetrics from the training pass.

        """
        optimizer = self._get_optimizer(policy)
        rnd_optimizer = self._get_rnd_optimizer(rnd) if rnd is not None else None
        device = policy.device
        policy.train()

        n_transitions = observations.shape[0]

        # Compute current policy's log-probs for V-trace
        with torch.no_grad():
            obs_dev = observations.to(device)
            acts_dev = actions.to(device)
            current_lp, current_vals, _, _, _ = policy.evaluate(obs_dev, acts_dev)
            current_lp = current_lp.cpu()
            current_vals_flat = current_vals.cpu()

        # V-trace targets and advantages
        vtrace_targets, vtrace_advantages = compute_vtrace(
            log_probs_current=current_lp,
            log_probs_behaviour=behaviour_log_probs,
            rewards=rewards,
            values=current_vals_flat,
            bootstrap_value=bootstrap_value,
            dones=dones,
            gamma=gamma,
        )

        # Normalize advantages
        if vtrace_advantages.numel() > 1:
            vtrace_advantages = (
                vtrace_advantages - vtrace_advantages.mean()
            ) / (vtrace_advantages.std() + 1e-8)

        # PPO epochs over the V-trace-corrected batch
        running_policy_loss = 0.0
        running_value_loss = 0.0
        running_entropy = 0.0
        running_kl = 0.0
        running_clip = 0.0
        running_total = 0.0
        running_rnd_loss = 0.0
        n_updates = 0

        for _epoch in range(ppo_epochs):
            perm = torch.randperm(n_transitions)
            for start in range(0, n_transitions, minibatch_size):
                idx = perm[start : start + minibatch_size]
                obs_mb = observations[idx].to(device)
                acts_mb = actions[idx].to(device)
                old_lp_mb = behaviour_log_probs[idx].to(device)
                adv_mb = vtrace_advantages[idx].to(device)
                ret_mb = vtrace_targets[idx].to(device)
                new_lp, new_vals, ent, _, z_mb = policy.evaluate(obs_mb, acts_mb)

                # Clipped surrogate
                ratio = (new_lp - old_lp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(
                    ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio,
                ) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss against V-trace targets
                value_loss: Tensor = 0.5 * (new_vals - ret_mb).pow(2).mean()

                entropy_loss = -ent
                total_loss: Tensor = (
                    policy_loss
                    + self._value_coeff * value_loss
                    + self._entropy_coeff * entropy_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), self._max_grad_norm,
                )
                optimizer.step()

                # RND distillation
                rnd_loss_val = 0.0
                if rnd is not None and rnd_optimizer is not None:
                    with torch.no_grad():
                        z_rnd = z_mb.detach()
                    rnd_loss = rnd.distillation_loss(z_rnd)
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()
                    rnd_loss_val = rnd_loss.item()

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - ratio.log()).mean().item()
                    )
                    clip_frac = (
                        ((ratio - 1.0).abs() > self._clip_ratio)
                        .float()
                        .mean()
                        .item()
                    )

                running_policy_loss += policy_loss.item()
                running_value_loss += value_loss.item()
                running_entropy += ent.item()
                running_kl += approx_kl
                running_clip += clip_frac
                running_total += total_loss.item()
                running_rnd_loss += rnd_loss_val
                n_updates += 1

        if n_updates == 0:
            return PpoMetrics(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                approx_kl=0.0,
                clip_fraction=0.0,
                total_loss=0.0,
                rnd_loss=0.0,
            )

        return PpoMetrics(
            policy_loss=running_policy_loss / n_updates,
            value_loss=running_value_loss / n_updates,
            entropy=running_entropy / n_updates,
            approx_kl=running_kl / n_updates,
            clip_fraction=running_clip / n_updates,
            total_loss=running_total / n_updates,
            rnd_loss=running_rnd_loss / n_updates,
        )
