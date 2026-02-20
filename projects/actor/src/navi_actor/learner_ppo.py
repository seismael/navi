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

__all__: list[str] = ["PpoLearner", "PpoMetrics"]


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
        rnd_learning_rate: float = 1e-3,
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
                new_lp, new_vals, ent, _ = policy.evaluate(obs, acts)

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
                    # Extract spatial embeddings for RND
                    with torch.no_grad():
                        z_rnd = policy.encode(obs)
                    rnd_loss = rnd.distillation_loss(z_rnd)
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()
                    rnd_loss_val = rnd_loss.item()

                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = (old_lp - new_lp).mean().item()
                    clip_frac = (
                        ((ratio - 1.0).abs() > self._clip_ratio).float().mean().item()
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
