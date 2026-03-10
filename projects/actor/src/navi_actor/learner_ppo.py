"""PPO learner with clipped surrogate objective and GAE for Ghost-Matrix actor."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from navi_actor.cognitive_policy import CognitiveMambaPolicy
    from navi_actor.rnd import RNDModule
    from navi_actor.rollout_buffer import MultiTrajectoryBuffer, TrajectoryBuffer

__all__: list[str] = ["PpoLearner", "PpoMetrics"]

_LOGGER = logging.getLogger(__name__)


def _prepare_minibatch_tensor(tensor: Tensor, device: torch.device) -> Tensor:
    """Reuse CUDA-resident minibatch tensors instead of re-copying them."""
    if tensor.device == device:
        return tensor
    return tensor.to(device=device, non_blocking=True)


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
    learning_rate: float
    rnd_learning_rate: float


class PpoLearner:
    """PPO learner with clipped surrogate, value clipping, and entropy bonus."""

    def __init__(
        self,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.005,
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
        self._policy_param_cache: tuple[Tensor, ...] | None = None

    def set_learning_rate(
        self, lr: float, rnd_lr: float | None = None
    ) -> None:
        """Update the learning rate for all optimizers (annealing)."""
        self._learning_rate = lr
        if rnd_lr is not None:
            self._rnd_learning_rate = rnd_lr

        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = self._learning_rate
        if self._rnd_optimizer is not None:
            for param_group in self._rnd_optimizer.param_groups:
                param_group["lr"] = self._rnd_learning_rate

    def _get_optimizer(self, policy: CognitiveMambaPolicy) -> torch.optim.Adam:
        """Lazily create or return the unified policy optimizer.

        Covers the entire model: encoder + temporal core + actor head + critic head.
        """
        if self._optimizer is None:
            use_cuda = policy.device.type == "cuda"
            self._optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=self._learning_rate,
                foreach=use_cuda,
            )
        return self._optimizer

    def _get_rnd_optimizer(self, rnd: RNDModule) -> torch.optim.Adam:
        """Lazily create or return the RND predictor optimizer."""
        if self._rnd_optimizer is None:
            use_foreach = next(rnd.predictor.parameters()).device.type == "cuda"
            self._rnd_optimizer = torch.optim.Adam(
                rnd.predictor.parameters(),
                lr=self._rnd_learning_rate,
                foreach=use_foreach,
            )
        return self._rnd_optimizer

    def _policy_params(self, policy: CognitiveMambaPolicy) -> tuple[Tensor, ...]:
        """Cache the policy parameter tuple used for gradient clipping."""
        if self._policy_param_cache is None:
            self._policy_param_cache = tuple(policy.parameters())
        return self._policy_param_cache

    def train_ppo_epoch(
        self,
        policy: CognitiveMambaPolicy,
        buffer: TrajectoryBuffer | MultiTrajectoryBuffer,
        *,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        seq_len: int = 32,
        rnd: RNDModule | None = None,
        progress_callback: Callable[[], None] | None = None,
    ) -> PpoMetrics:
        """Run multiple PPO mini-batch updates on a filled trajectory buffer."""
        optimizer = self._get_optimizer(policy)
        rnd_optimizer = self._get_rnd_optimizer(rnd) if rnd is not None else None
        device = policy.device
        policy.train()

        _LOGGER.debug("Starting PPO epoch with %d samples (epochs=%d, batch=%d)",
                      len(buffer), ppo_epochs, minibatch_size)

        params = self._policy_params(policy)

        metric_device = device
        running_policy_loss = torch.zeros((), device=metric_device)
        running_value_loss = torch.zeros((), device=metric_device)
        running_entropy = torch.zeros((), device=metric_device)
        running_kl = torch.zeros((), device=metric_device)
        running_clip = torch.zeros((), device=metric_device)
        running_total = torch.zeros((), device=metric_device)
        running_rnd_loss = torch.zeros((), device=metric_device)
        zero_metric = torch.zeros((), device=metric_device)
        n_updates = 0

        for _epoch in range(ppo_epochs):
            for mb in buffer.sample_minibatches(minibatch_size, seq_len):
                old_lp = _prepare_minibatch_tensor(mb.old_log_probs, device)
                adv = _prepare_minibatch_tensor(mb.advantages, device)
                ret = _prepare_minibatch_tensor(mb.returns, device)
                old_vals = _prepare_minibatch_tensor(mb.old_values, device)
                obs_seq_tensor = (
                    _prepare_minibatch_tensor(mb.sequence_observations, device)
                    if mb.sequence_observations is not None
                    else None
                )
                acts_seq_tensor = (
                    _prepare_minibatch_tensor(mb.sequence_actions, device)
                    if mb.sequence_actions is not None
                    else None
                )
                aux_seq_tensor = (
                    _prepare_minibatch_tensor(mb.sequence_aux_tensors, device)
                    if mb.sequence_aux_tensors is not None
                    else None
                )
                h0: Tensor | None = None
                dones_mask: Tensor | None = None
                if mb.hidden_batch is not None:
                    h0 = _prepare_minibatch_tensor(mb.hidden_batch, device)
                elif mb.hidden_states:
                    first_h = next((h for h in mb.hidden_states if h is not None), None)
                    if first_h is not None:
                        stacked = [h if h is not None else torch.zeros_like(first_h) for h in mb.hidden_states]
                        h0 = _prepare_minibatch_tensor(torch.cat(stacked, dim=1), device)
                if mb.dones is not None:
                    dones_mask = _prepare_minibatch_tensor(mb.dones, device)

                # Forward pass
                if obs_seq_tensor is not None and acts_seq_tensor is not None:
                    new_lp, new_vals, ent, _, z_mb = policy.evaluate_sequence(
                        obs_seq_tensor, acts_seq_tensor, hidden=h0, dones=dones_mask, aux_seq=aux_seq_tensor,
                    )
                else:
                    obs = _prepare_minibatch_tensor(mb.observations, device)
                    acts = _prepare_minibatch_tensor(mb.actions, device)
                    aux_tensor = (
                        _prepare_minibatch_tensor(mb.aux_tensors, device)
                        if mb.aux_tensors is not None
                        else None
                    )

                    if seq_len > 0 and obs.shape[0] >= seq_len:
                        total = obs.shape[0]
                        n_seqs = total // seq_len
                        usable = n_seqs * seq_len
                        obs_seq = obs[:usable].reshape(n_seqs, seq_len, *obs.shape[1:])
                        acts_seq = acts[:usable].reshape(n_seqs, seq_len, -1)
                        aux_seq = aux_tensor[:usable].reshape(n_seqs, seq_len, -1) if aux_tensor is not None else None

                        new_lp, new_vals, ent, _, z_mb = policy.evaluate_sequence(
                            obs_seq, acts_seq, hidden=h0, dones=dones_mask, aux_seq=aux_seq,
                        )
                        if usable < total:
                            rem_lp, rem_v, rem_e, _, rem_z = policy.evaluate(
                                obs[usable:], acts[usable:], aux_tensor=aux_tensor[usable:] if aux_tensor is not None else None
                            )
                            new_lp = torch.cat([new_lp, rem_lp])
                            new_vals = torch.cat([new_vals, rem_v])
                            ent = (ent * usable + rem_e * (total - usable)) / total
                            z_mb = torch.cat([z_mb, rem_z])
                    else:
                        new_lp, new_vals, ent, _, z_mb = policy.evaluate(obs, acts, aux_tensor=aux_tensor)

                # Squeeze outputs from policy to ensure they are 1D (B,)
                new_vals = new_vals.squeeze(-1)

                # PPO losses
                log_ratio = new_lp - old_lp
                ratio = log_ratio.exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_vals + torch.clamp(new_vals - old_vals, -self._clip_ratio, self._clip_ratio)
                vf_loss = 0.5 * torch.max((new_vals - ret)**2, (value_pred_clipped - ret)**2).mean()

                total_loss = policy_loss + self._value_coeff * vf_loss - self._entropy_coeff * ent

                # Optimization step
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self._max_grad_norm, foreach=(device.type == "cuda"))
                optimizer.step()

                # RND distillation
                rnd_loss_metric = zero_metric
                if rnd is not None and rnd_optimizer is not None:
                    z_rnd = z_mb.detach()
                    rnd_loss = rnd.distillation_loss(z_rnd)
                    rnd_optimizer.zero_grad(set_to_none=True)
                    rnd_loss.backward()  # type: ignore[no-untyped-call]
                    rnd_optimizer.step()
                    rnd_loss_metric = rnd_loss.detach()

                with torch.no_grad():
                    approx_kl = (ratio - 1.0 - log_ratio).mean()
                    clip_frac = ((ratio - 1.0).abs() > self._clip_ratio).float().mean()

                running_policy_loss += policy_loss.detach()
                running_value_loss += vf_loss.detach()
                running_entropy += ent.detach()
                running_kl += approx_kl.detach()
                running_clip += clip_frac.detach()
                running_total += total_loss.detach()
                running_rnd_loss += rnd_loss_metric
                n_updates += 1

                if progress_callback is not None:
                    progress_callback()

        if n_updates == 0:
            return PpoMetrics(
                policy_loss=0.0, value_loss=0.0, entropy=0.0, approx_kl=0.0,
                clip_fraction=0.0, total_loss=0.0, rnd_loss=0.0,
                learning_rate=self._learning_rate, rnd_learning_rate=self._rnd_learning_rate,
            )

        _LOGGER.debug("PPO epoch completed: loss=%0.4f, kl=%0.4f, entropy=%0.4f",
                      (running_total / n_updates).item(), (running_kl / n_updates).item(), (running_entropy / n_updates).item())

        return PpoMetrics(
            policy_loss=(running_policy_loss / n_updates).item(),
            value_loss=(running_value_loss / n_updates).item(),
            entropy=(running_entropy / n_updates).item(),
            approx_kl=(running_kl / n_updates).item(),
            clip_fraction=(running_clip / n_updates).item(),
            total_loss=(running_total / n_updates).item(),
            rnd_loss=(running_rnd_loss / n_updates).item(),
            learning_rate=self._learning_rate,
            rnd_learning_rate=self._rnd_learning_rate,
        )
