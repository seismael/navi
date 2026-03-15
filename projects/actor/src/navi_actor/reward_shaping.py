"""Reward shaping: extrinsic penalties + RND intrinsic reward + loop penalty."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

__all__: list[str] = ["RewardShaper", "ShapedReward"]

_LOGGER = logging.getLogger(__name__)


def _shape_batch_impl(
    raw_rewards: Tensor,
    dones: Tensor,
    forward_velocities: Tensor,
    intrinsic_rewards: Tensor,
    loop_similarities: Tensor,
    collision_penalty: float,
    existential_tax: float,
    velocity_weight: float,
    beta: float,
    loop_penalty_coeff: float,
    loop_threshold: float,
) -> Tensor:
    col_penalties = torch.where(
        dones.bool(),
        torch.full_like(raw_rewards, collision_penalty),
        torch.zeros_like(raw_rewards),
    )
    vel_bonus = velocity_weight * forward_velocities.clamp(min=0.0)
    intrinsic = beta * intrinsic_rewards
    loop_pen = loop_penalty_coeff * (loop_similarities - loop_threshold).clamp(min=0.0)
    return raw_rewards + col_penalties + existential_tax + vel_bonus + intrinsic - loop_pen


@dataclass(frozen=True)
class ShapedReward:
    """Decomposed reward components for telemetry and debugging."""

    extrinsic: float
    collision_penalty: float
    existential_tax: float
    velocity_bonus: float
    intrinsic: float
    loop_penalty: float
    total: float


class RewardShaper:
    """Combine extrinsic environment rewards with intrinsic curiosity signals.

    Reward formulation::

        r_total = r_extrinsic
                  + collision_penalty
                  + existential_tax
                  + velocity_bonus
                  + beta(t) * r_intrinsic
                  - lambda_loop * max(0, s_loop - tau)

    Args:
        collision_penalty: penalty on collision termination.
        existential_tax: per-step cost to force efficiency.
        velocity_weight: multiplier for forward-velocity heuristic.
        intrinsic_coeff_init: initial beta(t) for RND intrinsic reward.
        intrinsic_coeff_final: final beta(t) after annealing.
        intrinsic_anneal_steps: steps over which beta decays.
        loop_penalty_coeff: lambda_loop for episodic memory loop penalty.
        loop_threshold: tau - similarity threshold above which loop penalty fires.

    """

    def __init__(
        self,
        *,
        collision_penalty: float = 0.0,
        existential_tax: float = -0.01,
        velocity_weight: float = 0.0,
        intrinsic_coeff_init: float = 1.0,
        intrinsic_coeff_final: float = 0.01,
        intrinsic_anneal_steps: int = 500_000,
        loop_penalty_coeff: float = 0.5,
        loop_threshold: float = 0.85,
        torch_compile: bool = True,
    ) -> None:
        self._collision_penalty = collision_penalty
        self._existential_tax = existential_tax
        self._velocity_weight = velocity_weight
        self._beta_init = intrinsic_coeff_init
        self._beta_final = intrinsic_coeff_final
        self._anneal_steps = intrinsic_anneal_steps
        self._lambda_loop = loop_penalty_coeff
        self._tau = loop_threshold
        self._global_step = 0
        self._torch_compile_requested = bool(torch_compile)
        self._torch_compile_enabled = False
        self._shape_batch_fn = _shape_batch_impl
        if self._torch_compile_requested:
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:
                _LOGGER.warning(
                    "RewardShaper: torch.compile requested but unavailable; using eager batch shaping path",
                )
            elif not torch.cuda.is_available():
                _LOGGER.info(
                    "RewardShaper: torch.compile requested but CUDA is unavailable; using eager batch shaping path",
                )
            else:
                capability = tuple(int(value) for value in torch.cuda.get_device_capability())
                if capability < (7, 0):
                    _LOGGER.warning(
                        "RewardShaper: torch.compile requested but disabled on %s (sm_%d%d); using eager batch shaping path",
                        torch.cuda.get_device_name(),
                        capability[0],
                        capability[1],
                    )
                else:
                    self._shape_batch_fn = compile_fn(
                        _shape_batch_impl,
                        fullgraph=False,
                        dynamic=False,
                        mode="reduce-overhead",
                    )
                    self._torch_compile_enabled = True
                    _LOGGER.info("RewardShaper: torch.compile enabled for tensor-only batch shaping path")

    @property
    def torch_compile_requested(self) -> bool:
        return self._torch_compile_requested

    @property
    def torch_compile_enabled(self) -> bool:
        return self._torch_compile_enabled

    @property
    def beta(self) -> float:
        """Current intrinsic reward annealing coefficient."""
        if self._anneal_steps <= 0:
            return self._beta_final
        frac = min(1.0, self._global_step / self._anneal_steps)
        return self._beta_init + frac * (self._beta_final - self._beta_init)

    def step(self) -> None:
        """Advance the global step counter for annealing."""
        self._global_step += 1

    def step_batch(self, batch_size: int) -> None:
        """Advance the global step counter by a rollout batch size."""
        self._global_step += max(0, int(batch_size))

    def shape(
        self,
        *,
        raw_reward: float,
        done: bool,
        forward_velocity: float = 0.0,
        angular_velocity: float = 0.0,
        intrinsic_reward: float = 0.0,
        loop_similarity: float = 0.0,
    ) -> ShapedReward:
        """Compute the total shaped reward for a single transition.

        Args:
            raw_reward: extrinsic reward from the environment.
            done: whether the episode terminated (collision).
            forward_velocity: agent's forward velocity magnitude.
            angular_velocity: agent's angular velocity magnitude.
            intrinsic_reward: RND normalized intrinsic reward.
            loop_similarity: episodic memory cosine similarity score.

        Returns:
            ShapedReward with all components and the final total.

        """
        # Extrinsic
        extrinsic = raw_reward

        # Collision penalty — applied only on terminal steps
        col_penalty = self._collision_penalty if done else 0.0

        # Existential tax — per-step cost
        tax = self._existential_tax

        # Velocity heuristic: reward deliberate forward motion without punishing
        # information-seeking turns. Geometry-aware turn reward belongs env-side.
        del angular_velocity
        vel_bonus = self._velocity_weight * max(0.0, forward_velocity)

        # Intrinsic curiosity with annealing
        beta = self.beta
        intrinsic = beta * intrinsic_reward

        # Loop penalty: fires when similarity exceeds threshold
        loop_pen = self._lambda_loop * max(0.0, loop_similarity - self._tau)

        total = extrinsic + col_penalty + tax + vel_bonus + intrinsic - loop_pen

        return ShapedReward(
            extrinsic=extrinsic,
            collision_penalty=col_penalty,
            existential_tax=tax,
            velocity_bonus=vel_bonus,
            intrinsic=intrinsic,
            loop_penalty=loop_pen,
            total=total,
        )

    def shape_batch(
        self,
        *,
        raw_rewards: Tensor,
        dones: Tensor,
        forward_velocities: Tensor,
        angular_velocities: Tensor,
        intrinsic_rewards: Tensor,
        loop_similarities: Tensor,
    ) -> Tensor:
        """Vectorized reward shaping for a batch of transitions.

        All inputs are 1-D tensors of the same length.

        Returns:
            (B,) tensor of total shaped rewards.

        """
        del angular_velocities
        total: Tensor = self._shape_batch_fn(
            raw_rewards,
            dones,
            forward_velocities,
            intrinsic_rewards,
            loop_similarities,
            self._collision_penalty,
            self._existential_tax,
            self._velocity_weight,
            self.beta,
            self._lambda_loop,
            self._tau,
        )
        return total
