"""Random Network Distillation (RND) for intrinsic curiosity rewards."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__: list[str] = ["RNDModule"]


class RNDModule(nn.Module):  # type: ignore[misc]
    """RND intrinsic curiosity module.

    Maintains a frozen *target* network and a trainable *predictor* network.
    The intrinsic reward is the prediction error ``||target(z) - predictor(z)||^2``,
    normalized by a running mean/std for stability.

    Args:
        input_dim: dimensionality of the spatial embedding z_t.
        hidden_dim: hidden layer width for both networks.
        output_dim: output dimensionality for the projection.

    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Target network — fixed, randomly initialized
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        # Freeze target
        for p in self.target.parameters():
            p.requires_grad = False

        # Predictor network — trainable (same architecture as target so it
        # cannot perfectly replicate the target from different init weights,
        # keeping a residual prediction error that serves as the novelty signal).
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        # Running statistics for reward normalization
        self.register_buffer("_running_mean", torch.zeros(1))
        self.register_buffer("_running_var", torch.ones(1))
        self.register_buffer("_count", torch.tensor(1e-4))

        # Declare types for mypy (assigned by register_buffer)
        self._running_mean: Tensor
        self._running_var: Tensor
        self._count: Tensor

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Compute target and predictor projections.

        Args:
            z: (B, D) spatial embedding batch.

        Returns:
            target_out: (B, output_dim) frozen projection.
            predictor_out: (B, output_dim) trainable projection.

        """
        with torch.no_grad():
            target_out: Tensor = self.target(z)
        predictor_out: Tensor = self.predictor(z)
        return target_out, predictor_out

    def distillation_loss(self, z: Tensor) -> Tensor:
        """Compute the RND distillation loss (MSE prediction error).

        This loss is used to train the predictor network.

        Args:
            z: (B, D) spatial embedding batch.

        Returns:
            loss: scalar MSE between target and predictor outputs.

        """
        target_out, predictor_out = self.forward(z)
        return ((target_out - predictor_out) ** 2).mean()

    @torch.no_grad()  # type: ignore[misc]
    def intrinsic_reward(self, z: Tensor) -> Tensor:
        """Compute normalized intrinsic reward for a batch of embeddings.

        Args:
            z: (B, D) spatial embeddings.

        Returns:
            reward: (B,) normalized intrinsic reward scalars.

        """
        target_out, predictor_out = self.forward(z)
        raw_error = ((target_out - predictor_out) ** 2).sum(dim=-1)  # (B,)

        # Update running statistics (Chan's parallel algorithm)
        batch_mean = raw_error.mean()
        batch_var = (
            raw_error.var(correction=0)
            if raw_error.numel() > 1
            else torch.zeros(1, device=z.device)
        )
        batch_count = float(raw_error.numel())

        delta = batch_mean - self._running_mean
        total_count = self._count + batch_count
        self._running_mean = self._running_mean + delta * batch_count / total_count
        m_a = self._running_var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._count * batch_count / total_count
        self._running_var = m2 / total_count
        self._count = total_count

        # Normalize
        std = torch.sqrt(self._running_var + 1e-8)
        normalized: Tensor = (raw_error - self._running_mean) / std
        return normalized.clamp(-5.0, 5.0)

    @torch.no_grad()  # type: ignore[misc]
    def merge_running_stats(
        self,
        other_mean: Tensor,
        other_var: Tensor,
        other_count: Tensor,
    ) -> None:
        """Merge another set of running statistics into this module.

        Uses Chan's parallel algorithm to combine two independent
        running-mean/variance estimators without losing precision.

        This is used by the central learner in parallel training to
        accumulate RND running statistics from multiple workers.

        Args:
            other_mean: (1,) running mean from the other source.
            other_var: (1,) running variance from the other source.
            other_count: scalar sample count from the other source.

        """
        delta = other_mean - self._running_mean
        total_count = self._count + other_count
        self._running_mean = (
            self._running_mean + delta * other_count / total_count
        )
        m_a = self._running_var * self._count
        m_b = other_var * other_count
        m2 = m_a + m_b + delta**2 * self._count * other_count / total_count
        self._running_var = m2 / total_count
        self._count = total_count

    def get_running_stats(self) -> tuple[Tensor, Tensor, Tensor]:
        """Return current running statistics for serialization.

        Returns:
            Tuple of (running_mean, running_var, count) tensors.

        """
        return (
            self._running_mean.clone(),
            self._running_var.clone(),
            self._count.clone(),
        )
