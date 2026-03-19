"""Actor-Critic heads for continuous 4-DOF drone control."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__: list[str] = ["ActorCriticHeads"]

_SQRT2: float = math.sqrt(2.0)
_HALF_LOG_TWO_PI: float = 0.5 * math.log(2.0 * math.pi)


class ActorCriticHeads(nn.Module):
    """Separate actor (Gaussian policy) and critic (value) heads.

    Actor outputs a 4-dim mean and uses a learnable log_std parameter.
    Action dimensions: [fwd, vert, lat, yaw].

    Steering convention
    -------------------
    All four dimensions are **normalised steering commands** in [-1, 1].
    The policy has no concept of speed — it only expresses *directional
    intent* (forward/backward, up/down, left/right, turn).

    Actual velocity (m/s, rad/s) is determined by ``drone_max_speed`` and
    related parameters on the **backend** (Environment config).
    Speed is also **dynamic**: the backend scales it with front-hemisphere
    proximity so the drone crawls near walls and races in open space.
    This means the same trained model works at any flight speed.

    Critic outputs a scalar state-value estimate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        *,
        max_forward: float = 1.0,
        max_vertical: float = 1.0,
        max_lateral: float = 1.0,
        max_yaw: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = 4

        self.register_buffer(
            "action_scales",
            torch.tensor(
                [max_forward, max_vertical, max_lateral, max_yaw],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Actor MLP
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.action_dim),
            nn.Tanh(),  # outputs in [-1, 1], scaled by action_scales
        )

        # Learnable log standard deviation (per action dim)
        # INITIAL DAMPING: Start with smaller variance (log_std = -0.5 -> std ~ 0.6)
        # to stop the drone from "shaking" near walls at start of training.
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))

        # Critic MLP
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # ── PPO-standard orthogonal initialization ──
        # ReLU-preceding layers: gain = sqrt(2)
        # Final actor layer: gain = 0.01 — keeps pre-Tanh activations
        # near zero so Tanh stays in its linear region at init.
        # Final critic layer: gain = 1.0 — standard for value heads.
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization (PPO best-practice).

        * Hidden (ReLU) layers: ``gain = sqrt(2)``
        * Actor output (pre-Tanh): ``gain = 0.01`` — critical to prevent
          Tanh saturation that kills gradients after the first update.
        * Critic output: ``gain = 1.0``
        """
        for module in (self.actor, self.critic):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=_SQRT2)
                    nn.init.zeros_(layer.bias)

        # Actor final linear (index 2 in Sequential: Linear→ReLU→Linear→Tanh)
        actor_final = self.actor[2]
        assert isinstance(actor_final, nn.Linear)
        nn.init.orthogonal_(actor_final.weight, gain=0.01)
        nn.init.zeros_(actor_final.bias)

        # Critic final linear (index 2 in Sequential: Linear→ReLU→Linear)
        critic_final = self.critic[2]
        assert isinstance(critic_final, nn.Linear)
        nn.init.orthogonal_(critic_final.weight, gain=1.0)
        nn.init.zeros_(critic_final.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Compute action mean and state value.

        Args:
            features: (B, D) feature tensor from encoder (or Mamba core).

        Returns:
            action_mean: (B, 4) scaled action means.
            value: (B,) state-value estimates.

        """
        action_mean = self.action_mean(features)
        value: Tensor = self.critic(features).squeeze(-1)  # (B,)
        return action_mean, value

    def action_mean(self, features: Tensor) -> Tensor:
        """Compute the scaled actor mean without running the critic head."""
        raw_mean: Tensor = self.actor(features)
        action_scales: Tensor = self.action_scales
        return raw_mean * action_scales

    def log_prob(self, features: Tensor, actions: Tensor) -> Tensor:
        """Evaluate log probability of given actions under the current policy.

        Args:
            features: (B, D) feature tensor.
            actions: (B, 4) action tensor.

        Returns:
            log_prob: (B,) log probabilities (sum over action dims).

        """
        action_mean = self.action_mean(features)
        inv_var = torch.exp(-2.0 * self.log_std)
        # Gaussian log-prob per dim, then sum.
        log_p = -0.5 * ((actions - action_mean) ** 2) * inv_var - self.log_std - _HALF_LOG_TWO_PI
        log_prob: Tensor = log_p.sum(dim=-1)
        return log_prob

    def entropy(self) -> Tensor:
        """Compute policy entropy (independent Gaussian).

        Returns:
            entropy: scalar — sum of per-dim differential entropies.

        """
        ent = 0.5 + _HALF_LOG_TWO_PI + self.log_std
        return ent.sum()

    def sample(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample actions from the policy distribution.

        Args:
            features: (B, D) feature tensor.

        Returns:
            actions: (B, 4) sampled actions.
            log_probs: (B,) log probabilities.
            values: (B,) state-value estimates.

        """
        action_mean = self.action_mean(features)
        values: Tensor = self.critic(features).squeeze(-1)
        std = self.log_std.exp()
        noise = torch.randn_like(action_mean)
        actions = action_mean + noise * std
        action_delta = actions.detach() - action_mean
        log_probs = (-0.5 * ((action_delta / std) ** 2) - self.log_std - _HALF_LOG_TWO_PI).sum(
            dim=-1
        )
        return actions, log_probs, values
