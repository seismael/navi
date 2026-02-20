"""Cognitive Mamba Policy — end-to-end neural policy for PPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from navi_actor.actor_critic import ActorCriticHeads
from navi_actor.mamba_core import Mamba2TemporalCore
from navi_actor.perception import FoveatedEncoder

__all__: list[str] = ["CognitiveMambaPolicy"]


class CognitiveMambaPolicy(nn.Module):  # type: ignore[misc]
    """Composes FoveatedEncoder -> Mamba2TemporalCore -> ActorCriticHeads.

    Five-stage cognitive flow:
      1. Foveated Encoder: (B,2,Az,El) -> (B,D) spatial embedding z_t
      2. (RND + EpisodicMemory operate externally on z_t)
      3. Mamba2 Temporal Core: z_t -> temporal features f_t
      4. Actor-Critic Heads: f_t -> actions + value
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        azimuth_bins: int = 64,
        elevation_bins: int = 32,
        max_forward: float = 1.2,
        max_vertical: float = 0.8,
        max_lateral: float = 0.8,
        max_yaw: float = 1.2,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins

        self.encoder = FoveatedEncoder(embedding_dim=embedding_dim)
        self.temporal_core = Mamba2TemporalCore(
            d_model=embedding_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.heads = ActorCriticHeads(
            input_dim=embedding_dim,
            max_forward=max_forward,
            max_vertical=max_vertical,
            max_lateral=max_lateral,
            max_yaw=max_yaw,
        )

    @property
    def device(self) -> torch.device:
        """Return the device of the first parameter."""
        return next(self.parameters()).device

    def _obs_to_tensor(self, obs: Any) -> Tensor:
        """Convert a DistanceMatrix observation to (1, 2, Az, El) tensor.

        Accepts either a DistanceMatrix (with .depth, .semantic, .valid_mask
        attributes) or a raw numpy array / torch Tensor.
        """
        if isinstance(obs, Tensor):
            if obs.dim() == 3:
                return obs.unsqueeze(0).to(self.device)
            return obs.to(self.device)

        # DistanceMatrix wire contract
        depth: np.ndarray[Any, Any] = np.asarray(obs.depth)
        semantic: np.ndarray[Any, Any] = np.asarray(obs.semantic)

        # Stack depth + semantic as 2-channel image
        stacked = np.stack([depth.astype(np.float32), semantic.astype(np.float32)])
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def forward(
        self,
        obs_tensor: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Full forward pass: encode → temporal core → actor-critic.

        Args:
            obs_tensor: (B, 2, Az, El) observation tensor.
            hidden: recurrent hidden state for temporal core.

        Returns:
            actions: (B, 4) sampled actions.
            log_probs: (B,) log probabilities.
            values: (B,) state-value estimates.
            new_hidden: updated hidden state.

        """
        z_t = self.encoder(obs_tensor)

        # Temporal core: single-step inference
        features, new_hidden = self.temporal_core.forward_step(z_t, hidden)

        actions, log_probs, values = self.heads.sample(features)
        return actions, log_probs, values, new_hidden

    def encode(self, obs_tensor: Tensor) -> Tensor:
        """Extract spatial embedding z_t without temporal processing.

        Useful for RND / episodic memory which operate on raw embeddings.

        Args:
            obs_tensor: (B, 2, Az, El) observation tensor.

        Returns:
            z_t: (B, D) spatial embedding.

        """
        return self.encoder(obs_tensor)

    def evaluate(
        self,
        obs_tensor: Tensor,
        actions: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Evaluate actions under current policy (for PPO loss computation).

        Args:
            obs_tensor: (B, 2, Az, El) observation tensor.
            actions: (B, 4) actions to evaluate.
            hidden: recurrent hidden state.

        Returns:
            log_probs: (B,) log probabilities of given actions.
            values: (B,) state-value estimates.
            entropy: scalar entropy.
            new_hidden: updated hidden state.

        """
        z_t = self.encoder(obs_tensor)
        features, new_hidden = self.temporal_core.forward_step(z_t, hidden)

        log_probs = self.heads.log_prob(features, actions)
        _, values = self.heads(features)
        entropy = self.heads.entropy()
        return log_probs, values, entropy, new_hidden

    def evaluate_sequence(
        self,
        obs_seq: Tensor,
        actions_seq: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Evaluate a full sequence for BPTT training.

        Args:
            obs_seq: (B, T, 2, Az, El) observation sequence.
            actions_seq: (B, T, 4) action sequence.
            hidden: initial hidden state.

        Returns:
            log_probs: (B*T,) log probabilities.
            values: (B*T,) value estimates.
            entropy: scalar entropy.
            new_hidden: final hidden state.

        """
        batch, seq_len = obs_seq.shape[:2]
        # Encode all frames
        flat_obs = obs_seq.reshape(batch * seq_len, *obs_seq.shape[2:])
        z_seq = self.encoder(flat_obs).reshape(batch, seq_len, -1)

        # Temporal core: full sequence
        features_seq, new_hidden = self.temporal_core.forward(z_seq, hidden)
        flat_features = features_seq.reshape(batch * seq_len, -1)
        flat_actions = actions_seq.reshape(batch * seq_len, -1)

        log_probs = self.heads.log_prob(flat_features, flat_actions)
        _, values = self.heads(flat_features)
        entropy = self.heads.entropy()
        return log_probs, values, entropy, new_hidden

    @torch.no_grad()  # type: ignore[misc]
    def act(
        self,
        obs: Any,
        step_id: int,
        hidden: Tensor | None = None,
    ) -> tuple[list[float], Tensor | None]:
        """Inference-mode action selection for the server loop.

        Args:
            obs: DistanceMatrix observation.
            step_id: current time step (unused in Phase 1).
            hidden: recurrent hidden state.

        Returns:
            action_list: [fwd, vert, lat, yaw] as Python floats.
            new_hidden: updated hidden state.

        """
        self.eval()
        obs_tensor = self._obs_to_tensor(obs)
        actions, _, _, new_hidden = self.forward(obs_tensor, hidden)
        action_list: list[float] = actions.squeeze(0).cpu().tolist()
        return action_list, new_hidden

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model state dict."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> CognitiveMambaPolicy:
        """Load model from checkpoint."""
        policy = cls(**kwargs)
        state = torch.load(path, weights_only=True, map_location="cpu")
        policy.load_state_dict(state)
        return policy
