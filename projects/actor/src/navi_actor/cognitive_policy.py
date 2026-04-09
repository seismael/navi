"""Cognitive policy — end-to-end neural policy for PPO training.

**This module is sacred.**  The cognitive pipeline
(RayViTEncoder → TemporalCore → EpisodicMemory → ActorCriticHeads)
is never modified to accommodate new data sources or sensor types.
External data connects only through ``DatasetAdapter`` instances in
``environment/backends/`` that transform raw observations *to*
the engine's canonical ``(B, 3, Az, El)`` DistanceMatrix input.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from navi_actor.actor_critic import ActorCriticHeads
from navi_actor.config import SUPPORTED_TEMPORAL_CORES, TemporalCoreName
from navi_actor.gru_core import GRUTemporalCore
from navi_actor.mamba2_core import Mamba2SSDTemporalCore
from navi_actor.mambapy_core import MambapyTemporalCore

_log = logging.getLogger(__name__)

__all__: list[str] = ["CognitiveMambaPolicy", "PolicyEvalStageMetrics"]


@dataclass(frozen=True)
class _PolicyStageTiming:
    wall_ms: float = 0.0
    device_ms: float = 0.0


@contextlib.contextmanager
def _policy_stage_timer(*, device: torch.device, use_cuda_events: bool) -> Any:
    elapsed = _PolicyStageTiming()
    if use_cuda_events and device.type == "cuda":
        start_event: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        end_event: Any = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        t_start = time.perf_counter()
        torch.cuda.synchronize(device)
        start_event.record()
        try:
            yield elapsed
        finally:
            end_event.record()
            torch.cuda.synchronize(device)
            object.__setattr__(elapsed, "wall_ms", (time.perf_counter() - t_start) * 1000)
            object.__setattr__(elapsed, "device_ms", start_event.elapsed_time(end_event))
        return

    t_start = time.perf_counter()
    try:
        yield elapsed
    finally:
        object.__setattr__(elapsed, "wall_ms", (time.perf_counter() - t_start) * 1000)


@dataclass(frozen=True)
class PolicyEvalStageMetrics:
    encode_ms: float = 0.0
    temporal_ms: float = 0.0
    heads_ms: float = 0.0
    encode_device_ms: float = 0.0
    temporal_device_ms: float = 0.0
    heads_device_ms: float = 0.0


def _build_temporal_core(
    *,
    temporal_core: TemporalCoreName,
    embedding_dim: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> nn.Module:
    if temporal_core == "mambapy":
        return MambapyTemporalCore(
            d_model=embedding_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    if temporal_core == "gru":
        return GRUTemporalCore(d_model=embedding_dim)
    if temporal_core == "mamba2":
        return Mamba2SSDTemporalCore(
            d_model=embedding_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
    supported = ", ".join(SUPPORTED_TEMPORAL_CORES)
    raise ValueError(f"Unsupported temporal core '{temporal_core}'. Expected one of: {supported}")


class CognitiveMambaPolicy(nn.Module):  # type: ignore[misc]
    """Composes RayViTEncoder -> TemporalCore -> ActorCriticHeads.

    Five-stage cognitive flow:
      1. RayViT Encoder: (B,3,Az,El) -> (B,D) spatial embedding z_t
      2. (RND + EpisodicMemory operate externally on z_t)
    3. Temporal Core: z_t -> temporal features f_t
      4. Actor-Critic Heads: f_t -> actions + value
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        temporal_core: TemporalCoreName = "mamba2",
        azimuth_bins: int = 128,
        elevation_bins: int = 24,
        max_forward: float = 1.0,
        max_vertical: float = 1.0,
        max_lateral: float = 1.0,
        max_yaw: float = 1.0,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temporal_core_name = temporal_core
        self.azimuth_bins = azimuth_bins
        self.elevation_bins = elevation_bins

        from navi_actor.perception import RayViTEncoder

        self.encoder = RayViTEncoder(embedding_dim=embedding_dim)

        # Attempt torch.compile on encoder for kernel fusion on supported GPUs.
        # Falls back to eager on unsupported hardware or missing compiler.
        _compile_ok = False
        _compile_reason = "torch.compile unavailable"
        try:
            capability = torch.cuda.get_device_capability()
            if tuple(int(part) for part in capability) < (7, 0):
                _compile_reason = f"CUDA capability {capability[0]}.{capability[1]} is below the Triton minimum 7.0"
            else:
                # Validate that inductor can find a C/C++ compiler before wrapping.
                from torch._inductor.cpp_builder import get_cpp_compiler

                get_cpp_compiler()
                _compile_ok = True
        except Exception as exc:
            _compile_ok = False
            if _compile_reason == "torch.compile unavailable":
                _compile_reason = str(exc)
        if _compile_ok:
            try:
                # fullgraph=False: encoder has dynamic padding (pad_az/pad_el)
                # that prevents full-graph capture; jit.script also incompatible.
                self.encoder = torch.compile(  # type: ignore[assignment]
                    self.encoder,
                    fullgraph=False,
                    dynamic=False,
                    mode="reduce-overhead",
                )
                _log.info("RayViTEncoder: torch.compile activated")
            except Exception:
                _log.info("RayViTEncoder: torch.compile wrapping failed, using eager mode")
        else:
            _log.info(
                "RayViTEncoder: skipping torch.compile (%s), using eager mode", _compile_reason
            )

        self.temporal_core = _build_temporal_core(
            temporal_core=temporal_core,
            embedding_dim=embedding_dim,
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
        """Convert a DistanceMatrix observation to (1, 3, Az, El) tensor.

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
        valid: np.ndarray[Any, Any] = np.asarray(obs.valid_mask)

        # Stack depth + semantic + valid as 3-channel image
        # Strip leading env dimension if present
        if depth.ndim == 3:
            depth = depth[0]
            semantic = semantic[0]
            valid = valid[0]

        stacked = np.stack(
            [
                depth.astype(np.float32),
                semantic.astype(np.float32),
                valid.astype(np.float32),
            ]
        )
        return torch.from_numpy(stacked).unsqueeze(0).to(self.device)

    def forward(
        self,
        obs_tensor: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Full forward pass: encode -> temporal core -> actor-critic.

        Args:
            obs_tensor: (B, 3, Az, El) observation tensor.
            hidden: recurrent hidden state for temporal core.
            aux_tensor: (B, 3) optional auxiliary tensor.

        Returns:
            actions: (B, 4) sampled actions.
            log_probs: (B,) log probabilities.
            values: (B,) state-value estimates.
            new_hidden: updated hidden state.
            z_t: (B, D) spatial embedding (for RND / episodic memory).

        """
        z_t = self.encoder(obs_tensor)

        # Temporal core: single-step inference
        features, new_hidden = self.temporal_core.forward_step(z_t, hidden, aux_tensor)

        actions, log_probs, values = self.heads.sample(features)
        return actions, log_probs, values, new_hidden, z_t

    def encode(self, obs_tensor: Tensor) -> Tensor:
        """Extract spatial embedding z_t without temporal processing.

        Useful for RND / episodic memory which operate on raw embeddings.

        Args:
            obs_tensor: (B, 3, Az, El) observation tensor.

        Returns:
            z_t: (B, D) spatial embedding.

        """
        z_t: Tensor = self.encoder(obs_tensor)
        return z_t

    def evaluate(
        self,
        obs_tensor: Tensor,
        actions: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Evaluate actions under current policy (for PPO loss computation).

        Args:
            obs_tensor: (B, 3, Az, El) observation tensor.
            actions: (B, 4) actions to evaluate.
            hidden: recurrent hidden state.
            aux_tensor: (B, 3) optional auxiliary tensor.

        Returns:
            log_probs: (B,) log probabilities of given actions.
            values: (B,) state-value estimates.
            entropy: scalar entropy.
            new_hidden: updated hidden state.
            z_t: (B, D) spatial embedding (for RND distillation).

        """
        log_probs, values, entropy, new_hidden, z_t, _metrics = self.evaluate_profiled(
            obs_tensor,
            actions,
            hidden=hidden,
            aux_tensor=aux_tensor,
            use_cuda_events=False,
        )
        return log_probs, values, entropy, new_hidden, z_t

    def evaluate_profiled(
        self,
        obs_tensor: Tensor,
        actions: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
        *,
        use_cuda_events: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor, PolicyEvalStageMetrics]:
        """Evaluate actions and return per-stage timings for diagnostic attribution."""
        with _policy_stage_timer(
            device=obs_tensor.device, use_cuda_events=use_cuda_events
        ) as encode_timing:
            z_t = self.encoder(obs_tensor)
        with _policy_stage_timer(
            device=obs_tensor.device, use_cuda_events=use_cuda_events
        ) as temporal_timing:
            features, new_hidden = self.temporal_core.forward_step(z_t, hidden, aux_tensor)
        with _policy_stage_timer(
            device=obs_tensor.device, use_cuda_events=use_cuda_events
        ) as heads_timing:
            log_probs = self.heads.log_prob(features, actions)
            # Stop-gradient: critic sees detached features so value-loss
            # gradients never flow back through the shared backbone.
            values: Tensor = self.heads.critic(features.detach()).squeeze(-1)
            entropy = self.heads.entropy()
        return (
            log_probs,
            values,
            entropy,
            new_hidden,
            z_t.detach(),
            PolicyEvalStageMetrics(
                encode_ms=encode_timing.wall_ms,
                temporal_ms=temporal_timing.wall_ms,
                heads_ms=heads_timing.wall_ms,
                encode_device_ms=encode_timing.device_ms,
                temporal_device_ms=temporal_timing.device_ms,
                heads_device_ms=heads_timing.device_ms,
            ),
        )

    def evaluate_sequence(
        self,
        obs_seq: Tensor,
        actions_seq: Tensor,
        hidden: Tensor | None = None,
        aux_seq: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor]:
        """Evaluate a full sequence for BPTT training.

        Args:
            obs_seq: (B, T, 3, Az, El) observation sequence.
            actions_seq: (B, T, 4) action sequence.
            hidden: initial hidden state.
            aux_seq: (B, T, 3) optional auxiliary tensor sequence.

        Returns:
            log_probs: (B*T,) log probabilities.
            values: (B*T,) value estimates.
            entropy: scalar entropy.
            new_hidden: final hidden state.
            z_t: (B*T, D) spatial embeddings (for RND distillation).

        """
        log_probs, values, entropy, new_hidden, z_flat, _metrics = self.evaluate_sequence_profiled(
            obs_seq,
            actions_seq,
            hidden=hidden,
            aux_seq=aux_seq,
            use_cuda_events=False,
        )
        return log_probs, values, entropy, new_hidden, z_flat

    def evaluate_sequence_profiled(
        self,
        obs_seq: Tensor,
        actions_seq: Tensor,
        hidden: Tensor | None = None,
        aux_seq: Tensor | None = None,
        *,
        use_cuda_events: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor, PolicyEvalStageMetrics]:
        """Evaluate a full PPO sequence minibatch and return per-stage timings."""
        batch, seq_len = obs_seq.shape[:2]
        flat_obs = obs_seq.reshape(batch * seq_len, *obs_seq.shape[2:])
        with _policy_stage_timer(
            device=obs_seq.device, use_cuda_events=use_cuda_events
        ) as encode_timing:
            z_flat = self.encoder(flat_obs)
            z_seq = z_flat.reshape(batch, seq_len, -1)
        with _policy_stage_timer(
            device=obs_seq.device, use_cuda_events=use_cuda_events
        ) as temporal_timing:
            features_seq, new_hidden = self.temporal_core.forward(
                z_seq,
                hidden,
                aux_tensor=aux_seq,
            )
            flat_features = features_seq.reshape(batch * seq_len, -1)
        with _policy_stage_timer(
            device=obs_seq.device, use_cuda_events=use_cuda_events
        ) as heads_timing:
            flat_actions = actions_seq.reshape(batch * seq_len, -1)
            log_probs = self.heads.log_prob(flat_features, flat_actions)
            # Stop-gradient: critic sees detached features so value-loss
            # gradients never flow back through the shared backbone.
            values: Tensor = self.heads.critic(flat_features.detach()).squeeze(-1)
            entropy = self.heads.entropy()
        return (
            log_probs,
            values,
            entropy,
            new_hidden,
            z_flat.detach(),
            PolicyEvalStageMetrics(
                encode_ms=encode_timing.wall_ms,
                temporal_ms=temporal_timing.wall_ms,
                heads_ms=heads_timing.wall_ms,
                encode_device_ms=encode_timing.device_ms,
                temporal_device_ms=temporal_timing.device_ms,
                heads_device_ms=heads_timing.device_ms,
            ),
        )

    @torch.no_grad()  # type: ignore[untyped-decorator]
    def act(
        self,
        obs: Any,
        step_id: int,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[list[float], Tensor | None]:
        """Inference-mode action selection for the server loop.

        Args:
            obs: DistanceMatrix observation.
            step_id: current time step.
            hidden: recurrent hidden state.
            aux_tensor: (1, 3) optional auxiliary tensor.

        Returns:
            action_list: [fwd, vert, lat, yaw] as Python floats.
            new_hidden: updated hidden state.

        """
        del step_id
        self.eval()
        obs_tensor = self._obs_to_tensor(obs)
        actions, _, _, new_hidden, _ = self.forward(obs_tensor, hidden, aux_tensor=aux_tensor)
        action_list = [float(value) for value in actions.squeeze(0).cpu().tolist()]
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
        """Load model from checkpoint.

        Accepts plain state-dicts (from ``save_checkpoint``) and v3
        training snapshots that wrap the model weights under a
        ``policy_state_dict`` key.
        """
        policy = cls(**kwargs)
        state = torch.load(path, weights_only=False, map_location="cpu")
        if isinstance(state, dict) and "policy_state_dict" in state:
            state = state["policy_state_dict"]
        policy.load_state_dict(state)
        return policy
