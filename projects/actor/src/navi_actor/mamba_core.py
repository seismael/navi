"""Mamba2 temporal core with GRU fallback for sequence modeling."""

from __future__ import annotations

import logging

from torch import Tensor, nn

__all__: list[str] = ["Mamba2TemporalCore"]

_LOGGER = logging.getLogger(__name__)

_HAS_MAMBA = False
try:
    from mamba_ssm import Mamba2  # type: ignore[import-untyped]

    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False


class _GRUFallback(nn.Module):  # type: ignore[misc]
    """Single-layer GRU used when mamba-ssm is not available."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(
        self, x: Tensor, hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Run GRU forward.

        Args:
            x: (B, T, D) input sequence.
            hidden: (1, B, D) hidden state or None.

        Returns:
            output: (B, T, D) outputs.
            new_hidden: (1, B, D) final hidden state.

        """
        out, h_n = self.gru(x, hidden)
        return out, h_n


class Mamba2TemporalCore(nn.Module):  # type: ignore[misc]
    """Temporal sequence model: Mamba2 SSM when available, GRU fallback otherwise.

    Processes a sequence of embeddings and maintains a recurrent hidden state
    for online inference.

    Args:
        d_model: embedding dimension (must match encoder output).
        d_state: SSM state expansion factor (Mamba2 only).
        d_conv: local convolution width (Mamba2 only).
        expand: channel expansion factor (Mamba2 only).

    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_mamba = _HAS_MAMBA

        if _HAS_MAMBA:
            self.core: nn.Module = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            _LOGGER.info("Mamba2TemporalCore: using native Mamba2 SSM")
        else:
            self.core = _GRUFallback(d_model)
            _LOGGER.info("Mamba2TemporalCore: using GRU fallback (mamba-ssm not found)")

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, z_seq: Tensor, hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a sequence of spatial embeddings through the temporal core.

        Args:
            z_seq: (B, T, D) sequence of encoder outputs.
            hidden: recurrent state from previous call.
                - GRU: (1, B, D) tensor.
                - Mamba2: None (stateless in training; uses internal cache for
                  inference).

        Returns:
            output: (B, T, D) temporal representations.
            new_hidden: updated hidden state.

        """
        if self.use_mamba:
            # Mamba2 is stateless during training (full sequence at once).
            # For single-step inference, we wrap in a length-1 sequence.
            out: Tensor = self.core(z_seq)
            out = self.norm(out + z_seq)  # residual + norm
            return out, None

        # GRU fallback
        out, new_h = self.core(z_seq, hidden)
        out = self.norm(out + z_seq)  # residual + norm
        return out, new_h

    def forward_step(
        self, z_t: Tensor, hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a single time step for online inference.

        Args:
            z_t: (B, D) single embedding.
            hidden: recurrent state.

        Returns:
            output: (B, D) temporal representation.
            new_hidden: updated hidden state.

        """
        # Wrap as (B, 1, D) sequence
        z_seq = z_t.unsqueeze(1)
        out_seq, new_hidden = self.forward(z_seq, hidden)
        return out_seq.squeeze(1), new_hidden
