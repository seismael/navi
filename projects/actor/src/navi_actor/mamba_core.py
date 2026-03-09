"""Canonical mambapy temporal core for sequence modeling."""

from __future__ import annotations

import logging

from torch import Tensor, nn

__all__: list[str] = ["Mamba2TemporalCore"]

_LOGGER = logging.getLogger(__name__)


class Mamba2TemporalCore(nn.Module):  # type: ignore[misc]
    """Temporal sequence model using canonical mambapy Mamba.

    Args:
        d_model: embedding dimension (must match encoder output).
        d_state: SSM state expansion factor.
        d_conv: local convolution width.
        expand: channel expansion factor.

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
        try:
            from mambapy.mamba import Mamba, MambaConfig  # type: ignore[import-untyped]
        except Exception as exc:  # pragma: no cover - environment-dependent import path
            raise RuntimeError(
                "Canonical temporal core requires mambapy. "
                "Run `uv sync --project projects/actor --python 3.12` before launching actor runtime."
            ) from exc

        self.core = Mamba(
            MambaConfig(
                d_model=d_model,
                n_layers=1,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand,
            )
        )
        self.norm = nn.LayerNorm(d_model)
        self.aux_proj = nn.Linear(3, d_model)
        _LOGGER.info(
            "Mamba2TemporalCore: canonical mambapy runtime active",
        )

    def forward(
        self,
        z_seq: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a sequence of spatial embeddings through the canonical core.

        Args:
            z_seq: (B, T, D) sequence of encoder outputs.
            hidden: reserved for API compatibility; ignored for this core.
            dones: reserved for API compatibility; ignored for this core.
            aux_tensor: (B, T, 3) optional auxiliary tensor.

        Returns:
            output: (B, T, D) temporal representations.
            new_hidden: always ``None`` for this core.

        """
        del hidden, dones
        if aux_tensor is not None:
            z_seq = z_seq + self.aux_proj(aux_tensor)

        out: Tensor = self.core(z_seq)
        out = self.norm(out + z_seq)
        return out, None

    def forward_step(
        self,
        z_t: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a single time step for online inference."""
        del hidden
        z_seq = z_t.unsqueeze(1)
        aux_seq = aux_tensor.unsqueeze(1) if aux_tensor is not None else None
        out_seq, _ = self.forward(z_seq, aux_tensor=aux_seq)
        return out_seq.squeeze(1), None
