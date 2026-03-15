"""Canonical Windows-friendly Mamba temporal core."""

from __future__ import annotations

import logging

from torch import Tensor, nn

__all__: list[str] = ["MambapyTemporalCore"]

_LOGGER = logging.getLogger(__name__)


class MambapyTemporalCore(nn.Module):  # type: ignore[misc]
    """Windows-friendly Mamba temporal core used on the active machine."""

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
                "Canonical Mamba runtime requires mambapy to be installed.",
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
        _LOGGER.info("MambapyTemporalCore: canonical Windows-friendly Mamba runtime active")

    def forward(
        self,
        z_seq: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
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
        del hidden
        z_seq = z_t.unsqueeze(1)
        aux_seq = aux_tensor.unsqueeze(1) if aux_tensor is not None else None
        out_seq, _ = self.forward(z_seq, aux_tensor=aux_seq)
        return out_seq.squeeze(1), None
