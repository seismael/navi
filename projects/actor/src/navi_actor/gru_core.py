"""Native cuDNN GRU temporal core for sequence modeling."""

from __future__ import annotations

import logging

from torch import Tensor, nn

__all__: list[str] = ["GRUTemporalCore"]

_LOGGER = logging.getLogger(__name__)


class GRUTemporalCore(nn.Module):
    """Temporal sequence model using native PyTorch cuDNN GRU.

    This backend stays on the standard PyTorch + cuDNN stack with no custom
    extension build requirement on the active Windows training machines.

    It is available on the canonical trainer surface for controlled comparisons
    and may be promoted to the default runtime when bounded end-to-end training
    evidence proves the improvement.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.core = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
        self.aux_proj = nn.Linear(3, d_model)
        _LOGGER.info("GRUTemporalCore: native cuDNN GRU runtime active")

    def forward(
        self,
        z_seq: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del dones
        if aux_tensor is not None:
            z_seq = z_seq + self.aux_proj(aux_tensor)

        out, new_hidden = self.core(z_seq, hidden)
        return out, new_hidden

    def forward_step(
        self,
        z_t: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        z_seq = z_t.unsqueeze(1)
        aux_seq = aux_tensor.unsqueeze(1) if aux_tensor is not None else None
        out_seq, new_hidden = self.forward(z_seq, hidden, aux_tensor=aux_seq)
        return out_seq.squeeze(1), new_hidden
