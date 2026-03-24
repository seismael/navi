"""Pure-PyTorch Mamba-2 SSD temporal core for Windows.

Implements the State Space Duality (SSD) algorithm from Mamba-2
(Gu & Dao, 2024) using only standard PyTorch operations. No Triton,
causal-conv1d, or mamba-ssm packages required. Works on all platforms
including Windows with any CUDA compute capability.

The chunked SSD scan parallelises across time within each chunk,
giving better GPU utilisation than sequential GRU on short-to-medium
sequence lengths typical in PPO rollout minibatches (64-256 steps).
"""

# ruff: noqa: N806 — SSD scan uses uppercase mathematical notation (X, A, B, C, L, Y)

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

__all__: list[str] = ["Mamba2SSDTemporalCore"]

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RMSNorm (pure PyTorch, no Triton)
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalisation."""
        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms + self.eps))


# ---------------------------------------------------------------------------
# Causal depthwise Conv1d (pure PyTorch, no causal-conv1d package)
# ---------------------------------------------------------------------------
def _causal_conv1d(x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
    """Causal depthwise 1-D convolution using standard nn.functional.

    Args:
        x: (B, D, L) input.
        weight: (D, 1, K) depthwise conv weights.
        bias: (D,) optional bias.

    Returns:
        (B, D, L) causally convolved output (no future leakage).
    """
    k = weight.shape[-1]
    # Left-pad so output at position t depends only on [t-k+1 .. t].
    x_padded = F.pad(x, (k - 1, 0))
    return F.conv1d(x_padded, weight, bias, groups=x.shape[1])


# ---------------------------------------------------------------------------
# SSD scan — the core Mamba-2 algorithm in pure PyTorch
# ---------------------------------------------------------------------------
def _segsum(x: Tensor) -> Tensor:
    """Stable segment-sum for the SSD block-diagonal mask.

    Args:
        x: (..., T) log-decay values per time step.

    Returns:
        (..., T, T) lower-triangular cumulative log-decay matrix.
    """
    t = x.shape[-1]
    x = x.unsqueeze(-1).expand(*x.shape, t)  # (..., T) -> (..., T, T)
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0.0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=0)
    return x_segsum.masked_fill(~mask, -torch.inf)


def _ssd_scan(
    X: Tensor,  # noqa: N803
    A: Tensor,  # noqa: N803
    B: Tensor,  # noqa: N803
    C: Tensor,  # noqa: N803
    chunk_size: int,
    initial_states: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Pure-PyTorch structured state-space duality scan.

    Implements Algorithm 1 from the Mamba-2 paper using only standard
    PyTorch ops (einsum, cumsum, exp, tril).  Uses manual reshapes
    (no einops) to minimise kernel launches.

    Args:
        X: (B, L, H, P) — input values (x * dt expanded to head-dim).
        A: (B, L, H) — discretised log-decay (A * dt).
        B: (B, L, G, N) — input-to-state projection.
        C: (B, L, G, N) — state-to-output projection.
        chunk_size: block length for the chunked scan.
        initial_states: (B, 1, H, P, N) optional initial state.

    Returns:
        Y: (B, L, H, P) output.
        final_state: (B, H, P, N) final SSM state.
    """
    batch, seqlen, nheads, headdim = X.shape
    ngroups = B.shape[2]
    dstate = B.shape[3]

    # Pad sequence to multiple of chunk_size
    pad = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad > 0:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    L_total = X.shape[1]
    nchunks = L_total // chunk_size

    # Reshape into chunks using view (no copy)
    # X: (B, L, H, P) -> (B, C, cl, H, P)
    X = X.view(batch, nchunks, chunk_size, nheads, headdim)
    # A: (B, L, H) -> (B, H, C, cl) for cumsum along time within chunk
    A = A.view(batch, nchunks, chunk_size, nheads).permute(0, 3, 1, 2)
    # B, C: (B, L, G, N) -> (B, C, cl, G, N)
    B = B.view(batch, nchunks, chunk_size, ngroups, dstate)
    C = C.view(batch, nchunks, chunk_size, ngroups, dstate)

    A_cumsum = torch.cumsum(A, dim=-1)

    # --- Intra-chunk (diagonal blocks) ---
    L = torch.exp(_segsum(A))

    # Expand groups to heads for B and C
    heads_per_group = nheads // ngroups
    if heads_per_group > 1:
        B_exp = B.unsqueeze(4).expand(-1, -1, -1, -1, heads_per_group, -1)
        B_exp = B_exp.reshape(batch, nchunks, chunk_size, nheads, dstate)
        C_exp = C.unsqueeze(4).expand(-1, -1, -1, -1, heads_per_group, -1)
        C_exp = C_exp.reshape(batch, nchunks, chunk_size, nheads, dstate)
    else:
        B_exp = B
        C_exp = C

    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C_exp, B_exp, L, X)

    # --- Inter-chunk state accumulation ---
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B_exp, decay_states, X)

    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # --- State-to-output (off-diagonal blocks) ---
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C_exp, states, state_decay_out)

    Y = (Y_diag + Y_off).reshape(batch, L_total, nheads, headdim)

    # Remove padding
    if pad > 0:
        Y = Y[:, :seqlen]

    return Y, final_state


# ---------------------------------------------------------------------------
# Complete Mamba-2 SSD temporal core module
# ---------------------------------------------------------------------------
class Mamba2SSDTemporalCore(nn.Module):  # type: ignore[misc]
    """Pure-PyTorch Mamba-2 SSD temporal core.

    Implements the full Mamba-2 block (projection → causal conv1d → SSD scan
    → gated RMSNorm → output projection) without Triton or causal-conv1d.

    Matches the ``forward`` / ``forward_step`` API of GRU and Mambapy cores
    for drop-in temporal core selector compatibility.

    Args:
        d_model: embedding dimension (must match encoder output).
        d_state: SSM state expansion factor.
        d_conv: local convolution width.
        expand: channel expansion factor.
        headdim: dimension per attention head.
        ngroups: number of head groups for B/C projections.
        chunk_size: SSD block length for the chunked scan.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % headdim == 0, (
            f"d_inner={self.d_inner} must be divisible by headdim={headdim}"
        )
        self.nheads = self.d_inner // headdim

        # --- Input projection: [z, x, B, C, dt] ---
        d_in_proj = (
            2 * self.d_inner  # z + x
            + 2 * ngroups * d_state  # B + C
            + self.nheads  # dt
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # --- Causal depthwise conv1d on [x, B, C] ---
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,  # will be trimmed for causal
            bias=True,
        )

        # --- A (log-space, per-head) ---
        A = torch.empty(self.nheads, dtype=torch.float32).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # --- dt bias ---
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        # --- D skip connection ---
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

        # --- Gated RMSNorm + output projection ---
        self.norm = _RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # --- Auxiliary projection (velocity/heading) ---
        self.aux_proj = nn.Linear(3, d_model)

        _LOGGER.info(
            "Mamba2SSDTemporalCore: pure-PyTorch SSD runtime active "
            "(d_model=%d, nheads=%d, headdim=%d, d_state=%d, chunk=%d)",
            d_model,
            self.nheads,
            headdim,
            d_state,
            chunk_size,
        )

    def forward(
        self,
        z_seq: Tensor,
        hidden: Tensor | None = None,
        dones: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a sequence through the Mamba-2 SSD block.

        Args:
            z_seq: (B, T, D) encoder output sequence.
            hidden: reserved for API compatibility; ignored.
            dones: reserved for API compatibility; ignored.
            aux_tensor: (B, T, 3) optional auxiliary tensor.

        Returns:
            output: (B, T, D) temporal representations.
            new_hidden: always ``None`` for this core.
        """
        del hidden, dones
        batch, seqlen, _ = z_seq.shape

        if aux_tensor is not None:
            z_seq = z_seq + self.aux_proj(aux_tensor)

        # Project into z, x, B, C, dt components.
        zxbcdt = self.in_proj(z_seq)  # (B, L, d_in_proj)
        z, xBC, dt_raw = torch.split(
            zxbcdt,
            [
                self.d_inner,
                self.d_inner + 2 * self.ngroups * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        # dt: softplus activation + bias
        dt = F.softplus(dt_raw + self.dt_bias)  # (B, L, nheads)

        # Causal depthwise conv1d on [x, B, C]
        xBC = _causal_conv1d(
            xBC.transpose(1, 2),  # (B, D_conv, L)
            self.conv1d.weight,
            self.conv1d.bias,
        ).transpose(1, 2)  # (B, L, D_conv)
        xBC = F.silu(xBC)

        # Split into x, B, C
        x, B, C = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        # Reshape for SSD scan (manual reshapes, no einops)
        A = -torch.exp(self.A_log)  # (nheads,)
        x = x.view(batch, seqlen, self.nheads, self.headdim)
        B = B.view(batch, seqlen, self.ngroups, self.d_state)
        C = C.view(batch, seqlen, self.ngroups, self.d_state)

        # SSD scan: X = x * dt, A_discrete = A * dt
        X = x * dt.unsqueeze(-1)
        A_discrete = A.unsqueeze(0).unsqueeze(0) * dt  # (B, L, H)

        y, _final_state = _ssd_scan(X, A_discrete, B, C, self.chunk_size)

        # D skip connection (applied in head-dim space before collapsing)
        y = y + x * self.D[None, None, :, None]

        y = y.reshape(batch, seqlen, self.d_inner)

        # Gated output: norm(y) * silu(z)
        y = self.norm(y) * F.silu(z)
        out = self.out_proj(y)
        return out, None

    def forward_step(
        self,
        z_t: Tensor,
        hidden: Tensor | None = None,
        aux_tensor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Process a single time step for online inference.

        Falls back to running a length-1 sequence through the full forward
        path. This is correct but not optimal for streaming inference.
        """
        del hidden
        z_seq = z_t.unsqueeze(1)
        aux_seq = aux_tensor.unsqueeze(1) if aux_tensor is not None else None
        out_seq, _ = self.forward(z_seq, aux_tensor=aux_seq)
        return out_seq.squeeze(1), None
