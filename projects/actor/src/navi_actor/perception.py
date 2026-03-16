from __future__ import annotations

import torch
from torch import Tensor, nn

__all__: list[str] = ["RayViTEncoder"]


class RayViTEncoder(nn.Module):  # type: ignore[misc]
    """Vision Transformer encoder for spherical distance matrices.

    Implements §8.4 of ARCHITECTURE.md. Treats patches of the spherical
    grid as tokens and processes them via a Transformer Encoder.

    Uses FIXED SPHERICAL POSITIONAL ENCODINGS based on the canonical
    mapping where center=front and edges=back/up/down.

    Uses a [CLS] token for global spatial aggregation.

    Args:
        embedding_dim: output dimension D.
        patch_size: size of square patches.
        n_layers: number of Transformer layers.
        n_heads: number of attention heads.
        hidden_dim: internal Transformer dimension.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        patch_size: int = 8,
        n_layers: int = 2,
        n_heads: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Input projection: strided patch embedding avoids explicit patch materialization.
        self.patch_proj = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final projection to embedding_dim
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Cache for positional encodings
        self._pos_enc_cache: Tensor | None = None
        self._cache_key: tuple[int, int, int, torch.device, torch.dtype] | None = None

    def _get_fixed_pos_enc(
        self, n_az: int, n_el: int, dim: int, device: torch.device
    ) -> Tensor:
        """Compute fixed 2D sin/cos positional encodings for spherical patches."""
        dtype = self.cls_token.dtype
        cache_key = (n_az, n_el, dim, device, dtype)
        if self._pos_enc_cache is not None and self._cache_key == cache_key:
            return self._pos_enc_cache

        grid_az, grid_el = torch.meshgrid(
            torch.linspace(0, 2 * 3.14159, n_az, device=device, dtype=dtype),
            torch.linspace(-3.14159 / 2, 3.14159 / 2, n_el, device=device, dtype=dtype),
            indexing="ij",
        )
        # Flatten to patches: (N_patches,)
        flat_az = grid_az.reshape(-1)
        flat_el = grid_el.reshape(-1)

        # Frequency bands for encoding
        bands = dim // 4
        freqs = torch.pow(10000, -torch.arange(0, bands, device=device, dtype=dtype) / bands)

        # az_sin, az_cos, el_sin, el_cos
        enc_az = flat_az[:, None] * freqs[None, :]
        enc_el = flat_el[:, None] * freqs[None, :]

        pos_enc = torch.cat(
            [enc_az.sin(), enc_az.cos(), enc_el.sin(), enc_el.cos()], dim=-1
        )
        # Pad if dim is not multiple of 4
        if pos_enc.shape[-1] < dim:
            pos_enc = torch.nn.functional.pad(
                pos_enc, (0, dim - pos_enc.shape[-1])
            )

        # Cache and return (1, N_patches, D)
        final_enc = pos_enc.unsqueeze(0)
        self._pos_enc_cache = final_enc
        self._cache_key = cache_key
        return final_enc

    def forward(self, x: Tensor) -> Tensor:
        """Encode spherical observation via ViT.

        Args:
            x: (B, 3, Az, El) float tensor.

        Returns:
            z_t: (B, embedding_dim) spatial embedding.
        """
        batch, channels, az, el = x.shape
        p = self.patch_size

        # Ensure Az/El are multiples of p (pad if necessary)
        pad_az = (p - az % p) % p
        pad_el = (p - el % p) % p
        if pad_az > 0 or pad_el > 0:
            x = torch.nn.functional.pad(x, (0, pad_el, 0, pad_az))
            az, el = x.shape[2:]

        n_az = az // p
        n_el = el // p
        n_patches = n_az * n_el

        # Project directly into patch tokens with a strided convolution.
        h = self.patch_proj(x).flatten(2).transpose(1, 2)

        # Add [CLS] token and FIXED positional embeddings
        pos_enc = self._get_fixed_pos_enc(n_az, n_el, self.hidden_dim, x.device)
        h = h + pos_enc

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # Transformer layers
        features = self.transformer(h)

        # Extract [CLS] token output
        z_cls = features[:, 0]

        # Final projection to embedding_dim
        z_t: Tensor = self.fc(z_cls)
        return z_t
