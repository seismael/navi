from __future__ import annotations

import torch
from torch import Tensor, nn

__all__: list[str] = ["FoveatedEncoder", "RayViTEncoder"]


class FoveatedEncoder(nn.Module):  # type: ignore[misc]
    """4-layer CNN that encodes a (B, 3, Az, El) spherical image into z_t in R^D.

    Channel 0 = depth, channel 1 = semantic class id, channel 2 = valid_mask.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Encode spherical observation.

        Args:
            x: (B, 3, Az, El) float tensor.

        Returns:
            z_t: (B, embedding_dim) float tensor.

        """
        h: Tensor = self.convs(x)
        h = self.pool(h).squeeze(-1).squeeze(-1)
        z_t: Tensor = self.fc(h)
        return z_t


class RayViTEncoder(nn.Module):  # type: ignore[misc]
    """Vision Transformer encoder for spherical distance matrices.

    Implements §8.4 of ARCHITECTURE.md. Treats patches of the spherical
    grid as tokens and processes them via a Transformer Encoder.

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
        patch_size: int = 4,
        n_layers: int = 4,
        n_heads: int = 8,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        # Input projection: flatten patch + channels → hidden_dim
        # Channel 0=depth, 1=semantic, 2=valid_mask
        patch_input_dim = 3 * patch_size * patch_size
        self.patch_proj = nn.Linear(patch_input_dim, hidden_dim)

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

        # Output projection
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # Learnable positional embeddings (max 1024 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Encode spherical observation via ViT.

        Args:
            x: (B, 2, Az, El) float tensor.

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

        # Reshape into patches: (B, 2, Az/p, p, El/p, p)
        # -> (B, Az/p, El/p, 2, p, p)
        # -> (B, N_patches, 2*p*p)
        n_az = az // p
        n_el = el // p
        n_patches = n_az * n_el

        patches = x.view(batch, channels, n_az, p, n_el, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(batch, n_patches, -1)

        # Project to hidden_dim
        h = self.patch_proj(patches)

        # Add positional embeddings (truncate to actual n_patches)
        h = h + self.pos_embed[:, :n_patches, :]

        # Transformer layers
        features = self.transformer(h)

        # Global average pool across tokens
        z_pooled = features.mean(dim=1)

        # Final projection to embedding_dim
        z_t: Tensor = self.fc(z_pooled)
        return z_t
