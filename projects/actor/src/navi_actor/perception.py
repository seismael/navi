from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__: list[str] = ["CircularAzimuthConv2d", "DepthwiseSeparableConv", "RayViTEncoder", "SphericalCNNEncoder"]


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

    def _get_fixed_pos_enc(self, n_az: int, n_el: int, dim: int, device: torch.device) -> Tensor:
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

        pos_enc = torch.cat([enc_az.sin(), enc_az.cos(), enc_el.sin(), enc_el.cos()], dim=-1)
        # Pad if dim is not multiple of 4
        if pos_enc.shape[-1] < dim:
            pos_enc = torch.nn.functional.pad(pos_enc, (0, dim - pos_enc.shape[-1]))

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
        batch, _channels, az, el = x.shape
        p = self.patch_size

        # Ensure Az/El are multiples of p (pad if necessary)
        pad_az = (p - az % p) % p
        pad_el = (p - el % p) % p
        if pad_az > 0 or pad_el > 0:
            x = torch.nn.functional.pad(x, (0, pad_el, 0, pad_az))
            az, el = x.shape[2:]

        n_az = az // p
        n_el = el // p

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


class CircularAzimuthConv2d(nn.Module):  # type: ignore[misc]
    """Conv2d with circular padding on the azimuth (width) dimension only.

    PyTorch's native ``padding_mode='circular'`` wraps **both** spatial
    dimensions, which is incorrect for spherical distance matrices where
    elevation (height) must NOT wrap — sky is not adjacent to ground.
    This module applies circular wrap to width and zero-padding to
    height before a standard ``Conv2d(padding=0)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.pad_h: int = kernel_size[0] // 2  # elevation (height)
        self.pad_w: int = kernel_size[1] // 2  # azimuth (width)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply azimuth-circular + elevation-zero padding then convolve.

        Args:
            x: ``(B, C, H, W)`` where H=elevation, W=azimuth.

        Returns:
            ``(B, C_out, H_out, W_out)``.
        """
        # Step 1: circular-pad azimuth (width) only
        if self.pad_w > 0:
            x = F.pad(x, (self.pad_w, self.pad_w, 0, 0), mode="circular")
        # Step 2: zero-pad elevation (height) only
        if self.pad_h > 0:
            x = F.pad(x, (0, 0, self.pad_h, self.pad_h), mode="constant", value=0)
        return self.conv(x)


class DepthwiseSeparableConv(nn.Module):  # type: ignore[misc]
    """Depthwise separable convolution with circular azimuth wrapping.

    Splits a standard conv into a depthwise spatial filter followed by
    a pointwise (1×1) channel mixer, dramatically reducing FLOPs while
    preserving circular azimuth topology on the spatial filter.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int | tuple[int, int]) -> None:
        super().__init__()
        self.depthwise = CircularAzimuthConv2d(
            in_channels, in_channels, 3, stride=stride, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SphericalCNNEncoder(nn.Module):  # type: ignore[misc]
    """CNN encoder for spherical distance matrices with circular azimuth topology.

    Replaces the quadratic self-attention of ``RayViTEncoder`` with
    linear convolutions on the ``(Azimuth, Elevation)`` spherical grid
    treated as a 2D image.  Azimuth wraps circularly (ray 0 is adjacent
    to ray 255); elevation is zero-padded at the poles.

    .. code-block:: text

       Input:  (B, 3, Az=256, El=48)   depth, semantic, valid channels
       Output: (B, d_model=128)         spatial embedding for temporal core

    **FLOPs:** ~29M (vs RayViT's ~150M — 5.1× reduction).
    **Params:** ~225K (vs RayViT's ~306K — 1.36× reduction).
    """

    def __init__(self, in_channels: int = 3, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model

        # Stem: initial processing — halve azimuth via stride=(1,2)
        self.stem = nn.Sequential(
            CircularAzimuthConv2d(in_channels, 32, 3, stride=(1, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # → (B, 32, 48, 128)

        # Block 1: depthwise separable — spatial reduction
        self.block1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # → (B, 64, 24, 64)

        # Block 2: depthwise separable — spatial reduction
        self.block2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # → (B, 128, 12, 32)

        # Block 3: standard conv refinement — no spatial reduction
        self.block3 = nn.Sequential(
            CircularAzimuthConv2d(128, 128, 3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # → (B, 128, 12, 32)

        # Head: global average pooling → small projection
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(128, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Encode spherical observation via CNN with circular azimuth topology.

        Args:
            x: ``(B, 3, Az, El)`` float tensor.

        Returns:
            z_t: ``(B, d_model)`` spatial embedding.
        """
        # Transpose to (B, 3, H=El, W=Az) for Conv2d processing
        x = x.transpose(2, 3)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)
