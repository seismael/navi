"""Tests for RayViTEncoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from navi_actor.perception import RayViTEncoder


def test_output_shape_default() -> None:
    """Encoder should produce (B, 128) from (B, 3, Az, El)."""
    enc = RayViTEncoder(embedding_dim=128)
    x = torch.randn(2, 3, 128, 24)
    z = enc(x)
    assert z.shape == (2, 128)


def test_output_shape_custom_dim() -> None:
    """Encoder should respect custom embedding_dim."""
    enc = RayViTEncoder(embedding_dim=64)
    x = torch.randn(1, 3, 128, 24)
    z = enc(x)
    assert z.shape == (1, 64)


def test_different_spatial_sizes() -> None:
    """Encoder (adaptive pool) should handle varying spatial dims."""
    enc = RayViTEncoder(embedding_dim=128)
    for az, el in [(128, 24), (64, 32), (16, 8)]:
        x = torch.randn(1, 3, az, el)
        z = enc(x)
        assert z.shape == (1, 128), f"Failed for ({az}, {el})"


def test_gradient_flow() -> None:
    """Gradients should flow through the encoder."""
    enc = RayViTEncoder(embedding_dim=128)
    x = torch.randn(2, 3, 128, 24, requires_grad=True)
    z = enc(x)
    loss = z.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_patch_projection_matches_linearized_patch_projection() -> None:
    """Strided patch embedding must match the legacy patch-flattened linear projection."""
    enc = RayViTEncoder(embedding_dim=64, patch_size=8, hidden_dim=32)
    x = torch.randn(2, 3, 128, 24)

    p = enc.patch_size
    padded = x
    pad_az = (p - x.shape[2] % p) % p
    pad_el = (p - x.shape[3] % p) % p
    if pad_az > 0 or pad_el > 0:
        padded = F.pad(x, (0, pad_el, 0, pad_az))
    n_az = padded.shape[2] // p
    n_el = padded.shape[3] // p
    legacy_patches = padded.view(x.shape[0], 3, n_az, p, n_el, p)
    legacy_patches = legacy_patches.permute(0, 2, 4, 1, 3, 5).reshape(x.shape[0], n_az * n_el, -1)

    conv_tokens = enc.patch_proj(padded).flatten(2).transpose(1, 2)
    linear_tokens = F.linear(
        legacy_patches,
        enc.patch_proj.weight.reshape(enc.hidden_dim, -1),
        enc.patch_proj.bias,
    )

    torch.testing.assert_close(conv_tokens, linear_tokens, rtol=1e-4, atol=1e-5)
