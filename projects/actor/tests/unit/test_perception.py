"""Tests for RayViTEncoder."""

from __future__ import annotations

import torch

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
