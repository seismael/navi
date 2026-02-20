"""Tests for FoveatedEncoder CNN."""

from __future__ import annotations

import torch

from navi_actor.perception import FoveatedEncoder


def test_output_shape_default() -> None:
    """Encoder should produce (B, 128) from (B, 2, Az, El)."""
    enc = FoveatedEncoder(embedding_dim=128)
    x = torch.randn(2, 2, 64, 32)
    z = enc(x)
    assert z.shape == (2, 128)


def test_output_shape_custom_dim() -> None:
    """Encoder should respect custom embedding_dim."""
    enc = FoveatedEncoder(embedding_dim=64)
    x = torch.randn(1, 2, 32, 16)
    z = enc(x)
    assert z.shape == (1, 64)


def test_different_spatial_sizes() -> None:
    """Encoder (adaptive pool) should handle varying spatial dims."""
    enc = FoveatedEncoder(embedding_dim=128)
    for az, el in [(64, 32), (128, 64), (16, 8)]:
        x = torch.randn(1, 2, az, el)
        z = enc(x)
        assert z.shape == (1, 128), f"Failed for ({az}, {el})"


def test_gradient_flow() -> None:
    """Gradients should flow through the encoder."""
    enc = FoveatedEncoder(embedding_dim=128)
    x = torch.randn(2, 2, 64, 32, requires_grad=True)
    z = enc(x)
    loss = z.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
