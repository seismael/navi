"""Tests for SphericalCNNEncoder and circular azimuth padding."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from navi_actor.perception import (  # noqa: E402
    CircularAzimuthConv2d,
    DepthwiseSeparableConv,
    SphericalCNNEncoder,
)


def test_output_shape_default() -> None:
    """Encoder should produce (B, 128) from (B, 3, 256, 48)."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128)
    x = torch.randn(2, 3, 256, 48)
    z = enc(x)
    assert z.shape == (2, 128)


def test_output_shape_custom_dim() -> None:
    """Encoder should respect custom d_model."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=64)
    x = torch.randn(1, 3, 256, 48)
    z = enc(x)
    assert z.shape == (1, 64)


def test_output_shape_varying_resolution() -> None:
    """Encoder should handle different spatial dimensions."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128)
    for az, el in [(128, 24), (256, 64), (64, 16)]:
        x = torch.randn(1, 3, az, el)
        z = enc(x)
        assert z.shape == (1, 128), f"Failed for ({az}, {el}): got {z.shape}"


def test_gradient_flow() -> None:
    """Gradients should flow through all layers."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128)
    x = torch.randn(2, 3, 256, 48, requires_grad=True)
    z = enc(x)
    loss = z.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Verify gradients reach the stem
    assert enc.stem[0].conv.weight.grad is not None
    # Verify gradients reach the proj
    assert enc.proj[0].weight.grad is not None


def test_circular_azimuth_continuity() -> None:
    """Azimuth rays 0 and 255 should produce near-identical features for uniform depth."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128).eval()
    x = torch.zeros(2, 3, 256, 48)
    x[..., 0, :] = 0.5    # mark azimuth 0
    x[..., -1, :] = 0.5   # mark azimuth 255 (adjacent on sphere)
    with torch.no_grad():
        z = enc(x)
    # Both observations should produce similar features
    cos_sim = F.cosine_similarity(z[0:1], z[1:2])
    assert cos_sim > 0.90, f"Circular continuity broken: cos_sim={cos_sim:.4f}"


def test_elevation_does_not_wrap() -> None:
    """Top and bottom elevation rows should produce different features."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128)
    enc.train()  # train mode so BatchNorm uses per-batch stats
    x_top = torch.zeros(1, 3, 256, 48)
    x_top[0, 0, :, 0:24] = 1.0   # top half bright in depth channel
    x_top[0, 0, :, 24:48] = 0.0  # bottom half dark
    x_bot = torch.zeros(1, 3, 256, 48)
    x_bot[0, 0, :, 0:24] = 0.0   # top half dark
    x_bot[0, 0, :, 24:48] = 1.0  # bottom half bright
    z_top = enc(x_top)
    z_bot = enc(x_bot)
    mse = F.mse_loss(z_top, z_bot)
    # Top/bottom should be distinguishable (non-zero MSE)
    assert mse > 1e-6, f"Elevation pooling too aggressive: mse={mse:.6f}"


def test_semantic_channel_matters() -> None:
    """Changing the semantic channel should change encoder output."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128).eval()
    x1 = torch.randn(1, 3, 256, 48)
    x2 = x1.clone()
    x2[:, 1] = 0.5  # modify semantic channel
    with torch.no_grad():
        z1 = enc(x1)
        z2 = enc(x2)
    assert not torch.allclose(z1, z2, rtol=1e-2, atol=1e-3)


def test_valid_channel_matters() -> None:
    """Changing the valid-mask channel should change encoder output."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128).eval()
    x1 = torch.randn(1, 3, 256, 48)
    x2 = x1.clone()
    x2[:, 2] = 0.0  # set valid to all-false
    with torch.no_grad():
        z1 = enc(x1)
        z2 = enc(x2)
    assert not torch.allclose(z1, z2, rtol=1e-2, atol=1e-3)


def test_parameter_count() -> None:
    """Parameter count should be in a reasonable range (< 350K)."""
    enc = SphericalCNNEncoder(in_channels=3, d_model=128)
    n_params = sum(p.numel() for p in enc.parameters())
    assert 100_000 < n_params < 350_000, f"Unexpected param count: {n_params}"


def test_circular_padding_boundary() -> None:
    """CircularAzimuthConv2d should wrap azimuth correctly."""
    conv = CircularAzimuthConv2d(1, 1, 3, stride=1, bias=False)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0 / 9.0)  # average kernel

    # Create a tensor with a boundary discontinuity
    x = torch.zeros(1, 1, 4, 8)
    x[0, 0, :, 0] = 1.0   # left edge (azimuth 0)
    x[0, 0, :, -1] = 1.0  # right edge (azimuth 7)

    with torch.no_grad():
        y = conv(x)

    # With circular padding, the interior near the right edge should see
    # the left edge values through the padding
    # Column 6 (near right): kernel center at (row 1-2, col 6)
    # should see: col 5,6,7 padded — where padded col 8 = col 0 (circular)
    interior_val = y[0, 0, 1, 6]
    # Without circular padding: sees zeros at col 8 → low value
    # With circular padding: sees 1.0 at col 8 (wrapped from col 0) → higher value
    assert interior_val > 0.1, (
        f"Circular padding not effective at azimuth boundary: {interior_val:.4f}"
    )


def test_depthwise_separable_conv_shape() -> None:
    """Depthwise separable conv should produce correct output shape."""
    from navi_actor.perception import DepthwiseSeparableConv

    conv = DepthwiseSeparableConv(16, 32, stride=2)
    x = torch.randn(2, 16, 24, 64)
    y = conv(x)
    assert y.shape == (2, 32, 12, 32)
