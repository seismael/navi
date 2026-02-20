"""Tests for Mamba2TemporalCore (GRU fallback path)."""

from __future__ import annotations

import torch

from navi_actor.mamba_core import Mamba2TemporalCore


def _make_core(d_model: int = 128) -> Mamba2TemporalCore:
    return Mamba2TemporalCore(d_model=d_model)


def test_forward_sequence_shapes() -> None:
    """forward() with a sequence should return (B, T, D) output."""
    core = _make_core()
    z_seq = torch.randn(2, 8, 128)
    out, hidden = core.forward(z_seq)
    assert out.shape == (2, 8, 128)


def test_forward_step_shapes() -> None:
    """forward_step() should return (B, D) output."""
    core = _make_core()
    z_t = torch.randn(2, 128)
    out, hidden = core.forward_step(z_t)
    assert out.shape == (2, 128)


def test_residual_connection() -> None:
    """Output should not be identical to input due to LayerNorm + transform."""
    core = _make_core()
    z_t = torch.randn(1, 128)
    out, _ = core.forward_step(z_t)
    # They should be different (transformed)
    assert not torch.equal(out, z_t)


def test_hidden_state_continuity() -> None:
    """Hidden state from step 1 should be usable in step 2."""
    core = _make_core()
    z1 = torch.randn(1, 128)
    out1, h1 = core.forward_step(z1)

    z2 = torch.randn(1, 128)
    out2, h2 = core.forward_step(z2, h1)
    assert out2.shape == (1, 128)
    # Hidden state should have been updated
    if h1 is not None and h2 is not None:
        assert not torch.equal(h1, h2)


def test_gradient_flow() -> None:
    """Gradients should flow through the temporal core."""
    core = _make_core()
    z_seq = torch.randn(2, 4, 128, requires_grad=True)
    out, _ = core.forward(z_seq)
    out.sum().backward()
    assert z_seq.grad is not None
    assert z_seq.grad.abs().sum() > 0


def test_different_d_model() -> None:
    """Core should work with non-default d_model."""
    core = _make_core(d_model=64)
    z_t = torch.randn(1, 64)
    out, _ = core.forward_step(z_t)
    assert out.shape == (1, 64)
