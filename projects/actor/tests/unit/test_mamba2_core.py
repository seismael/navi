"""Tests for the pure-PyTorch Mamba2 SSD temporal core."""

from __future__ import annotations

import torch

from navi_actor.mamba2_core import Mamba2SSDTemporalCore


def _make_core(d_model: int = 128) -> Mamba2SSDTemporalCore:
    return Mamba2SSDTemporalCore(d_model=d_model)


def test_forward_sequence_shapes() -> None:
    """forward() with a sequence should return (B, T, D) output."""
    core = _make_core()
    z_seq = torch.randn(2, 8, 128)
    out, hidden = core.forward(z_seq)
    assert out.shape == (2, 8, 128)
    assert hidden is None


def test_forward_step_shapes() -> None:
    """forward_step() should return (B, D) output."""
    core = _make_core()
    z_t = torch.randn(2, 128)
    out, hidden = core.forward_step(z_t)
    assert out.shape == (2, 128)
    assert hidden is None


def test_forward_with_aux_tensor() -> None:
    """forward() should accept auxiliary (velocity/heading) input."""
    core = _make_core()
    z_seq = torch.randn(2, 16, 128)
    aux = torch.randn(2, 16, 3)
    out, _ = core.forward(z_seq, aux_tensor=aux)
    assert out.shape == (2, 16, 128)


def test_forward_step_with_aux_tensor() -> None:
    """forward_step() should accept auxiliary input."""
    core = _make_core()
    z_t = torch.randn(2, 128)
    aux_t = torch.randn(2, 3)
    out, _ = core.forward_step(z_t, aux_tensor=aux_t)
    assert out.shape == (2, 128)


def test_non_chunk_aligned_sequence() -> None:
    """Sequences not divisible by chunk_size should be padded internally."""
    core = Mamba2SSDTemporalCore(d_model=128, chunk_size=64)
    z_seq = torch.randn(2, 37, 128)  # 37 not divisible by 64
    out, _ = core.forward(z_seq)
    assert out.shape == (2, 37, 128)


def test_gradient_flow() -> None:
    """Gradients should flow through the temporal core."""
    core = _make_core()
    z_seq = torch.randn(2, 4, 128, requires_grad=True)
    out, _ = core.forward(z_seq)
    (out**2).sum().backward()
    assert z_seq.grad is not None
    assert z_seq.grad.abs().sum() > 0


def test_different_d_model() -> None:
    """Core should work with non-default d_model."""
    core = Mamba2SSDTemporalCore(d_model=64, headdim=32)
    z_t = torch.randn(1, 64)
    out, _ = core.forward_step(z_t)
    assert out.shape == (1, 64)


def test_residual_connection() -> None:
    """Output should not be identical to input due to transforms."""
    core = _make_core()
    z_t = torch.randn(1, 128)
    out, _ = core.forward_step(z_t)
    assert not torch.equal(out, z_t)


def test_deterministic_given_same_input() -> None:
    """Same input should produce same output in eval mode."""
    core = _make_core()
    core.eval()
    z_seq = torch.randn(1, 16, 128)
    with torch.no_grad():
        out1, _ = core(z_seq)
        out2, _ = core(z_seq)
    assert torch.allclose(out1, out2, atol=1e-6)
