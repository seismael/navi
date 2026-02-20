"""Tests for CognitiveMambaPolicy."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from navi_actor.cognitive_policy import CognitiveMambaPolicy


def _make_policy() -> CognitiveMambaPolicy:
    return CognitiveMambaPolicy(
        embedding_dim=128,
        azimuth_bins=64,
        elevation_bins=32,
    )


def test_forward_shapes() -> None:
    """forward() should return (B,4), (B,), (B,), hidden."""
    policy = _make_policy()
    obs = torch.randn(2, 2, 64, 32)
    actions, log_probs, values, hidden = policy.forward(obs)
    assert actions.shape == (2, 4)
    assert log_probs.shape == (2,)
    assert values.shape == (2,)
    # With temporal core active, hidden may be a Tensor (GRU) or None (Mamba2)


def test_evaluate_shapes() -> None:
    """evaluate() should return log_probs, values, entropy, hidden."""
    policy = _make_policy()
    obs = torch.randn(2, 2, 64, 32)
    acts = torch.randn(2, 4)
    lp, val, ent, hidden = policy.evaluate(obs, acts)
    assert lp.shape == (2,)
    assert val.shape == (2,)
    assert ent.dim() == 0


def test_act_returns_list() -> None:
    """act() inference should return list of floats."""
    policy = _make_policy()
    obs = torch.randn(2, 64, 32)  # (C, Az, El) — no batch dim
    action_list, hidden = policy.act(obs, step_id=0)
    assert isinstance(action_list, list)
    assert len(action_list) == 4
    assert all(isinstance(x, float) for x in action_list)


def test_encode_returns_embedding() -> None:
    """encode() should return spatial embedding without temporal processing."""
    policy = _make_policy()
    obs = torch.randn(2, 2, 64, 32)
    z = policy.encode(obs)
    assert z.shape == (2, 128)


def test_evaluate_sequence_shapes() -> None:
    """evaluate_sequence() should handle (B,T,...) observation sequences."""
    policy = _make_policy()
    obs_seq = torch.randn(2, 4, 2, 64, 32)  # (B, T, C, Az, El)
    acts_seq = torch.randn(2, 4, 4)  # (B, T, 4)
    lp, val, ent, hidden = policy.evaluate_sequence(obs_seq, acts_seq)
    assert lp.shape == (8,)  # B*T
    assert val.shape == (8,)
    assert ent.dim() == 0


def test_checkpoint_roundtrip() -> None:
    """Save and reload should produce identical state_dict."""
    policy = _make_policy()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "test.pt"
        policy.save_checkpoint(ckpt)
        loaded = CognitiveMambaPolicy.load_checkpoint(
            ckpt, embedding_dim=128, azimuth_bins=64, elevation_bins=32,
        )
    for key in policy.state_dict():
        assert torch.equal(policy.state_dict()[key], loaded.state_dict()[key])


def test_gradient_flow() -> None:
    """Gradients should propagate through the full policy."""
    policy = _make_policy()
    obs = torch.randn(2, 2, 64, 32)
    actions, log_probs, values, _ = policy.forward(obs)
    loss = log_probs.sum() + values.sum()
    loss.backward()
    # Check at least some parameters have gradients
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0
