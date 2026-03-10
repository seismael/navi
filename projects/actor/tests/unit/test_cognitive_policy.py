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
    """forward() should return (B,4), (B,), (B,), hidden, (B,D)."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    actions, log_probs, values, _, z_t = policy.forward(obs)
    assert actions.shape == (2, 4)
    assert log_probs.shape == (2,)
    assert values.shape == (2,)
    assert z_t.shape == (2, 128)
    # With temporal core active, hidden may be a Tensor (GRU) or None (Mamba2)


def test_evaluate_shapes() -> None:
    """evaluate() should return log_probs, values, entropy, hidden, z_t."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    acts = torch.randn(2, 4)
    lp, val, ent, _, z_t = policy.evaluate(obs, acts)
    assert lp.shape == (2,)
    assert val.shape == (2,)
    assert ent.dim() == 0
    assert z_t.shape == (2, 128)


def test_act_returns_list() -> None:
    """act() inference should return list of floats."""
    policy = _make_policy()
    obs = torch.randn(3, 128, 24)  # (C, Az, El) — no batch dim
    action_list, _ = policy.act(obs, step_id=0)
    assert isinstance(action_list, list)
    assert len(action_list) == 4
    assert all(isinstance(x, float) for x in action_list)


def test_encode_returns_embedding() -> None:
    """encode() should return spatial embedding without temporal processing."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    z = policy.encode(obs)
    assert z.shape == (2, 128)


def test_evaluate_sequence_shapes() -> None:
    """evaluate_sequence() should handle (B,T,...) observation sequences."""
    policy = _make_policy()
    obs_seq = torch.randn(2, 4, 3, 128, 24)  # (B, T, C, Az, El)
    acts_seq = torch.randn(2, 4, 4)  # (B, T, 4)
    lp, val, ent, _, z_t = policy.evaluate_sequence(obs_seq, acts_seq)
    assert lp.shape == (8,)  # B*T
    assert val.shape == (8,)
    assert ent.dim() == 0
    assert z_t.shape == (8, 128)


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
    obs = torch.randn(2, 3, 128, 24)
    _, log_probs, values, _, _ = policy.forward(obs)
    loss = log_probs.sum() + values.sum()
    loss.backward()  # type: ignore[no-untyped-call]
    # Check at least some parameters have gradients
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_evaluate_value_stop_gradient() -> None:
    """evaluate() must stop-gradient: value loss must NOT flow to encoder/temporal."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    acts = torch.randn(2, 4)
    _, values, _, _, _ = policy.evaluate(obs, acts)

    # Backward through values only (simulating critic loss)
    policy.zero_grad()
    values.sum().backward()  # type: ignore[no-untyped-call]

    # Encoder and temporal core should have NO gradients
    for name, p in policy.encoder.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"encoder.{name} must not receive value-loss gradients"
        )
    for name, p in policy.temporal_core.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"temporal_core.{name} must not receive value-loss gradients"
        )

    # Critic head SHOULD have gradients
    critic_grads = [
        p.grad for p in policy.heads.critic.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(critic_grads) > 0, "Critic head must receive value-loss gradients"


def test_evaluate_sequence_value_stop_gradient() -> None:
    """evaluate_sequence() must stop-gradient: value loss must NOT flow to backbone."""
    policy = _make_policy()
    obs_seq = torch.randn(2, 4, 3, 128, 24)
    acts_seq = torch.randn(2, 4, 4)
    _, values, _, _, _ = policy.evaluate_sequence(obs_seq, acts_seq)

    policy.zero_grad()
    values.sum().backward()  # type: ignore[no-untyped-call]

    for name, p in policy.encoder.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"encoder.{name} must not receive value-loss gradients"
        )
    for name, p in policy.temporal_core.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"temporal_core.{name} must not receive value-loss gradients"
        )

    critic_grads = [
        p.grad for p in policy.heads.critic.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(critic_grads) > 0, "Critic head must receive value-loss gradients"


def test_evaluate_actor_gradient_flows_to_backbone() -> None:
    """evaluate() actor loss SHOULD propagate through encoder+temporal (not blocked)."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    acts = torch.randn(2, 4)
    log_probs, _, _, _, _ = policy.evaluate(obs, acts)

    policy.zero_grad()
    (-log_probs.sum()).backward()  # type: ignore[no-untyped-call]

    # Encoder and temporal core SHOULD have gradients from policy loss
    encoder_grads = [
        p.grad for p in policy.encoder.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(encoder_grads) > 0, "Encoder must receive policy-loss gradients"

    temporal_grads = [
        p.grad for p in policy.temporal_core.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(temporal_grads) > 0, "Temporal core must receive policy-loss gradients"
