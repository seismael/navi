"""Tests for ActorCriticHeads."""

from __future__ import annotations

import torch

from navi_actor.actor_critic import ActorCriticHeads


def test_forward_shapes() -> None:
    """Forward should return (B, 4) means and (B,) values."""
    heads = ActorCriticHeads(input_dim=128)
    features = torch.randn(4, 128)
    mean, value = heads(features)
    assert mean.shape == (4, 4)
    assert value.shape == (4,)


def test_action_scaling() -> None:
    """Action means should be bounded by action_scales (normalised to 1.0)."""
    heads = ActorCriticHeads(
        input_dim=128,
        max_forward=1.0,
        max_vertical=1.0,
        max_lateral=1.0,
        max_yaw=1.0,
    )
    features = torch.randn(100, 128)
    mean, _ = heads(features)
    # Tanh output * scale => |mean| <= scale
    assert mean[:, 0].abs().max() <= 1.0 + 1e-5
    assert mean[:, 1].abs().max() <= 1.0 + 1e-5
    assert mean[:, 2].abs().max() <= 1.0 + 1e-5
    assert mean[:, 3].abs().max() <= 1.0 + 1e-5


def test_log_prob_shape() -> None:
    """log_prob should return (B,) scalar per sample."""
    heads = ActorCriticHeads(input_dim=128)
    features = torch.randn(8, 128)
    actions = torch.randn(8, 4)
    lp = heads.log_prob(features, actions)
    assert lp.shape == (8,)


def test_entropy_scalar() -> None:
    """Entropy should be a scalar."""
    heads = ActorCriticHeads(input_dim=128)
    ent = heads.entropy()
    assert ent.dim() == 0


def test_sample_shapes() -> None:
    """sample() should return actions, log_probs, values."""
    heads = ActorCriticHeads(input_dim=128)
    features = torch.randn(4, 128)
    actions, log_probs, values = heads.sample(features)
    assert actions.shape == (4, 4)
    assert log_probs.shape == (4,)
    assert values.shape == (4,)


def test_gradient_flow() -> None:
    """Gradients should flow through both actor and critic."""
    heads = ActorCriticHeads(input_dim=128)
    features = torch.randn(4, 128)
    _, log_probs, values = heads.sample(features)
    # Use log_probs (touches log_std) + values to ensure all params get gradients
    loss = log_probs.sum() + values.sum()
    loss.backward()
    for p in heads.parameters():
        assert p.grad is not None
