"""Tests for RNDModule (Random Network Distillation)."""

from __future__ import annotations

import torch

from navi_actor.rnd import RNDModule


def _make_rnd(input_dim: int = 128) -> RNDModule:
    return RNDModule(input_dim=input_dim)


def test_forward_shapes() -> None:
    """forward() should return target and predictor outputs of shape (B, 64)."""
    rnd = _make_rnd()
    z = torch.randn(4, 128)
    target_out, pred_out = rnd.forward(z)
    assert target_out.shape == (4, 64)
    assert pred_out.shape == (4, 64)


def test_target_frozen() -> None:
    """Target network parameters should not require gradients."""
    rnd = _make_rnd()
    for p in rnd.target.parameters():
        assert not p.requires_grad


def test_predictor_trainable() -> None:
    """Predictor network parameters should require gradients."""
    rnd = _make_rnd()
    for p in rnd.predictor.parameters():
        assert p.requires_grad


def test_distillation_loss_positive() -> None:
    """Distillation loss should be non-negative."""
    rnd = _make_rnd()
    z = torch.randn(8, 128)
    loss = rnd.distillation_loss(z)
    assert loss.item() >= 0.0


def test_distillation_loss_backward() -> None:
    """Distillation loss should produce gradients for predictor only."""
    rnd = _make_rnd()
    z = torch.randn(4, 128)
    loss = rnd.distillation_loss(z)
    loss.backward()

    # Predictor should have gradients
    pred_grads = [p.grad for p in rnd.predictor.parameters() if p.grad is not None]
    assert len(pred_grads) > 0

    # Target should NOT have gradients
    target_grads = [p.grad for p in rnd.target.parameters() if p.grad is not None]
    assert len(target_grads) == 0


def test_intrinsic_reward_shape() -> None:
    """intrinsic_reward() should return (B,) shaped tensor."""
    rnd = _make_rnd()
    z = torch.randn(4, 128)
    reward = rnd.intrinsic_reward(z)
    assert reward.shape == (4,)


def test_intrinsic_reward_clamped() -> None:
    """Intrinsic reward should be clamped to [-5, 5]."""
    rnd = _make_rnd()
    z = torch.randn(16, 128)
    reward = rnd.intrinsic_reward(z)
    assert reward.min() >= -5.0
    assert reward.max() <= 5.0


def test_running_statistics_update() -> None:
    """Running mean/var should change after computing intrinsic reward."""
    rnd = _make_rnd()
    initial_mean = rnd._running_mean.clone()
    z = torch.randn(32, 128)
    rnd.intrinsic_reward(z)
    # Running mean should have been updated
    assert not torch.equal(rnd._running_mean, initial_mean)


def test_different_input_dim() -> None:
    """RND should work with non-default input dimensions."""
    rnd = RNDModule(input_dim=64, hidden_dim=64, output_dim=32)
    z = torch.randn(4, 64)
    target_out, pred_out = rnd.forward(z)
    assert target_out.shape == (4, 32)
    assert pred_out.shape == (4, 32)


def test_predictor_matches_target_architecture() -> None:
    """Predictor and target should have the same number of layers.

    Equal capacity prevents the predictor from perfectly memorizing the
    target, maintaining a residual prediction error as the novelty signal.
    """
    rnd = _make_rnd()
    target_layers = [m for m in rnd.target if isinstance(m, torch.nn.Linear)]
    pred_layers = [m for m in rnd.predictor if isinstance(m, torch.nn.Linear)]
    assert len(target_layers) == len(pred_layers), (
        f"Predictor ({len(pred_layers)} linear layers) should match "
        f"target ({len(target_layers)} linear layers)"
    )


def test_rnd_maintains_nonzero_loss_after_training() -> None:
    """RND loss should remain nonzero after moderate training.

    With equal predictor/target capacity and different random init,
    the predictor cannot perfectly replicate the target, ensuring a
    persistent novelty signal.
    """
    rnd = _make_rnd()
    optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=3e-5)

    # Train for 200 steps on random embeddings
    for _ in range(200):
        z = torch.randn(32, 128)
        loss = rnd.distillation_loss(z)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Loss should still be measurably nonzero
    z_test = torch.randn(64, 128)
    final_loss = rnd.distillation_loss(z_test).item()
    assert final_loss > 0.001, (
        f"RND loss collapsed to {final_loss:.6f} after 200 training steps; "
        "predictor and target should have equal capacity to prevent this"
    )
