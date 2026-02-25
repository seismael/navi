"""Tests for full training-state checkpoint (knowledge accumulation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from navi_actor.config import ActorConfig
from navi_actor.training.ppo_trainer import PpoTrainer


def _make_trainer() -> PpoTrainer:
    """Create a PpoTrainer with minimal config (no ZMQ sockets needed)."""
    cfg = ActorConfig(
        sub_address="tcp://127.0.0.1:19000",
        pub_address="tcp://127.0.0.1:19001",
        step_endpoint="tcp://127.0.0.1:19002",
        mode="step",
        embedding_dim=64,
    )
    return PpoTrainer(cfg)


def test_save_load_training_state_roundtrip() -> None:
    """Full training state should survive a save/load cycle."""
    trainer = _make_trainer()

    # Mutate state to verify it persists
    trainer._reward_shaper._global_step = 12345

    # Force optimizer creation by doing a dummy forward/backward
    obs = torch.randn(1, 3, 128, 24, device=trainer._device)
    _actions, log_probs, values, _, _ = trainer._policy(obs)
    loss = -log_probs.sum() + values.sum()
    opt = trainer._learner._get_optimizer(trainer._policy)
    opt.zero_grad()
    loss.backward()
    opt.step()

    # RND predictor step
    z = trainer._policy.encode(obs.detach())
    rnd_opt = trainer._learner._get_rnd_optimizer(trainer._rnd)
    rnd_loss = trainer._rnd.distillation_loss(z)
    rnd_opt.zero_grad()
    rnd_loss.backward()
    rnd_opt.step()

    # Snapshot state before save
    policy_sd = {k: v.clone() for k, v in trainer._policy.state_dict().items()}
    rnd_sd = {k: v.clone() for k, v in trainer._rnd.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "full_state.pt"
        trainer.save_training_state(ckpt)

        # Create a fresh trainer and load
        trainer2 = _make_trainer()
        assert trainer2._reward_shaper._global_step == 0  # fresh
        trainer2.load_training_state(ckpt)

    # Verify policy weights match
    for key in policy_sd:
        assert torch.equal(
            policy_sd[key], trainer2._policy.state_dict()[key]
        ), f"Policy param mismatch: {key}"

    # Verify RND weights match (including frozen target)
    for key in rnd_sd:
        assert torch.equal(
            rnd_sd[key], trainer2._rnd.state_dict()[key]
        ), f"RND param mismatch: {key}"

    # Verify reward shaper step
    assert trainer2._reward_shaper._global_step == 12345

    # Verify optimizers were restored
    assert trainer2._learner._optimizer is not None
    assert trainer2._learner._rnd_optimizer is not None


def test_legacy_checkpoint_compat() -> None:
    """Loading a legacy checkpoint (plain state_dict) should still work."""
    trainer = _make_trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "legacy.pt"
        # Save old-format: just the policy state_dict
        torch.save(trainer._policy.state_dict(), ckpt)

        trainer2 = _make_trainer()
        trainer2.load_training_state(ckpt)

    # Policy weights should match
    for key in trainer._policy.state_dict():
        assert torch.equal(
            trainer._policy.state_dict()[key],
            trainer2._policy.state_dict()[key],
        )

    # RND, reward shaper, optimizers stay at defaults
    assert trainer2._reward_shaper._global_step == 0
    assert trainer2._learner._optimizer is None


def test_beta_annealing_continues() -> None:
    """Beta should continue from the saved global_step, not restart."""
    trainer = _make_trainer()
    # Advance 1000 steps
    for _ in range(1000):
        trainer._reward_shaper.step()
    beta_at_1000 = trainer._reward_shaper.beta

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "state.pt"
        trainer.save_training_state(ckpt)

        trainer2 = _make_trainer()
        trainer2.load_training_state(ckpt)

    assert trainer2._reward_shaper.beta == beta_at_1000
    assert trainer2._reward_shaper._global_step == 1000


def test_rnd_target_network_preserved() -> None:
    """RND target network (frozen random) must be identical after reload.

    This is critical: if the target randomises, curiosity signals become
    meaningless across scenes.
    """
    trainer = _make_trainer()
    z = torch.randn(1, 64)

    # Get raw RND outputs (target + predictor) — no running-stat mutation
    with torch.no_grad():
        target_before, pred_before = trainer._rnd(z)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "state.pt"
        trainer.save_training_state(ckpt)

        trainer2 = _make_trainer()
        trainer2.load_training_state(ckpt)

    with torch.no_grad():
        target_loaded, pred_loaded = trainer2._rnd(z)

    assert torch.allclose(target_before, target_loaded, atol=1e-6), (
        "RND target network mismatch after reload"
    )
    assert torch.allclose(pred_before, pred_loaded, atol=1e-6), (
        "RND predictor network mismatch after reload"
    )
