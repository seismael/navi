"""Tests for RewardShaper (extrinsic + intrinsic + loop penalty)."""

from __future__ import annotations

import torch

from navi_actor.reward_shaping import RewardShaper, ShapedReward


def _make_shaper(**kwargs: object) -> RewardShaper:
    return RewardShaper(**kwargs)  # type: ignore[arg-type]


def test_basic_shaping() -> None:
    """Shaped reward should combine all components."""
    shaper = _make_shaper()
    result = shaper.shape(raw_reward=1.0, done=False)
    assert isinstance(result, ShapedReward)
    assert result.extrinsic == 1.0
    assert result.collision_penalty == 0.0
    assert result.existential_tax == -0.01


def test_collision_penalty_on_done() -> None:
    """Collision penalty should apply only when done=True."""
    shaper = _make_shaper(collision_penalty=-10.0)
    not_done = shaper.shape(raw_reward=0.0, done=False)
    assert not_done.collision_penalty == 0.0

    is_done = shaper.shape(raw_reward=0.0, done=True)
    assert is_done.collision_penalty == -10.0


def test_velocity_heuristic() -> None:
    """Forward velocity should yield positive bonus, spinning yields negative."""
    shaper = _make_shaper(velocity_weight=0.1)
    # Forward motion
    fwd = shaper.shape(raw_reward=0.0, done=False, forward_velocity=1.0)
    assert fwd.velocity_bonus > 0.0

    # Pure spinning
    spin = shaper.shape(
        raw_reward=0.0, done=False, forward_velocity=0.0, angular_velocity=2.0,
    )
    assert spin.velocity_bonus < 0.0


def test_intrinsic_reward_scaling() -> None:
    """Intrinsic reward should be scaled by beta."""
    shaper = _make_shaper(intrinsic_coeff_init=1.0)
    result = shaper.shape(raw_reward=0.0, done=False, intrinsic_reward=5.0)
    assert abs(result.intrinsic - 5.0) < 1e-6


def test_loop_penalty_threshold() -> None:
    """Loop penalty should fire only above threshold."""
    shaper = _make_shaper(loop_penalty_coeff=0.5, loop_threshold=0.85)
    below = shaper.shape(raw_reward=0.0, done=False, loop_similarity=0.80)
    assert below.loop_penalty == 0.0

    above = shaper.shape(raw_reward=0.0, done=False, loop_similarity=0.95)
    assert above.loop_penalty > 0.0
    expected = 0.5 * (0.95 - 0.85)
    assert abs(above.loop_penalty - expected) < 1e-6


def test_annealing_schedule() -> None:
    """Beta should decay from init to final over anneal_steps."""
    shaper = _make_shaper(
        intrinsic_coeff_init=1.0,
        intrinsic_coeff_final=0.0,
        intrinsic_anneal_steps=100,
    )
    assert abs(shaper.beta - 1.0) < 1e-6

    for _ in range(50):
        shaper.step()
    assert abs(shaper.beta - 0.5) < 1e-6

    for _ in range(50):
        shaper.step()
    assert abs(shaper.beta - 0.0) < 1e-6

    # Beyond anneal should stay at final
    for _ in range(50):
        shaper.step()
    assert abs(shaper.beta - 0.0) < 1e-6


def test_total_reward_composition() -> None:
    """Total should be sum of all components."""
    shaper = _make_shaper(
        collision_penalty=-10.0,
        existential_tax=-0.01,
        velocity_weight=0.0,
        intrinsic_coeff_init=1.0,
        loop_penalty_coeff=0.5,
        loop_threshold=0.85,
    )
    result = shaper.shape(
        raw_reward=1.0,
        done=False,
        intrinsic_reward=2.0,
        loop_similarity=0.90,
    )
    expected = 1.0 + 0.0 + (-0.01) + 0.0 + 2.0 - 0.5 * (0.90 - 0.85)
    assert abs(result.total - expected) < 1e-6


def test_shape_batch() -> None:
    """shape_batch should return (B,) tensor of shaped rewards."""
    shaper = _make_shaper()
    n = 8
    totals = shaper.shape_batch(
        raw_rewards=torch.ones(n),
        dones=torch.zeros(n),
        forward_velocities=torch.ones(n),
        angular_velocities=torch.zeros(n),
        intrinsic_rewards=torch.ones(n),
        loop_similarities=torch.zeros(n),
    )
    assert totals.shape == (n,)
    # All rewards should be positive (1.0 + tax + vel + intrinsic)
    assert (totals > 0).all()
