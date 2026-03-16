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
    assert result.existential_tax == -0.02


def test_collision_penalty_on_done() -> None:
    """Collision penalty should apply only when done=True."""
    shaper = _make_shaper(collision_penalty=-10.0)
    not_done = shaper.shape(raw_reward=0.0, done=False)
    assert not_done.collision_penalty == 0.0

    is_done = shaper.shape(raw_reward=0.0, done=True)
    assert is_done.collision_penalty == -10.0


def test_velocity_heuristic() -> None:
    """Forward velocity should yield positive bonus, turning should not be punished."""
    shaper = _make_shaper(velocity_weight=0.1)
    # Forward motion
    fwd = shaper.shape(raw_reward=0.0, done=False, forward_velocity=1.0)
    assert fwd.velocity_bonus > 0.0

    # Pure turning should be neutral so env-side inspection reward can dominate
    spin = shaper.shape(
        raw_reward=0.0, done=False, forward_velocity=0.0, angular_velocity=2.0,
    )
    assert spin.velocity_bonus == 0.0


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


def test_shape_batch_matches_disabled_compile_path() -> None:
    """Disabling torch.compile should preserve batch reward semantics."""
    shaper = _make_shaper(
        collision_penalty=-3.0,
        existential_tax=-0.02,
        velocity_weight=0.25,
        intrinsic_coeff_init=0.5,
        loop_penalty_coeff=0.75,
        loop_threshold=0.8,
        torch_compile=False,
    )
    totals = shaper.shape_batch(
        raw_rewards=torch.tensor([1.0, 2.0]),
        dones=torch.tensor([False, True]),
        forward_velocities=torch.tensor([2.0, -1.0]),
        angular_velocities=torch.tensor([0.0, 3.0]),
        intrinsic_rewards=torch.tensor([0.4, 0.2]),
        loop_similarities=torch.tensor([0.7, 0.95]),
    )
    expected = torch.tensor([
        1.0 - 0.02 + (0.25 * 2.0) + (0.5 * 0.4),
        2.0 - 3.0 - 0.02 + 0.0 + (0.5 * 0.2) - (0.75 * (0.95 - 0.8)),
    ])
    assert torch.allclose(totals, expected)


def test_compile_status_reports_disabled_request() -> None:
    """The shaper should expose requested vs active compile state for attribution."""
    shaper = _make_shaper(torch_compile=False)
    assert shaper.torch_compile_requested is False
    assert shaper.torch_compile_enabled is False


def test_step_batch_matches_repeated_single_steps() -> None:
    """Batch step advancement should match repeated scalar step() calls."""
    shaper_a = _make_shaper(
        intrinsic_coeff_init=1.0,
        intrinsic_coeff_final=0.0,
        intrinsic_anneal_steps=100,
    )
    shaper_b = _make_shaper(
        intrinsic_coeff_init=1.0,
        intrinsic_coeff_final=0.0,
        intrinsic_anneal_steps=100,
    )

    for _ in range(7):
        shaper_a.step()
    shaper_b.step_batch(7)

    assert abs(shaper_a.beta - shaper_b.beta) < 1e-6


def test_default_collision_penalty_is_zero() -> None:
    """Default collision_penalty should be 0.0 (backend supplies its own).

    The backend already includes a collision penalty in the raw extrinsic
    reward. The RewardShaper should not add a second penalty by default.
    """
    shaper = _make_shaper()  # default args
    result = shaper.shape(raw_reward=-1.0, done=True)
    # collision_penalty component should be 0 by default
    assert result.collision_penalty == 0.0
    # total should only include extrinsic + tax (no extra penalty)
    assert abs(result.total - (-1.0 + (-0.02))) < 1e-6
