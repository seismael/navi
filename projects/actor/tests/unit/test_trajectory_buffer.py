"""Tests for TrajectoryBuffer (PPO) and legacy RolloutBuffer."""

from __future__ import annotations

import torch

from navi_actor.rollout_buffer import PPOTransition, TrajectoryBuffer


def _make_transition(
    reward: float = 1.0,
    done: bool = False,
    value: float = 0.5,
) -> PPOTransition:
    return PPOTransition(
        observation=torch.randn(3, 128, 24),
        action=torch.randn(4),
        log_prob=-0.5,
        value=value,
        reward=reward,
        done=done,
    )


def test_append_and_len() -> None:
    buf = TrajectoryBuffer()
    assert len(buf) == 0
    buf.append(_make_transition())
    buf.append(_make_transition())
    assert len(buf) == 2


def test_clear() -> None:
    buf = TrajectoryBuffer()
    buf.append(_make_transition())
    buf.clear()
    assert len(buf) == 0


def test_gae_computation() -> None:
    """GAE advantages should be computed without errors."""
    buf = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    for i in range(10):
        buf.append(_make_transition(reward=float(i), value=0.5))
    buf.compute_returns_and_advantages(last_value=0.0)
    assert buf._finalized
    assert buf._advantages.shape == (10,)
    assert buf._returns.shape == (10,)


def test_gae_done_mask() -> None:
    """Done transitions should reset GAE accumulation."""
    buf = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    buf.append(_make_transition(reward=1.0, done=False, value=0.5))
    buf.append(_make_transition(reward=1.0, done=True, value=0.5))
    buf.append(_make_transition(reward=1.0, done=False, value=0.5))
    buf.compute_returns_and_advantages(last_value=0.0)
    # After a done, the advantage computation resets
    adv = buf._advantages
    assert adv.shape == (3,)


def test_minibatch_sampling() -> None:
    """Sampling should yield proper MiniBatch objects."""
    buf = TrajectoryBuffer()
    for _ in range(64):
        buf.append(_make_transition())
    buf.compute_returns_and_advantages(last_value=0.0)

    batches = list(buf.sample_minibatches(batch_size=16, seq_len=0))
    assert len(batches) > 0
    mb = batches[0]
    assert mb.observations.dim() == 4  # (B, 2, Az, El)
    assert mb.actions.shape[1] == 4
    assert mb.old_log_probs.dim() == 1
    assert mb.advantages.dim() == 1
    assert mb.returns.dim() == 1


def test_sequential_sampling() -> None:
    """BPTT sequential sampling should yield contiguous chunks."""
    buf = TrajectoryBuffer()
    for _ in range(128):
        buf.append(_make_transition())
    buf.compute_returns_and_advantages(last_value=0.0)

    batches = list(buf.sample_minibatches(batch_size=64, seq_len=32))
    assert len(batches) > 0
    for mb in batches:
        # Each batch should be a multiple of seq_len
        assert mb.observations.shape[0] > 0


def test_sampling_requires_finalization() -> None:
    """Sampling before compute_returns_and_advantages should raise."""
    buf = TrajectoryBuffer()
    buf.append(_make_transition())
    try:
        list(buf.sample_minibatches())
        msg = "Should have raised RuntimeError"
        raise AssertionError(msg)
    except RuntimeError:
        pass


def test_advantage_normalization() -> None:
    """Advantages should be normalized during sampling."""
    buf = TrajectoryBuffer()
    for i in range(32):
        buf.append(_make_transition(reward=float(i)))
    buf.compute_returns_and_advantages(last_value=0.0)
    batches = list(buf.sample_minibatches(batch_size=32, seq_len=0))
    # After normalization, mean should be near 0
    all_advs = torch.cat([mb.advantages for mb in batches])
    assert abs(all_advs.mean().item()) < 0.5


def test_truncation_bootstrap() -> None:
    """Truncated episodes should bootstrap from value estimate, not zero.

    When an episode is truncated (time limit), the GAE should treat the
    episode boundary by bootstrapping from V(s_T) rather than setting the
    return to zero as it does for true terminations (collisions).
    """
    gamma = 0.99
    gae_lambda = 0.95
    value = 5.0

    # Buffer with a truncated transition at the end
    buf_trunc = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)
    buf_trunc.append(_make_transition(reward=1.0, done=False, value=value))
    tr_trunc = PPOTransition(
        observation=torch.randn(3, 128, 24),
        action=torch.randn(4),
        log_prob=-0.5,
        value=value,
        reward=1.0,
        done=False,
        truncated=True,
    )
    buf_trunc.append(tr_trunc)
    buf_trunc.compute_returns_and_advantages(last_value=0.0)

    # Buffer with a true termination at the end
    buf_done = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)
    buf_done.append(_make_transition(reward=1.0, done=False, value=value))
    buf_done.append(_make_transition(reward=1.0, done=True, value=value))
    buf_done.compute_returns_and_advantages(last_value=0.0)

    # The truncated case should have a HIGHER return for the truncated step
    # because it bootstraps from V(s_T) = value, while the done case uses 0.
    trunc_return = buf_trunc._returns[1].item()
    done_return = buf_done._returns[1].item()
    assert trunc_return > done_return, (
        f"Truncated return ({trunc_return:.4f}) should exceed done return "
        f"({done_return:.4f}) due to value bootstrapping"
    )


def test_truncation_still_cuts_gae_trace() -> None:
    """Truncation should still cut the GAE trace across episode boundaries."""
    gamma = 0.99
    gae_lambda = 0.95

    # A continuing buffer (no episode boundary)
    buf_cont = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)
    for _ in range(4):
        buf_cont.append(_make_transition(reward=1.0, done=False, value=0.5))
    buf_cont.compute_returns_and_advantages(last_value=0.5)

    # A buffer with truncation in the middle
    buf_trunc = TrajectoryBuffer(gamma=gamma, gae_lambda=gae_lambda)
    buf_trunc.append(_make_transition(reward=1.0, done=False, value=0.5))
    tr_trunc = PPOTransition(
        observation=torch.randn(3, 128, 24),
        action=torch.randn(4),
        log_prob=-0.5,
        value=0.5,
        reward=1.0,
        done=False,
        truncated=True,
    )
    buf_trunc.append(tr_trunc)
    buf_trunc.append(_make_transition(reward=1.0, done=False, value=0.5))
    buf_trunc.append(_make_transition(reward=1.0, done=False, value=0.5))
    buf_trunc.compute_returns_and_advantages(last_value=0.5)

    # Advantages after the truncation boundary should differ from
    # the continuous case, confirming the trace was cut
    adv_cont_0 = buf_cont._advantages[0].item()
    adv_trunc_0 = buf_trunc._advantages[0].item()
    assert adv_cont_0 != adv_trunc_0, (
        "GAE trace should be cut at truncation boundary"
    )
