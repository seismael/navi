"""Tests for TrajectoryBuffer (PPO)."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from navi_actor.rollout_buffer import (
    MultiTrajectoryBuffer,
    PPOTransition,
    TrajectoryBuffer,
)


def _assert_same_device(actual: torch.device, expected: torch.device) -> None:
    """Treat CUDA default-device aliases as equivalent during device placement checks."""
    assert actual.type == expected.type
    if actual.type == "cuda":
        assert actual.index in (None, 0)
        assert expected.index in (None, 0)
    else:
        assert actual.index == expected.index


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


def test_preallocated_buffer_caches_normalized_advantages_once() -> None:
    buffer = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95, capacity=4)
    for reward in (1.0, 2.0, 3.0, 4.0):
        transition = _make_transition(reward=reward, value=0.25)
        buffer.append_fields(
            observation=transition.observation,
            action=transition.action,
            log_prob=torch.tensor(transition.log_prob, dtype=torch.float32),
            value=torch.tensor(transition.value, dtype=torch.float32),
            reward=torch.tensor(transition.reward, dtype=torch.float32),
            done=torch.tensor(transition.done, dtype=torch.bool),
            truncated=torch.tensor(transition.truncated, dtype=torch.bool),
            hidden_state=transition.hidden_state,
            aux_tensor=transition.aux_tensor,
        )

    buffer.compute_returns_and_advantages(last_value=torch.tensor(0.0))
    first_batches = list(buffer.sample_minibatches(batch_size=2, seq_len=0))
    second_batches = list(buffer.sample_minibatches(batch_size=2, seq_len=0))

    first_advs = torch.cat([mb.advantages for mb in first_batches]).sort().values
    second_advs = torch.cat([mb.advantages for mb in second_batches]).sort().values
    expected = buffer._normalized_advantages.sort().values

    assert torch.allclose(first_advs, expected)
    assert torch.allclose(second_advs, expected)


def test_multi_trajectory_sampling_normalizes_advantages_once() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    buffer.append_batch(
        observations=torch.randn(2, 3, 16, 8),
        actions=torch.randn(2, 4),
        log_probs=torch.randn(2),
        values=torch.tensor([0.1, 0.2]),
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
    )
    buffer.append_batch(
        observations=torch.randn(2, 3, 16, 8),
        actions=torch.randn(2, 4),
        log_probs=torch.randn(2),
        values=torch.tensor([0.3, 0.4]),
        rewards=torch.tensor([3.0, 4.0]),
        dones=torch.tensor([False, True], dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
    )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    flat_batches = list(buffer.sample_minibatches(batch_size=4, seq_len=0))
    seq_batches = list(buffer.sample_minibatches(batch_size=4, seq_len=2))

    normalized = cast("torch.Tensor", buffer._all_advantages_normalized)
    flat_advs = torch.cat([mb.advantages for mb in flat_batches]).sort().values
    seq_advs = torch.cat([mb.advantages for mb in seq_batches]).sort().values
    expected = normalized.flatten().sort().values

    assert torch.allclose(flat_advs, expected)
    assert torch.allclose(seq_advs, expected)


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
    buf_trunc.compute_returns_and_advantages(last_value=value)

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
    assert adv_cont_0 != adv_trunc_0, "GAE trace should be cut at truncation boundary"


def test_preallocated_tensor_storage_matches_standard_path() -> None:
    rewards = [1.0, 0.5, -0.25, 2.0]
    dones = [False, False, True, False]

    standard = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    preallocated = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95, capacity=len(rewards))

    for reward, done in zip(rewards, dones, strict=True):
        transition = _make_transition(reward=reward, done=done, value=0.25)
        standard.append(transition)
        preallocated.append_fields(
            observation=transition.observation,
            action=transition.action,
            log_prob=torch.tensor(transition.log_prob, dtype=torch.float32),
            value=torch.tensor(transition.value, dtype=torch.float32),
            reward=torch.tensor(transition.reward, dtype=torch.float32),
            done=torch.tensor(transition.done, dtype=torch.bool),
            truncated=torch.tensor(transition.truncated, dtype=torch.bool),
            hidden_state=transition.hidden_state,
            aux_tensor=transition.aux_tensor,
        )

    standard.compute_returns_and_advantages(last_value=0.0)
    preallocated.compute_returns_and_advantages(last_value=torch.tensor(0.0))

    assert torch.allclose(preallocated._advantages.cpu(), standard._advantages.cpu())
    assert torch.allclose(preallocated._returns.cpu(), standard._returns.cpu())

    standard_batches = list(standard.sample_minibatches(batch_size=4, seq_len=0))
    preallocated_batches = list(preallocated.sample_minibatches(batch_size=4, seq_len=0))
    assert len(standard_batches) == len(preallocated_batches) == 1
    assert standard_batches[0].observations.shape == preallocated_batches[0].observations.shape
    assert standard_batches[0].actions.shape == preallocated_batches[0].actions.shape


def test_multi_trajectory_append_batch_supports_sequence_sampling() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    for step in range(1, 3):
        obs = torch.randn(2, 3, 16, 8)
        actions = torch.randn(2, 4)
        log_probs = torch.randn(2)
        values = torch.full((2,), 0.5 * step)
        rewards = torch.full((2,), float(step))
        dones = torch.tensor([False, step == 2], dtype=torch.bool)
        truncateds = torch.zeros(2, dtype=torch.bool)
        aux = torch.randn(2, 3)
        buffer.append_batch(
            observations=obs,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            truncateds=truncateds,
            aux_tensors=aux,
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))
    batches = list(buffer.sample_minibatches(batch_size=4, seq_len=2))

    assert batches
    first = batches[0]
    assert first.observations.shape[0] == 4
    assert first.sequence_observations is not None
    assert first.sequence_observations.shape == (2, 2, 3, 16, 8)
    assert first.sequence_actions is not None
    assert first.sequence_actions.shape == (2, 2, 4)
    assert first.dones is None
    assert first.sequence_aux_tensors is not None
    assert first.sequence_aux_tensors.shape == (2, 2, 3)


def test_multi_trajectory_append_batch_keeps_batched_storage_on_input_device() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    buffer.append_batch(
        observations=torch.randn(2, 3, 16, 8, device=device),
        actions=torch.randn(2, 4, device=device),
        log_probs=torch.randn(2, device=device),
        values=torch.randn(2, device=device),
        rewards=torch.randn(2, device=device),
        dones=torch.zeros(2, dtype=torch.bool, device=device),
        truncateds=torch.zeros(2, dtype=torch.bool, device=device),
        aux_tensors=torch.randn(2, 3, device=device),
    )

    assert buffer._batch_obs is not None
    assert buffer._batch_actions is not None
    assert buffer._batch_log_probs is not None
    assert buffer._batch_values is not None
    assert buffer._batch_rewards is not None
    assert buffer._batch_dones is not None
    assert buffer._batch_truncateds is not None
    assert buffer._batch_aux is not None
    _assert_same_device(buffer._batch_obs.device, device)
    _assert_same_device(buffer._batch_actions.device, device)
    _assert_same_device(buffer._batch_log_probs.device, device)
    _assert_same_device(buffer._batch_values.device, device)
    _assert_same_device(buffer._batch_rewards.device, device)
    _assert_same_device(buffer._batch_dones.device, device)
    _assert_same_device(buffer._batch_truncateds.device, device)
    _assert_same_device(buffer._batch_aux.device, device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA to exercise mixed-device guard"
)
def test_multi_trajectory_append_batch_rejects_mixed_device_rollout_tensors() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    with pytest.raises(RuntimeError, match="requires all rollout tensors on one device"):
        buffer.append_batch(
            observations=torch.randn(2, 3, 16, 8, device="cuda"),
            actions=torch.randn(2, 4),
            log_probs=torch.randn(2, device="cuda"),
            values=torch.randn(2, device="cuda"),
            rewards=torch.randn(2, device="cuda"),
            dones=torch.zeros(2, dtype=torch.bool, device="cuda"),
            truncateds=torch.zeros(2, dtype=torch.bool, device="cuda"),
            aux_tensors=None,
        )


def test_multi_trajectory_short_rollout_yields_no_sequence_minibatches() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=1)
    obs = torch.randn(2, 3, 16, 8)
    actions = torch.randn(2, 4)
    log_probs = torch.randn(2)
    values = torch.full((2,), 0.5)
    rewards = torch.ones(2)
    dones = torch.zeros(2, dtype=torch.bool)
    truncateds = torch.zeros(2, dtype=torch.bool)

    buffer.append_batch(
        observations=obs,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        truncateds=truncateds,
        aux_tensors=None,
    )
    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    batches = list(buffer.sample_minibatches(batch_size=4, seq_len=8))

    assert batches == []


def test_multi_trajectory_batched_storage_reuses_allocations_across_clear() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    buffer.append_batch(
        observations=torch.randn(2, 3, 16, 8),
        actions=torch.randn(2, 4),
        log_probs=torch.randn(2),
        values=torch.randn(2),
        rewards=torch.randn(2),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=torch.randn(2, 3),
    )

    obs_storage = buffer._batch_obs
    action_storage = buffer._batch_actions
    aux_storage = buffer._batch_aux

    buffer.clear()

    buffer.append_batch(
        observations=torch.randn(2, 3, 16, 8),
        actions=torch.randn(2, 4),
        log_probs=torch.randn(2),
        values=torch.randn(2),
        rewards=torch.randn(2),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
    )
    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))
    buffer._ensure_batch_cache()

    assert buffer._batch_obs is obs_storage
    assert buffer._batch_actions is action_storage
    assert buffer._batch_aux is aux_storage
    assert buffer._all_aux is None


def test_multi_trajectory_append_batch_supports_sparse_actor_indices() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=4, gamma=0.99, gae_lambda=0.95, capacity=2)

    buffer.append_batch(
        observations=torch.full((2, 3, 16, 8), 1.0),
        actions=torch.full((2, 4), 1.0),
        log_probs=torch.tensor([0.1, 0.2]),
        values=torch.tensor([0.3, 0.4]),
        rewards=torch.tensor([1.0, 2.0]),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
        actor_indices=torch.tensor([2, 0], dtype=torch.int64),
    )
    buffer.append_batch(
        observations=torch.full((2, 3, 16, 8), 2.0),
        actions=torch.full((2, 4), 2.0),
        log_probs=torch.tensor([0.5, 0.6]),
        values=torch.tensor([0.7, 0.8]),
        rewards=torch.tensor([3.0, 4.0]),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
        actor_indices=torch.tensor([1, 3], dtype=torch.int64),
    )
    buffer.append_batch(
        observations=torch.full((2, 3, 16, 8), 3.0),
        actions=torch.full((2, 4), 3.0),
        log_probs=torch.tensor([0.9, 1.0]),
        values=torch.tensor([1.1, 1.2]),
        rewards=torch.tensor([5.0, 6.0]),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
        actor_indices=torch.tensor([2, 0], dtype=torch.int64),
    )
    buffer.append_batch(
        observations=torch.full((2, 3, 16, 8), 4.0),
        actions=torch.full((2, 4), 4.0),
        log_probs=torch.tensor([1.3, 1.4]),
        values=torch.tensor([1.5, 1.6]),
        rewards=torch.tensor([7.0, 8.0]),
        dones=torch.zeros(2, dtype=torch.bool),
        truncateds=torch.zeros(2, dtype=torch.bool),
        aux_tensors=None,
        actor_indices=torch.tensor([1, 3], dtype=torch.int64),
    )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(4))
    buffer._ensure_batch_cache()

    assert buffer._all_obs is not None
    assert buffer._all_obs.shape[:2] == (4, 2)
    assert torch.allclose(buffer._all_obs[0, 0], torch.full((3, 16, 8), 1.0))
    assert torch.allclose(buffer._all_obs[0, 1], torch.full((3, 16, 8), 3.0))
    assert torch.allclose(buffer._all_obs[1, 0], torch.full((3, 16, 8), 2.0))
    assert torch.allclose(buffer._all_obs[1, 1], torch.full((3, 16, 8), 4.0))
    assert len(buffer) == 8


def test_multi_trajectory_sparse_append_requires_equal_rollout_lengths_before_finalize() -> None:
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    buffer.append_batch(
        observations=torch.randn(1, 3, 16, 8),
        actions=torch.randn(1, 4),
        log_probs=torch.randn(1),
        values=torch.randn(1),
        rewards=torch.randn(1),
        dones=torch.zeros(1, dtype=torch.bool),
        truncateds=torch.zeros(1, dtype=torch.bool),
        aux_tensors=None,
        actor_indices=torch.tensor([0], dtype=torch.int64),
    )

    import pytest

    with pytest.raises(RuntimeError, match="equal per-actor rollout lengths"):
        buffer.compute_returns_and_advantages(last_values=torch.zeros(2))


def test_multi_trajectory_requires_explicit_capacity_on_canonical_path() -> None:
    import pytest

    with pytest.raises(ValueError, match="requires capacity"):
        MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95)


def test_trajectory_buffer_sequence_sampling_exposes_tensor_native_sequence_views() -> None:
    buffer = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)

    for step in range(4):
        buffer.append(
            PPOTransition(
                observation=torch.randn(3, 16, 8),
                action=torch.randn(4),
                log_prob=-0.25 * (step + 1),
                value=0.5 * (step + 1),
                reward=float(step + 1),
                done=(step == 3),
                hidden_state=torch.randn(1, 1, 8),
                aux_tensor=torch.randn(3),
            )
        )

    buffer.compute_returns_and_advantages(last_value=0.0)
    batches = list(buffer.sample_minibatches(batch_size=4, seq_len=2))

    assert batches
    first = batches[0]
    assert first.sequence_observations is not None
    assert first.sequence_observations.shape == (2, 2, 3, 16, 8)
    assert first.sequence_actions is not None
    assert first.sequence_actions.shape == (2, 2, 4)
    assert first.dones is not None
    assert first.dones.shape == (2, 2)
    assert first.sequence_aux_tensors is not None
    assert first.sequence_aux_tensors.shape == (2, 2, 3)
