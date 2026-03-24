"""Tests for PpoLearner (full PPO training).

PPO learner tests pin temporal_core="gru" for stable gradient dynamics on
synthetic data.  The canonical Mamba2 SSD default is validated by
test_cognitive_policy.py::test_policy_uses_canonical_mamba2_temporal_core.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.learner_ppo import PpoLearner, PpoMetrics, _materialize_metric_means
from navi_actor.rollout_buffer import MultiTrajectoryBuffer, PPOTransition, TrajectoryBuffer


def _make_ppo_policy() -> CognitiveMambaPolicy:
    """GRU-pinned policy for stable PPO mechanics tests on synthetic data."""
    return CognitiveMambaPolicy(embedding_dim=128, temporal_core="gru")


@torch.no_grad()
def _fill_buffer(policy: CognitiveMambaPolicy, n: int = 64) -> TrajectoryBuffer:
    """Create a trajectory buffer with policy-derived log_probs for KL stability."""
    buf = TrajectoryBuffer(gamma=0.99, gae_lambda=0.95)
    policy.eval()
    for _ in range(n):
        obs = torch.randn(3, 128, 24)
        action, log_prob, value, *_ = policy(obs.unsqueeze(0))
        buf.append(
            PPOTransition(
                observation=obs,
                action=action.squeeze(0).detach(),
                log_prob=float(log_prob.squeeze()),
                value=float(value.squeeze()),
                reward=1.0,
                done=False,
            )
        )
    policy.train()
    buf.compute_returns_and_advantages(last_value=0.0)
    return buf


def test_ppo_epoch_returns_metrics() -> None:
    """train_ppo_epoch should return PpoMetrics."""
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(policy, 64)

    metrics = learner.train_ppo_epoch(
        policy,
        buf,
        ppo_epochs=1,
        minibatch_size=32,
        seq_len=0,
    )
    assert isinstance(metrics, PpoMetrics)
    assert metrics.policy_loss != 0.0 or metrics.value_loss != 0.0
    assert metrics.minibatch_prep_ms >= 0.0
    assert metrics.policy_eval_ms >= 0.0
    assert metrics.backward_ms >= 0.0
    assert metrics.grad_clip_ms >= 0.0
    assert metrics.optimizer_step_ms >= 0.0
    assert metrics.rnd_step_ms >= 0.0
    assert metrics.post_update_stats_ms >= 0.0
    assert metrics.update_loop_overhead_ms >= 0.0
    assert metrics.epoch_finalize_ms >= 0.0
    assert metrics.epoch_setup_ms >= 0.0
    assert metrics.policy_optimizer_init_ms >= 0.0
    assert metrics.rnd_optimizer_init_ms >= 0.0
    assert metrics.policy_train_mode_ms >= 0.0
    assert metrics.policy_param_cache_ms >= 0.0
    assert metrics.setup_overhead_ms >= 0.0
    assert metrics.epoch_total_ms >= metrics.epoch_finalize_ms
    assert metrics.policy_eval_device_ms_total == 0.0
    assert metrics.backward_device_ms_total == 0.0
    assert metrics.optimizer_step_device_ms_total == 0.0
    assert metrics.n_updates > 0
    assert metrics.policy_eval_ms_total >= metrics.policy_eval_ms
    assert metrics.optimizer_step_ms_total >= metrics.optimizer_step_ms
    assert metrics.post_update_stats_ms_total >= metrics.post_update_stats_ms
    assert metrics.update_loop_overhead_ms_total >= metrics.update_loop_overhead_ms


def test_materialize_metric_means_packs_epoch_metrics_once() -> None:
    """Learner epoch metrics should be materialized through one packed host tensor."""
    payload = _materialize_metric_means(
        running_policy_loss=torch.tensor(6.0),
        running_value_loss=torch.tensor(9.0),
        running_entropy=torch.tensor(12.0),
        running_kl=torch.tensor(15.0),
        running_clip=torch.tensor(18.0),
        running_total=torch.tensor(21.0),
        running_rnd_loss=torch.tensor(24.0),
        n_updates=3,
    )

    assert payload.device.type == "cpu"
    assert payload.shape == (7,)
    assert torch.allclose(
        payload,
        torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32),
    )


def test_ppo_epoch_improves_loss() -> None:
    """Multiple PPO epochs should change the loss."""
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(policy, 128)

    m1 = learner.train_ppo_epoch(policy, buf, ppo_epochs=1, minibatch_size=64, seq_len=0)
    # Rebuild buffer for second pass (same data)
    buf2 = _fill_buffer(policy, 128)
    m2 = learner.train_ppo_epoch(policy, buf2, ppo_epochs=1, minibatch_size=64, seq_len=0)
    # Losses should differ between calls (policy updated)
    assert m1.total_loss != m2.total_loss or m1.policy_loss != m2.policy_loss


def test_ppo_clip_fraction_bounded() -> None:
    """Clip fraction should be between 0 and 1."""
    policy = _make_ppo_policy()
    learner = PpoLearner(clip_ratio=0.2)
    buf = _fill_buffer(policy, 64)
    metrics = learner.train_ppo_epoch(
        policy,
        buf,
        ppo_epochs=2,
        minibatch_size=32,
        seq_len=0,
    )
    assert 0.0 <= metrics.clip_fraction <= 1.0


def test_ppo_epoch_can_skip_summary_scalar_materialization() -> None:
    """Canonical trainer may skip epoch-end scalar host sync when update telemetry is off."""
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(policy, 64)

    metrics = learner.train_ppo_epoch(
        policy,
        buf,
        ppo_epochs=1,
        minibatch_size=32,
        seq_len=0,
        materialize_summary_scalars=False,
    )

    assert metrics.n_updates > 0
    assert metrics.epoch_finalize_ms == 0.0
    assert metrics.policy_loss == 0.0
    assert metrics.value_loss == 0.0
    assert metrics.entropy == 0.0
    assert metrics.approx_kl == 0.0
    assert metrics.clip_fraction == 0.0
    assert metrics.total_loss == 0.0
    assert metrics.rnd_loss == 0.0


def test_empty_buffer_returns_zeros() -> None:
    """An empty buffer should return zero metrics."""
    policy = _make_ppo_policy()
    learner = PpoLearner()
    buf = TrajectoryBuffer()
    buf.compute_returns_and_advantages(last_value=0.0)
    metrics = learner.train_ppo_epoch(policy, buf, ppo_epochs=1, minibatch_size=32, seq_len=0)
    assert metrics.total_loss == 0.0
    assert metrics.minibatch_prep_ms == 0.0
    assert metrics.policy_eval_ms == 0.0
    assert metrics.backward_ms == 0.0
    assert metrics.grad_clip_ms == 0.0
    assert metrics.optimizer_step_ms == 0.0
    assert metrics.rnd_step_ms == 0.0
    assert metrics.post_update_stats_ms == 0.0
    assert metrics.update_loop_overhead_ms == 0.0
    assert metrics.n_updates == 0
    assert metrics.epoch_setup_ms == 0.0
    assert metrics.policy_optimizer_init_ms == 0.0
    assert metrics.rnd_optimizer_init_ms == 0.0
    assert metrics.policy_train_mode_ms == 0.0
    assert metrics.policy_param_cache_ms == 0.0
    assert metrics.setup_overhead_ms == 0.0
    assert metrics.policy_eval_ms_total == 0.0
    assert metrics.optimizer_step_ms_total == 0.0
    assert metrics.post_update_stats_ms_total == 0.0
    assert metrics.update_loop_overhead_ms_total == 0.0
    assert metrics.epoch_finalize_ms == 0.0
    assert metrics.epoch_total_ms == 0.0
    assert metrics.policy_eval_device_ms_total == 0.0
    assert metrics.backward_device_ms_total == 0.0
    assert metrics.optimizer_step_device_ms_total == 0.0


def test_default_value_coeff_is_low() -> None:
    """value_coeff default should be 0.5 (canonical PPO value coefficient)."""
    from navi_actor.config import ActorConfig

    cfg = ActorConfig()
    assert cfg.value_coeff == 0.5


def test_optimizer_covers_full_policy() -> None:
    """PpoLearner should keep one optimizer over the full policy module."""
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)

    optimizer = learner._get_optimizer(policy)

    optimizer_param_ids = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer_param_ids.add(id(parameter))

    all_policy_param_ids = {id(parameter) for parameter in policy.parameters()}
    critic_param_ids = {id(p) for p in policy.heads.critic.parameters()}

    assert optimizer_param_ids == all_policy_param_ids, (
        "Unified PPO optimizer must cover the full policy module"
    )
    assert critic_param_ids.issubset(optimizer_param_ids), (
        "Unified PPO optimizer must include critic head parameters"
    )


def test_prime_update_runtime_eagerly_populates_optimizer_and_param_cache() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)

    learner.prime_update_runtime(policy)

    assert learner._optimizer is not None
    assert learner._policy_param_cache is not None
    assert len(learner._policy_param_cache) > 0


def test_create_adam_optimizer_falls_back_to_foreach_when_fused_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = PpoLearner(learning_rate=1e-3)
    parameter = torch.nn.Parameter(torch.zeros(1))
    calls: list[dict[str, object]] = []

    def fake_adam(params: object, **kwargs: object) -> torch.optim.Optimizer:
        del params
        calls.append(dict(kwargs))
        if kwargs.get("fused"):
            raise RuntimeError("fused unsupported")
        return cast("torch.optim.Optimizer", object())

    monkeypatch.setattr(torch.optim, "Adam", fake_adam)

    optimizer = learner._create_adam_optimizer(
        (parameter,),
        learning_rate=1e-3,
        use_cuda=True,
        label="policy",
    )

    assert optimizer is not None
    assert calls == [
        {"lr": 1e-3, "fused": True},
        {"lr": 1e-3, "foreach": True},
    ]


def test_gradient_isolation_policy_updates_actor() -> None:
    """After one PPO step, actor head weights should change but not due to value loss."""
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-2, value_coeff=0.005)

    # Record initial actor + critic weights
    actor_w_before = policy.heads.actor[0].weight.data.clone()
    critic_w_before = policy.heads.critic[0].weight.data.clone()

    buf = _fill_buffer(policy, 64)
    learner.train_ppo_epoch(
        policy,
        buf,
        ppo_epochs=2,
        minibatch_size=32,
        seq_len=0,
    )

    actor_w_after = policy.heads.actor[0].weight.data
    critic_w_after = policy.heads.critic[0].weight.data

    # Both heads should have been updated
    assert not torch.allclose(actor_w_before, actor_w_after), (
        "Actor head should be updated by policy loss"
    )
    assert not torch.allclose(critic_w_before, critic_w_after), (
        "Critic head should be updated by value loss"
    )


def test_ppo_epoch_invokes_progress_callback() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buf = _fill_buffer(policy, 64)
    callback_calls = 0

    def on_progress() -> None:
        nonlocal callback_calls
        callback_calls += 1

    learner.train_ppo_epoch(
        policy,
        buf,
        ppo_epochs=1,
        minibatch_size=32,
        seq_len=0,
        progress_callback=on_progress,
    )

    assert callback_calls > 0


def test_ppo_epoch_accepts_tensor_native_sequence_minibatches() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    policy.eval()
    with torch.no_grad():
        for step in range(1, 3):
            obs = torch.randn(2, 3, 128, 24)
            actions_list, lp_list, val_list = [], [], []
            for i in range(2):
                a, lp, v, *_ = policy(obs[i : i + 1])
                actions_list.append(a.squeeze(0))
                lp_list.append(lp.squeeze())
                val_list.append(v.squeeze())
            buffer.append_batch(
                observations=obs,
                actions=torch.stack(actions_list),
                log_probs=torch.stack(lp_list),
                values=torch.stack(val_list),
                rewards=torch.full((2,), float(step)),
                dones=torch.zeros(2, dtype=torch.bool),
                truncateds=torch.zeros(2, dtype=torch.bool),
                aux_tensors=torch.randn(2, 3),
            )
    policy.train()

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    metrics = learner.train_ppo_epoch(
        policy,
        buffer,
        ppo_epochs=1,
        minibatch_size=4,
        seq_len=2,
    )

    assert isinstance(metrics, PpoMetrics)
    assert metrics.total_loss != 0.0 or metrics.policy_loss != 0.0


def test_ppo_epoch_sequence_minibatch_does_not_require_flat_obs_tensors() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    for _ in range(2):
        buffer.append_batch(
            observations=torch.randn(2, 3, 128, 24),
            actions=torch.randn(2, 4),
            log_probs=torch.randn(2),
            values=torch.randn(2),
            rewards=torch.randn(2),
            dones=torch.zeros(2, dtype=torch.bool),
            truncateds=torch.zeros(2, dtype=torch.bool),
            aux_tensors=torch.randn(2, 3),
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    first_minibatch = next(buffer.sample_minibatches(batch_size=4, seq_len=2))
    poisoned_minibatch = replace(
        first_minibatch,
        observations=torch.empty((4, 3, 128, 24), device=torch.device("meta")),
        actions=torch.empty((4, 4), device=torch.device("meta")),
        aux_tensors=torch.empty((4, 3), device=torch.device("meta")),
    )

    class _SingleBatchBuffer:
        def __len__(self) -> int:
            return 4

        def sample_minibatches(
            self,
            batch_size: int,
            seq_len: int,
        ) -> object:
            del batch_size, seq_len
            yield poisoned_minibatch

    metrics = learner.train_ppo_epoch(
        policy,
        cast("TrajectoryBuffer | MultiTrajectoryBuffer", _SingleBatchBuffer()),
        ppo_epochs=1,
        minibatch_size=4,
        seq_len=2,
    )

    assert isinstance(metrics, PpoMetrics)


def test_ppo_epoch_sequence_minibatch_requires_sequence_native_views() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    for _ in range(2):
        buffer.append_batch(
            observations=torch.randn(2, 3, 128, 24),
            actions=torch.randn(2, 4),
            log_probs=torch.randn(2),
            values=torch.randn(2),
            rewards=torch.randn(2),
            dones=torch.zeros(2, dtype=torch.bool),
            truncateds=torch.zeros(2, dtype=torch.bool),
            aux_tensors=torch.randn(2, 3),
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    first_minibatch = next(buffer.sample_minibatches(batch_size=4, seq_len=2))
    poisoned_minibatch = replace(
        first_minibatch,
        sequence_observations=None,
        sequence_actions=None,
        sequence_aux_tensors=None,
    )

    class _SingleBatchBuffer:
        def __len__(self) -> int:
            return 4

        def sample_minibatches(
            self,
            batch_size: int,
            seq_len: int,
        ) -> object:
            del batch_size, seq_len
            yield poisoned_minibatch

    with pytest.raises(RuntimeError, match="requires sequence-native minibatches"):
        learner.train_ppo_epoch(
            policy,
            cast("TrajectoryBuffer | MultiTrajectoryBuffer", _SingleBatchBuffer()),
            ppo_epochs=1,
            minibatch_size=4,
            seq_len=2,
        )


def test_ppo_epoch_sequence_minibatch_works_without_hidden_fields() -> None:
    policy = _make_ppo_policy()
    learner = PpoLearner(learning_rate=1e-3)
    buffer = MultiTrajectoryBuffer(n_actors=2, gamma=0.99, gae_lambda=0.95, capacity=2)

    for _ in range(2):
        buffer.append_batch(
            observations=torch.randn(2, 3, 128, 24),
            actions=torch.randn(2, 4),
            log_probs=torch.randn(2),
            values=torch.randn(2),
            rewards=torch.randn(2),
            dones=torch.zeros(2, dtype=torch.bool),
            truncateds=torch.zeros(2, dtype=torch.bool),
            aux_tensors=torch.randn(2, 3),
        )

    buffer.compute_returns_and_advantages(last_values=torch.zeros(2))

    metrics = learner.train_ppo_epoch(
        policy,
        buffer,
        ppo_epochs=1,
        minibatch_size=4,
        seq_len=2,
    )

    assert isinstance(metrics, PpoMetrics)
