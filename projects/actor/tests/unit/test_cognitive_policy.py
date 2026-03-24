"""Tests for CognitiveMambaPolicy."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from navi_actor.cognitive_policy import CognitiveMambaPolicy
from navi_actor.config import TemporalCoreName
from navi_actor.gru_core import GRUTemporalCore
from navi_actor.mamba2_core import Mamba2SSDTemporalCore
from navi_actor.mambapy_core import MambapyTemporalCore


def _make_policy(*, temporal_core: TemporalCoreName = "mamba2") -> CognitiveMambaPolicy:
    return CognitiveMambaPolicy(
        embedding_dim=128,
        temporal_core=temporal_core,
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
    assert z_t.requires_grad is False


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
    assert z_t.requires_grad is False


def test_evaluate_profiled_reports_stage_metrics() -> None:
    """Profiled evaluation should expose per-stage attribution on the canonical path."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    acts = torch.randn(2, 4)

    lp, val, ent, _, z_t, metrics = policy.evaluate_profiled(obs, acts, use_cuda_events=False)

    assert lp.shape == (2,)
    assert val.shape == (2,)
    assert ent.dim() == 0
    assert z_t.shape == (2, 128)
    assert z_t.requires_grad is False
    assert metrics.encode_ms >= 0.0
    assert metrics.temporal_ms >= 0.0
    assert metrics.heads_ms >= 0.0


def test_evaluate_sequence_profiled_reports_stage_metrics() -> None:
    """Profiled sequence evaluation should expose encoder/temporal/heads attribution."""
    policy = _make_policy()
    obs_seq = torch.randn(2, 4, 3, 128, 24)
    acts_seq = torch.randn(2, 4, 4)

    lp, val, ent, _, z_t, metrics = policy.evaluate_sequence_profiled(
        obs_seq,
        acts_seq,
        use_cuda_events=False,
    )

    assert lp.shape == (8,)
    assert val.shape == (8,)
    assert ent.dim() == 0
    assert z_t.shape == (8, 128)
    assert z_t.requires_grad is False
    assert metrics.encode_ms >= 0.0
    assert metrics.temporal_ms >= 0.0
    assert metrics.heads_ms >= 0.0


def test_checkpoint_roundtrip() -> None:
    """Save and reload should produce identical state_dict."""
    policy = _make_policy()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "test.pt"
        policy.save_checkpoint(ckpt)
        loaded = CognitiveMambaPolicy.load_checkpoint(
            ckpt,
            embedding_dim=128,
            azimuth_bins=64,
            elevation_bins=32,
        )
    for key in policy.state_dict():
        assert torch.equal(policy.state_dict()[key], loaded.state_dict()[key])


def test_gradient_flow() -> None:
    """Gradients should propagate through the full policy."""
    policy = _make_policy()
    obs = torch.randn(2, 3, 128, 24)
    _, log_probs, values, _, _ = policy.forward(obs)
    loss = log_probs.sum() + values.sum()
    loss.backward()
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
    values.sum().backward()

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
        p.grad
        for p in policy.heads.critic.parameters()
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
    values.sum().backward()

    for name, p in policy.encoder.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"encoder.{name} must not receive value-loss gradients"
        )
    for name, p in policy.temporal_core.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), (
            f"temporal_core.{name} must not receive value-loss gradients"
        )

    critic_grads = [
        p.grad
        for p in policy.heads.critic.parameters()
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
    (-log_probs.sum()).backward()

    # Encoder and temporal core SHOULD have gradients from policy loss
    encoder_grads = [
        p.grad
        for p in policy.encoder.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(encoder_grads) > 0, "Encoder must receive policy-loss gradients"

    temporal_grads = [
        p.grad
        for p in policy.temporal_core.parameters()
        if p.grad is not None and not torch.all(p.grad == 0)
    ]
    assert len(temporal_grads) > 0, "Temporal core must receive policy-loss gradients"


def test_policy_uses_canonical_mamba2_temporal_core() -> None:
    """The default policy should instantiate the canonical Mamba2 SSD core."""
    policy = _make_policy()

    assert isinstance(policy.temporal_core, Mamba2SSDTemporalCore)


def test_gru_temporal_core_flattens_once_per_cuda_device() -> None:
    """GRU core should only request cuDNN parameter flattening once per device."""
    core = GRUTemporalCore(d_model=8)
    calls: list[torch.device] = []

    def record_flatten() -> None:
        calls.append(torch.device("cuda", 0))

    core.core.flatten_parameters = record_flatten  # type: ignore[method-assign]

    core._maybe_flatten_parameters(torch.device("cuda", 0))
    core._maybe_flatten_parameters(torch.device("cuda", 0))
    core._maybe_flatten_parameters(torch.device("cuda", 1))
    core._maybe_flatten_parameters(torch.device("cpu"))
    core._maybe_flatten_parameters(torch.device("cuda", 0))

    assert len(calls) == 3


def test_policy_can_select_mambapy_temporal_core() -> None:
    """The policy should instantiate Mambapy when the selector requests it."""
    pytest.importorskip("mambapy")
    policy = _make_policy(temporal_core="mambapy")

    assert isinstance(policy.temporal_core, MambapyTemporalCore)


def test_policy_rejects_unsupported_temporal_core() -> None:
    """Unsupported temporal-core names must fail fast at policy construction."""
    with pytest.raises(ValueError, match="Unsupported temporal core"):
        CognitiveMambaPolicy(temporal_core=cast("Any", "mamba-ssm"))
