"""Tests for full training-state checkpoint (knowledge accumulation)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from navi_actor.config import ActorConfig
from navi_actor.training.ppo_trainer import PpoTrainer
from navi_contracts import (
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    DistanceMatrix,
    RobotPose,
    TelemetryEvent,
    deserialize,
)


def _make_trainer() -> PpoTrainer:
    """Create a canonical PpoTrainer with minimal config for state tests."""
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
    _actions, log_probs, values, _, _ = trainer._learner_policy(obs)
    loss = -log_probs.sum() + values.sum()
    opt = trainer._learner._get_optimizer(trainer._learner_policy)
    opt.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    opt.step()

    # RND predictor step
    z = trainer._learner_policy.encode(obs.detach())
    rnd_opt = trainer._learner._get_rnd_optimizer(trainer._rnd)
    rnd_loss = trainer._rnd.distillation_loss(z)
    rnd_opt.zero_grad()
    rnd_loss.backward()  # type: ignore[no-untyped-call]
    rnd_opt.step()

    # Snapshot state before save
    policy_sd = {k: v.clone() for k, v in trainer._learner_policy.state_dict().items()}
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
            policy_sd[key], trainer2._learner_policy.state_dict()[key]
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


def test_legacy_checkpoint_is_rejected() -> None:
    """Loading a non-canonical checkpoint format should fail fast."""
    trainer = _make_trainer()

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "legacy.pt"
        # Save old-format: just the policy state_dict
        torch.save(trainer._learner_policy.state_dict(), ckpt)

        trainer2 = _make_trainer()
        with pytest.raises(RuntimeError, match="expected version=2 canonical state"):
            trainer2.load_training_state(ckpt)


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


def test_publish_observation_emits_distance_matrix_topic() -> None:
    """Canonical trainer should provide a low-volume live observation feed."""
    trainer = _make_trainer()

    class FakeSocket:
        def __init__(self) -> None:
            self.messages: list[list[bytes]] = []

        def send_multipart(self, parts: list[bytes]) -> None:
            self.messages.append(parts)

    fake_socket = FakeSocket()
    trainer._pub_socket = fake_socket  # type: ignore[assignment]

    dm = DistanceMatrix(
        episode_id=1,
        env_ids=np.array([0], dtype=np.int32),
        matrix_shape=(8, 4),
        depth=np.ones((1, 8, 4), dtype=np.float32),
        delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
        semantic=np.zeros((1, 8, 4), dtype=np.int32),
        valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
        overhead=np.zeros((8, 8, 3), dtype=np.float32),
        robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
        step_id=7,
        timestamp=1.0,
    )

    trainer._publish_observation(dm)

    assert len(fake_socket.messages) == 1
    topic, payload = fake_socket.messages[0]
    assert topic == TOPIC_DISTANCE_MATRIX.encode("utf-8")
    restored = deserialize(payload)
    assert isinstance(restored, DistanceMatrix)
    assert restored.step_id == 7


def test_publish_perf_telemetry_tolerates_socket_send_failure() -> None:
    """Perf-only attribution paths should fail safely if PUB transport errors."""
    trainer = _make_trainer()

    class FailingSocket:
        def send_multipart(self, parts: list[bytes]) -> None:
            del parts
            raise RuntimeError("synthetic send failure")

    trainer._pub_socket = FailingSocket()  # type: ignore[assignment]

    trainer._publish_perf_telemetry(
        step_id=256,
        sps=120.0,
        forward_pass_ms=10.0,
        batch_step_ms=8.0,
        memory_query_ms=1.0,
        transition_ms=5.0,
        tick_total_ms=25.0,
        zero_wait_ratio=0.0,
        ppo_update_ms=19_000.0,
        host_extract_ms=0.5,
        telemetry_publish_ms=0.0,
    )


def test_publish_runtime_perf_emits_environment_perf_event() -> None:
    """Canonical trainer should publish coarse sdfdag runtime perf events."""
    trainer = _make_trainer()

    class FakeSocket:
        def __init__(self) -> None:
            self.messages: list[list[bytes]] = []

        def send_multipart(self, parts: list[bytes]) -> None:
            self.messages.append(parts)

    class FakeRuntime:
        def perf_snapshot(self) -> object:
            return type(
                "Snapshot",
                (),
                {
                    "sps": 63.5,
                    "last_batch_step_ms": 14.2,
                    "ema_batch_step_ms": 13.8,
                    "avg_batch_step_ms": 14.0,
                    "avg_actor_step_ms": 3.5,
                    "total_batches": 9,
                    "total_actor_steps": 36,
                },
            )()

        def close(self) -> None:
            return None

    fake_socket = FakeSocket()
    trainer._pub_socket = fake_socket  # type: ignore[assignment]
    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    trainer._publish_runtime_perf(step_id=200)

    assert len(fake_socket.messages) == 1
    topic, payload = fake_socket.messages[0]
    assert topic == TOPIC_TELEMETRY_EVENT.encode("utf-8")
    restored = deserialize(payload)
    assert isinstance(restored, TelemetryEvent)
    assert restored.event_type == "environment.sdfdag.perf"
    assert restored.step_id == 200


def test_publish_runtime_perf_tolerates_runtime_snapshot_failure() -> None:
    """Perf-only diagnostics should not crash when runtime perf snapshot fails."""
    trainer = _make_trainer()

    class FakeRuntime:
        def perf_snapshot(self) -> object:
            raise RuntimeError("synthetic perf failure")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]
    trainer._pub_socket = object()  # type: ignore[assignment]

    trainer._publish_runtime_perf(step_id=200)


def test_rnd_target_network_preserved() -> None:
    """RND target network (frozen random) must be identical after reload.

    This is critical: if the target randomises, curiosity signals become
    meaningless across scenes.
    """
    trainer = _make_trainer()
    z = torch.randn(1, 64, device=trainer._device)

    # Get raw RND outputs (target + predictor) — no running-stat mutation
    with torch.no_grad():
        target_before, pred_before = trainer._rnd(z)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "state.pt"
        trainer.save_training_state(ckpt)

        trainer2 = _make_trainer()
        trainer2.load_training_state(ckpt)

    with torch.no_grad():
        target_loaded, pred_loaded = trainer2._rnd(z.to(trainer2._device))

    assert torch.allclose(target_before, target_loaded, atol=1e-6), (
        "RND target network mismatch after reload"
    )
    assert torch.allclose(pred_before, pred_loaded, atol=1e-6), (
        "RND predictor network mismatch after reload"
    )


def test_request_batch_step_tensor_actions_requires_runtime_action_tensor_seam() -> None:
    """Canonical trainer should fail fast if tensor action stepping is missing."""
    trainer = _make_trainer()

    class FakeRuntime:
        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            del episode_id, actor_id, materialize
            return torch.ones((3, 8, 4), device=trainer._device), None

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="requires tensor-native runtime seams"):
        trainer._request_batch_step_tensor_actions(
            torch.ones((1, 4), device="cpu", dtype=torch.float32),
            11,
            publish_actor_ids=(0,),
        )


def test_request_batch_step_tensor_actions_uses_runtime_action_tensor_seam() -> None:
    """Canonical trainer should step through the runtime action tensor seam."""
    trainer = _make_trainer()

    class FakeTensorBatch:
        def __init__(self) -> None:
            self.observation_tensor = torch.ones((1, 3, 8, 4), device=trainer._device)
            self.reward_tensor = torch.tensor([1.25], dtype=torch.float32, device=trainer._device)
            self.done_tensor = torch.tensor([False], dtype=torch.bool, device=trainer._device)
            self.truncated_tensor = torch.tensor([True], dtype=torch.bool, device=trainer._device)
            self.episode_id_tensor = torch.tensor([5], dtype=torch.int64, device=trainer._device)
            self.published_observations: dict[int, DistanceMatrix] = {}

    class FakeRuntime:
        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            del episode_id, actor_id, materialize
            return torch.ones((3, 8, 4), device=trainer._device), None

        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            publish_actor_ids: tuple[int, ...] = (),
        ) -> tuple[FakeTensorBatch, tuple[object, ...]]:
            del publish_actor_ids
            assert action_tensor.shape == (1, 4)
            assert step_id == 19
            result = type("FakeResult", (), {"episode_id": 5})()
            return FakeTensorBatch(), (result,)

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    payload = trainer._request_batch_step_tensor_actions(
        torch.ones((1, 4), device="cpu", dtype=torch.float32),
        19,
        publish_actor_ids=(0,),
    )

    step_batch, results = payload
    assert step_batch.observation_tensor.shape == (1, 3, 8, 4)
    assert torch.allclose(step_batch.reward_tensor, torch.tensor([1.25], dtype=torch.float32, device=trainer._device))
    assert torch.equal(step_batch.done_tensor, torch.tensor([False], dtype=torch.bool, device=trainer._device))
    assert torch.equal(step_batch.truncated_tensor, torch.tensor([True], dtype=torch.bool, device=trainer._device))
    assert torch.equal(step_batch.episode_id_tensor, torch.tensor([5], dtype=torch.int64, device=trainer._device))
    assert len(results) == 1
    assert step_batch.published_observations == {}


def test_load_initial_observation_batch_requires_tensor_reset_seam() -> None:
    """Canonical trainer should fail fast if tensor reset is missing."""
    trainer = _make_trainer()

    class FakeRuntime:
        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            publish_actor_ids: tuple[int, ...] = (),
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids
            raise AssertionError("not used")

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="requires tensor-native runtime seams"):
        trainer._load_initial_observation_batch()


def test_load_initial_observation_batch_uses_tensor_reset_seam() -> None:
    """Canonical trainer should seed initial rollout state from tensor resets."""
    trainer = _make_trainer()

    class FakeRuntime:
        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            publish_actor_ids: tuple[int, ...] = (),
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids
            raise AssertionError("not used")

        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            assert episode_id == 0
            obs = torch.full((3, 8, 4), float(actor_id + 1), device=trainer._device)
            published = None
            if materialize:
                published = DistanceMatrix(
                    episode_id=episode_id,
                    env_ids=np.array([actor_id], dtype=np.int32),
                    matrix_shape=(8, 4),
                    depth=np.ones((1, 8, 4), dtype=np.float32) * float(actor_id + 1),
                    delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
                    semantic=np.zeros((1, 8, 4), dtype=np.int32),
                    valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
                    overhead=np.zeros((8, 8, 3), dtype=np.float32),
                    robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
                    step_id=0,
                    timestamp=1.0,
                )
            return obs, published

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    obs_batch, published = trainer._load_initial_observation_batch()

    assert obs_batch.shape == (1, 3, 8, 4)
    assert float(obs_batch[0, 0, 0, 0]) == 1.0
    assert 0 in published
    assert published[0].step_id == 0


def test_extract_host_rollout_scalars_skips_telemetry_columns_when_not_needed() -> None:
    """Canonical trainer should not mirror intrinsic and loop scalars when step telemetry is off."""
    trainer = _make_trainer()

    payload = trainer._extract_host_rollout_scalars(
        shaped_rewards=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
        loop_similarities=torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
        include_telemetry_scalars=False,
    )

    assert payload.shaped_reward_tensor.device.type == "cpu"
    assert torch.allclose(payload.shaped_reward_tensor, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert payload.intrinsic_reward_tensor is None
    assert payload.loop_similarity_tensor is None


def test_extract_host_rollout_scalars_batches_telemetry_columns_when_needed() -> None:
    """Canonical trainer should batch the unavoidable scalar host copy for telemetry ticks."""
    trainer = _make_trainer()

    payload = trainer._extract_host_rollout_scalars(
        shaped_rewards=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
        loop_similarities=torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
        include_telemetry_scalars=True,
    )

    assert payload.intrinsic_reward_tensor is not None
    assert payload.loop_similarity_tensor is not None
    assert payload.shaped_reward_tensor.device.type == "cpu"
    assert payload.intrinsic_reward_tensor.device.type == "cpu"
    assert payload.loop_similarity_tensor.device.type == "cpu"
    assert torch.allclose(payload.shaped_reward_tensor, torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert torch.allclose(payload.intrinsic_reward_tensor, torch.tensor([3.0, 4.0], dtype=torch.float32))
    assert torch.allclose(payload.loop_similarity_tensor, torch.tensor([5.0, 6.0], dtype=torch.float32))
