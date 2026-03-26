"""Tests for full training-state checkpoint (knowledge accumulation)."""

from __future__ import annotations

import json
import queue
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from navi_actor.config import ActorConfig
from navi_actor.rollout_buffer import MultiTrajectoryBuffer
from navi_actor.training.ppo_trainer import PpoTrainer, _PendingRolloutGroup
from navi_contracts import (
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    DistanceMatrix,
    RobotPose,
    TelemetryEvent,
    deserialize,
)
from navi_contracts.testing.oracle_house import (
    house_observation,
    house_observation_after_forward_motion,
)


def _assert_same_device(actual: torch.device, expected: torch.device) -> None:
    """Treat CUDA default-device aliases as equivalent during device placement checks."""
    assert actual.type == expected.type
    if actual.type == "cuda":
        assert actual.index in (None, 0)
        assert expected.index in (None, 0)
    else:
        assert actual.index == expected.index


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


def _make_trainer_with_actors(n_actors: int) -> PpoTrainer:
    """Create a canonical PpoTrainer with a configurable actor count for overlap tests."""
    cfg = ActorConfig(
        sub_address="tcp://127.0.0.1:19000",
        pub_address="tcp://127.0.0.1:19001",
        step_endpoint="tcp://127.0.0.1:19002",
        mode="step",
        embedding_dim=64,
        n_actors=n_actors,
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
        assert torch.equal(policy_sd[key], trainer2._learner_policy.state_dict()[key]), (
            f"Policy param mismatch: {key}"
        )

    # Verify RND weights match (including frozen target)
    for key in rnd_sd:
        assert torch.equal(rnd_sd[key], trainer2._rnd.state_dict()[key]), (
            f"RND param mismatch: {key}"
        )

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
    trainer._telemetry_queue = queue.Queue()

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

    topic, payload, label = trainer._telemetry_queue.get_nowait()
    assert topic == TOPIC_DISTANCE_MATRIX
    assert label == "observation"
    raw = payload() if callable(payload) else payload
    restored = deserialize(raw)
    assert isinstance(restored, DistanceMatrix)
    assert restored.step_id == 7


def test_publish_dashboard_observations_emits_selected_frames() -> None:
    """Dashboard observation publication should emit only actor 0 frames."""
    trainer = _make_trainer()

    trainer._telemetry_queue = queue.Queue()

    observations = {
        actor_id: DistanceMatrix(
            episode_id=1,
            env_ids=np.array([actor_id], dtype=np.int32),
            matrix_shape=(8, 4),
            depth=np.ones((1, 8, 4), dtype=np.float32),
            delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
            semantic=np.zeros((1, 8, 4), dtype=np.int32),
            valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
            overhead=np.zeros((8, 8, 3), dtype=np.float32),
            robot_pose=RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0),
            step_id=actor_id + 10,
            timestamp=1.0,
        )
        for actor_id in (0, 1, 2)
    }

    trainer._publish_dashboard_observations(observations)

    restored_env_ids = []
    for _ in range(3):
        topic, payload, label = trainer._telemetry_queue.get_nowait()
        assert topic == TOPIC_DISTANCE_MATRIX
        assert label == "observation"
        raw = payload() if callable(payload) else payload
        restored = deserialize(raw)
        assert isinstance(restored, DistanceMatrix)
        restored_env_ids.append(int(restored.env_ids[0]))
    assert restored_env_ids == [0, 1, 2]


def test_dashboard_publish_gate_uses_configured_live_cadence() -> None:
    """Selected-actor observation publication should use the configured cadence."""
    trainer = _make_trainer()
    trainer._config.dashboard_observation_hz = 10.0
    trainer._last_dashboard_heartbeat_at = 100.0

    perf_values = iter((100.05, 100.11))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "navi_actor.training.ppo_trainer.time.perf_counter", lambda: next(perf_values)
    )
    try:
        assert trainer._should_publish_dashboard_observation() is False
        assert trainer._should_publish_dashboard_observation() is True
    finally:
        monkeypatch.undo()


def test_reward_ema_batch_update_matches_sequential_actor_order() -> None:
    """Vectorized reward EMA should preserve the historic per-actor update order."""
    trainer = _make_trainer()
    rewards = torch.tensor([1.25, -0.5, 0.75, 0.0], dtype=torch.float32, device=trainer._device)
    initial_ema = torch.tensor(0.33, dtype=torch.float32, device=trainer._device)

    expected = float(initial_ema.detach().to(device="cpu", dtype=torch.float32))
    for reward in rewards.detach().to(device="cpu", dtype=torch.float32).tolist():
        expected = 0.01 * float(reward) + 0.99 * expected

    actual = trainer._update_reward_ema(initial_ema, rewards)

    assert float(actual.detach().to(device="cpu", dtype=torch.float32)) == pytest.approx(expected)


def test_active_actor_local_indices_preserve_selected_actor_order() -> None:
    trainer = _make_trainer()

    active_actor_indices = torch.tensor([3, 1], dtype=torch.int64, device=trainer._device)
    selected_actor_indices = torch.tensor([1, 2, 3], dtype=torch.int64, device=trainer._device)

    local_indices = trainer._active_actor_local_indices(
        active_actor_indices, selected_actor_indices
    )

    assert torch.equal(
        local_indices, torch.tensor([1, 0], dtype=torch.int64, device=trainer._device)
    )


def test_run_ppo_update_bootstrap_ignores_hidden_state_on_canonical_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical PPO bootstrap should not feed hidden state into the learner policy."""
    trainer = _make_trainer()
    n = trainer._n_actors
    obs = torch.randn(n, 3, 128, 24, device=trainer._device)
    aux = torch.zeros((n, 3), dtype=torch.float32, device=trainer._device)
    buffer = MultiTrajectoryBuffer(
        n_actors=n,
        gamma=trainer._config.gamma,
        gae_lambda=trainer._config.gae_lambda,
        capacity=1,
    )
    buffer.append_batch(
        observations=obs,
        actions=torch.zeros((n, 4), dtype=torch.float32, device=trainer._device),
        log_probs=torch.zeros((n,), dtype=torch.float32, device=trainer._device),
        values=torch.zeros((n,), dtype=torch.float32, device=trainer._device),
        rewards=torch.zeros((n,), dtype=torch.float32, device=trainer._device),
        dones=torch.zeros((n,), dtype=torch.bool, device=trainer._device),
        truncateds=torch.zeros((n,), dtype=torch.bool, device=trainer._device),
        aux_tensors=aux,
    )

    captured: dict[str, torch.Tensor | None] = {"hidden": None}

    original_forward = trainer._learner_policy.forward

    def recording_forward(
        observation: torch.Tensor,
        hidden: torch.Tensor | None = None,
        *,
        aux_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        captured["hidden"] = hidden
        return original_forward(observation, hidden, aux_tensor=aux_tensor)

    monkeypatch.setattr(trainer._learner_policy, "forward", recording_forward)

    def fake_train_ppo_epoch(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return None

    monkeypatch.setattr(trainer._learner, "train_ppo_epoch", fake_train_ppo_epoch)

    trainer._run_ppo_update(
        multi_buffer=buffer,
        obs_batch=obs,
        aux_batch=aux,
        ppo_epochs=1,
        minibatch_size=n,
        seq_len=0,
    )

    assert captured["hidden"] is None
    assert len(buffer) == 0


def test_selected_episode_telemetry_actor_indices_skip_when_training_telemetry_disabled() -> None:
    """Done-actor host extraction should be skipped when episode telemetry is off."""
    trainer = _make_trainer()
    trainer._config.emit_training_telemetry = False
    trainer._pub_socket = object()  # type: ignore[assignment]
    done_indices = torch.tensor([0, 2, 3], dtype=torch.int64, device=trainer._device)

    selected = trainer._selected_episode_telemetry_actor_indices(done_indices)

    assert selected.numel() == 0


def test_selected_episode_telemetry_actor_indices_filter_to_selected_actor() -> None:
    """Default sparse episode telemetry should only materialize actor 0."""
    trainer = _make_trainer()
    trainer._config.emit_training_telemetry = True
    trainer._pub_socket = object()  # type: ignore[assignment]
    done_indices = torch.tensor([0, 2, 3], dtype=torch.int64, device=trainer._device)

    selected = trainer._selected_episode_telemetry_actor_indices(done_indices)

    assert torch.equal(selected, torch.tensor([0], dtype=torch.int64, device=trainer._device))


def test_selected_step_telemetry_actor_indices_skip_when_training_telemetry_disabled() -> None:
    """Per-step telemetry host extraction should be skipped when training telemetry is off."""
    trainer = _make_trainer()
    trainer._config.emit_training_telemetry = False

    selected = trainer._selected_step_telemetry_actor_indices()

    assert selected.numel() == 0


def test_selected_step_telemetry_actor_indices_filter_to_selected_actor() -> None:
    """Default sparse step telemetry should mirror only actor 0."""
    trainer = _make_trainer()
    trainer._config.emit_training_telemetry = True

    selected = trainer._selected_step_telemetry_actor_indices()

    assert torch.equal(selected, torch.tensor([0], dtype=torch.int64, device=trainer._device))


def test_extract_host_rollout_scalars_uses_selected_actor_subset() -> None:
    """Sparse rollout scalar extraction should select only the requested actor subset on device."""
    trainer = _make_trainer()
    batch = trainer._extract_host_rollout_scalars(
        raw_rewards=torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=trainer._device
        ),
        shaped_rewards=torch.tensor(
            [1.5, 2.5, 3.5, 4.5], dtype=torch.float32, device=trainer._device
        ),
        intrinsic_rewards=torch.tensor(
            [0.1, 0.2, 0.3, 0.4], dtype=torch.float32, device=trainer._device
        ),
        loop_similarities=torch.tensor(
            [0.0, 0.9, 0.2, 0.8], dtype=torch.float32, device=trainer._device
        ),
        done_flags=torch.tensor(
            [False, True, False, True], dtype=torch.bool, device=trainer._device
        ),
        actor_indices=torch.tensor([1, 3], dtype=torch.int64, device=trainer._device),
    )

    assert batch.raw_reward_tensor is not None
    assert batch.shaped_reward_tensor is not None
    assert batch.intrinsic_reward_tensor is not None
    assert batch.loop_similarity_tensor is not None
    assert batch.done_tensor is not None
    _assert_same_device(batch.raw_reward_tensor.device, trainer._device)
    _assert_same_device(batch.shaped_reward_tensor.device, trainer._device)
    _assert_same_device(batch.intrinsic_reward_tensor.device, trainer._device)
    _assert_same_device(batch.loop_similarity_tensor.device, trainer._device)
    _assert_same_device(batch.done_tensor.device, trainer._device)
    assert torch.equal(
        batch.raw_reward_tensor,
        torch.tensor([2.0, 4.0], dtype=torch.float32, device=trainer._device),
    )
    assert torch.equal(
        batch.shaped_reward_tensor,
        torch.tensor([2.5, 4.5], dtype=torch.float32, device=trainer._device),
    )
    assert torch.equal(
        batch.intrinsic_reward_tensor,
        torch.tensor([0.2, 0.4], dtype=torch.float32, device=trainer._device),
    )
    assert torch.equal(
        batch.loop_similarity_tensor,
        torch.tensor([0.9, 0.8], dtype=torch.float32, device=trainer._device),
    )
    assert torch.equal(
        batch.done_tensor, torch.tensor([True, True], dtype=torch.bool, device=trainer._device)
    )


def test_extract_completed_episode_host_batch_uses_selected_actor_subset() -> None:
    """Completed-episode host extraction should only mirror the requested actor subset."""
    trainer = _make_trainer()
    episode_returns = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=trainer._device
    )
    episode_lengths = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=trainer._device)

    batch = trainer._extract_completed_episode_host_batch(
        actor_indices=torch.tensor([1, 3], dtype=torch.int64, device=trainer._device),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
    )

    assert batch is not None
    assert torch.equal(batch.actor_id_tensor, torch.tensor([1, 3], dtype=torch.int64))
    assert torch.equal(batch.episode_return_tensor, torch.tensor([2.0, 4.0], dtype=torch.float32))
    assert torch.equal(batch.episode_length_tensor, torch.tensor([20, 40], dtype=torch.int32))


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

    trainer._telemetry_queue = queue.Queue()
    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    trainer._publish_runtime_perf(step_id=200)

    topic, payload, label = trainer._telemetry_queue.get_nowait()
    assert topic == TOPIC_TELEMETRY_EVENT
    assert label == "runtime perf telemetry"
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
            self.env_id_tensor = torch.tensor([3], dtype=torch.int64, device=trainer._device)
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
            actor_indices: torch.Tensor | None = None,
            scratch_slot: int = 0,
            publish_actor_ids: tuple[int, ...] = (),
            materialize_results: bool = False,
        ) -> tuple[FakeTensorBatch, tuple[object, ...]]:
            del publish_actor_ids, scratch_slot
            assert action_tensor.shape == (1, 4)
            assert step_id == 19
            assert actor_indices is not None
            assert materialize_results is False
            assert torch.equal(
                actor_indices, torch.tensor([3], dtype=torch.int64, device=trainer._device)
            )
            return FakeTensorBatch(), ()

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    payload = trainer._request_batch_step_tensor_actions(
        torch.ones((1, 4), device="cpu", dtype=torch.float32),
        19,
        actor_indices=torch.tensor([3], device=trainer._device, dtype=torch.int64),
        publish_actor_ids=(0,),
    )

    step_batch = payload
    assert step_batch.observation_tensor.shape == (1, 3, 8, 4)
    assert torch.allclose(
        step_batch.reward_tensor, torch.tensor([1.25], dtype=torch.float32, device=trainer._device)
    )
    assert torch.equal(
        step_batch.done_tensor, torch.tensor([False], dtype=torch.bool, device=trainer._device)
    )
    assert torch.equal(
        step_batch.truncated_tensor, torch.tensor([True], dtype=torch.bool, device=trainer._device)
    )
    assert torch.equal(
        step_batch.episode_id_tensor, torch.tensor([5], dtype=torch.int64, device=trainer._device)
    )
    assert torch.equal(
        step_batch.env_id_tensor, torch.tensor([3], dtype=torch.int64, device=trainer._device)
    )
    assert step_batch.published_observations == {}


def test_finalize_rollout_group_preserves_actor_local_transition_order() -> None:
    """Finalize must append the old observation while updating current state to the new observation."""
    trainer = _make_trainer_with_actors(4)
    trainer._config.emit_training_telemetry = False
    trainer._config.emit_observation_stream = False
    trainer._config.enable_episodic_memory = False
    trainer._config.enable_reward_shaping = False

    captured: dict[str, torch.Tensor] = {}

    def capture_append_batch(**kwargs: torch.Tensor) -> None:
        captured["observations"] = kwargs["observations"].detach().clone()
        captured["actor_indices"] = kwargs["actor_indices"].detach().clone()

    trainer._multi_buffers.append_batch = capture_append_batch  # type: ignore[method-assign]

    actor_indices = torch.tensor([2, 0], dtype=torch.int64, device=trainer._device)
    old_observations = torch.arange(
        2 * 3 * 8 * 4, dtype=torch.float32, device=trainer._device
    ).reshape(2, 3, 8, 4)
    next_observations = torch.full((2, 3, 8, 4), 77.0, dtype=torch.float32, device=trainer._device)
    current_obs_batch = torch.zeros(
        (trainer._n_actors, 3, 8, 4), dtype=torch.float32, device=trainer._device
    )
    aux_states = torch.zeros((trainer._n_actors, 3), dtype=torch.float32, device=trainer._device)
    reward_sum = torch.zeros((), dtype=torch.float32, device=trainer._device)
    reward_ema = torch.zeros((), dtype=torch.float32, device=trainer._device)
    episode_returns = torch.zeros(
        (trainer._n_actors,), dtype=torch.float32, device=trainer._device
    )
    episode_lengths = torch.zeros((trainer._n_actors,), dtype=torch.int32, device=trainer._device)

    class FakeTensorBatch:
        def __init__(self) -> None:
            self.observation_tensor = next_observations
            self.reward_tensor = torch.tensor(
                [1.0, 2.0], dtype=torch.float32, device=trainer._device
            )
            self.done_tensor = torch.tensor(
                [False, False], dtype=torch.bool, device=trainer._device
            )
            self.truncated_tensor = torch.tensor(
                [False, False], dtype=torch.bool, device=trainer._device
            )
            self.episode_id_tensor = torch.tensor(
                [9, 10], dtype=torch.int64, device=trainer._device
            )
            self.env_id_tensor = actor_indices
            self.reward_component_tensor = None
            self.published_observations: dict[int, DistanceMatrix] = {}

    pending = _PendingRolloutGroup(
        group_slot=0,
        actor_indices=actor_indices,
        observations=old_observations,
        aux_tensor=torch.zeros((2, 3), dtype=torch.float32, device=trainer._device),
        actions=torch.zeros((2, 4), dtype=torch.float32, device=trainer._device),
        embeddings=torch.zeros(
            (2, trainer._config.embedding_dim), dtype=torch.float32, device=trainer._device
        ),
        log_probs=torch.zeros((2,), dtype=torch.float32, device=trainer._device),
        values=torch.zeros((2,), dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.zeros((2,), dtype=torch.float32, device=trainer._device),
        step_batch=FakeTensorBatch(),
        fwd_ms=0.0,
        step_ms=0.0,
    )

    trainer._finalize_rollout_group(
        pending=pending,
        current_obs_batch=current_obs_batch,
        aux_states=aux_states,
        reward_sum_tensor=reward_sum,
        reward_ema=reward_ema,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        step_id=100,
        publish_step_telemetry=False,
        publish_observations=False,
    )

    assert torch.equal(captured["observations"], old_observations)
    assert torch.equal(captured["actor_indices"], actor_indices)
    assert torch.equal(current_obs_batch.index_select(0, actor_indices), next_observations)


def test_prepare_rollout_group_reads_updated_group_state_after_finalize() -> None:
    """Preparing a group after finalize should observe the latest per-actor observation rows."""
    trainer = _make_trainer_with_actors(4)
    trainer._config.emit_training_telemetry = False
    trainer._config.emit_observation_stream = False
    trainer._config.enable_episodic_memory = False
    trainer._config.enable_reward_shaping = False

    actor_indices = trainer._rollout_group_indices[0]
    current_obs_batch = torch.zeros(
        (trainer._n_actors, 3, 8, 4), dtype=torch.float32, device=trainer._device
    )
    aux_states = torch.zeros((trainer._n_actors, 3), dtype=torch.float32, device=trainer._device)
    updated_rows = torch.full(
        (int(actor_indices.numel()), 3, 8, 4), 11.0, dtype=torch.float32, device=trainer._device
    )

    trainer._write_actor_rows(current_obs_batch, actor_indices, updated_rows)
    prepared = trainer._prepare_rollout_group(
        group_slot=0,
        actor_indices=actor_indices,
        current_obs_batch=current_obs_batch,
        aux_states=aux_states,
    )

    assert torch.equal(prepared.observations, updated_rows)


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
            materialize_results: bool = False,
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids, materialize_results
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
            materialize_results: bool = False,
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids, materialize_results
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
                    robot_pose=RobotPose(
                        x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0
                    ),
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


def test_observation_publish_actor_ids_defaults_to_selected_actor_only() -> None:
    """Canonical trainer should default observation publication to actor 0."""
    trainer = _make_trainer()
    trainer._n_actors = 4
    trainer._config.emit_observation_stream = False
    assert trainer._observation_publish_actor_ids() == ()

    trainer._config.emit_observation_stream = True
    assert trainer._observation_publish_actor_ids() == (0,)


def test_load_initial_observation_batch_skips_materialization_when_stream_disabled() -> None:
    """Initial tensor resets should not request published observations when streaming is off."""
    trainer = _make_trainer()
    trainer._config.emit_observation_stream = False
    materialize_calls: list[bool] = []

    class FakeRuntime:
        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            publish_actor_ids: tuple[int, ...] = (),
            materialize_results: bool = False,
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids, materialize_results
            raise AssertionError("not used")

        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            del episode_id, actor_id
            materialize_calls.append(materialize)
            return torch.ones((3, 8, 4), device=trainer._device), None

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    obs_batch, published = trainer._load_initial_observation_batch()

    assert obs_batch.shape == (1, 3, 8, 4)
    assert published == {}
    assert materialize_calls == [False]


def test_load_initial_observation_batch_preserves_oracle_house_tensor_and_public_frame() -> None:
    trainer = _make_trainer()
    trainer._config.emit_observation_stream = True

    oracle = house_observation()
    observation_tensor = torch.from_numpy(
        np.stack(
            [
                oracle.depth,
                oracle.semantic.astype(np.float32),
                oracle.valid.astype(np.float32),
            ],
            axis=0,
        )
    ).to(device=trainer._device, dtype=torch.float32)

    class FakeRuntime:
        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            publish_actor_ids: tuple[int, ...] = (),
            materialize_results: bool = False,
        ) -> tuple[object, tuple[object, ...]]:
            del action_tensor, step_id, publish_actor_ids, materialize_results
            raise AssertionError("not used")

        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            published = None
            if materialize:
                published = DistanceMatrix(
                    episode_id=episode_id,
                    env_ids=np.array([actor_id], dtype=np.int32),
                    matrix_shape=oracle.depth.shape,
                    depth=oracle.depth[np.newaxis, ...],
                    delta_depth=np.zeros_like(oracle.depth[np.newaxis, ...]),
                    semantic=oracle.semantic[np.newaxis, ...],
                    valid_mask=oracle.valid[np.newaxis, ...],
                    overhead=np.zeros((8, 8, 3), dtype=np.float32),
                    robot_pose=RobotPose(
                        x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=1.0
                    ),
                    step_id=0,
                    timestamp=1.0,
                )
            return observation_tensor, published

        def perf_snapshot(self) -> object:
            raise AssertionError("not used")

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime()  # type: ignore[assignment]

    obs_batch, published = trainer._load_initial_observation_batch()

    np.testing.assert_allclose(obs_batch[0, 0].detach().cpu().numpy(), oracle.depth)
    np.testing.assert_allclose(
        obs_batch[0, 1].detach().cpu().numpy(), oracle.semantic.astype(np.float32)
    )
    np.testing.assert_allclose(
        obs_batch[0, 2].detach().cpu().numpy(), oracle.valid.astype(np.float32)
    )
    np.testing.assert_allclose(published[0].depth[0], oracle.depth)
    np.testing.assert_array_equal(published[0].valid_mask[0], oracle.valid)


def test_extract_host_rollout_scalars_handles_oracle_motion_delta_without_host_shape_drift() -> (
    None
):
    first = house_observation()
    second = house_observation_after_forward_motion()

    delta = second.depth - first.depth
    assert delta.shape == first.depth.shape
    assert float(delta.min()) < 0.0


def test_extract_host_rollout_scalars_skips_telemetry_columns_when_not_needed() -> None:
    """Canonical trainer should skip rollout scalar host mirrors when step telemetry is off."""
    trainer = _make_trainer()

    payload = trainer._extract_host_rollout_scalars(
        raw_rewards=torch.tensor([0.5, 1.5], dtype=torch.float32, device=trainer._device),
        shaped_rewards=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
        loop_similarities=torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
        done_flags=torch.tensor([True, False], dtype=torch.bool, device=trainer._device),
        actor_indices=None,
    )

    assert payload.raw_reward_tensor is None
    assert payload.shaped_reward_tensor is None
    assert payload.intrinsic_reward_tensor is None
    assert payload.loop_similarity_tensor is None
    assert payload.done_tensor is None


def test_extract_host_rollout_scalars_batches_telemetry_columns_when_needed() -> None:
    """Canonical trainer should keep selected telemetry scalars on device until final materialization."""
    trainer = _make_trainer()

    payload = trainer._extract_host_rollout_scalars(
        raw_rewards=torch.tensor([0.5, 1.5], dtype=torch.float32, device=trainer._device),
        shaped_rewards=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
        loop_similarities=torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
        done_flags=torch.tensor([True, False], dtype=torch.bool, device=trainer._device),
        actor_indices=torch.tensor([0, 1], dtype=torch.int64, device=trainer._device),
    )

    assert payload.raw_reward_tensor is not None
    assert payload.intrinsic_reward_tensor is not None
    assert payload.loop_similarity_tensor is not None
    assert payload.done_tensor is not None
    _assert_same_device(payload.raw_reward_tensor.device, trainer._device)
    _assert_same_device(payload.shaped_reward_tensor.device, trainer._device)
    _assert_same_device(payload.intrinsic_reward_tensor.device, trainer._device)
    _assert_same_device(payload.loop_similarity_tensor.device, trainer._device)
    _assert_same_device(payload.done_tensor.device, trainer._device)
    assert torch.allclose(
        payload.raw_reward_tensor,
        torch.tensor([0.5, 1.5], dtype=torch.float32, device=trainer._device),
    )
    assert torch.allclose(
        payload.shaped_reward_tensor,
        torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
    )
    assert torch.allclose(
        payload.intrinsic_reward_tensor,
        torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
    )
    assert torch.allclose(
        payload.loop_similarity_tensor,
        torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
    )
    assert torch.equal(
        payload.done_tensor, torch.tensor([True, False], dtype=torch.bool, device=trainer._device)
    )


def test_materialize_step_telemetry_host_batch_packs_selected_columns_once() -> None:
    """Step telemetry host materialization should batch selected scalars, actions, and reward components once."""
    trainer = _make_trainer()

    scalar_batch = trainer._extract_host_rollout_scalars(
        raw_rewards=torch.tensor([0.5, 1.5], dtype=torch.float32, device=trainer._device),
        shaped_rewards=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        intrinsic_rewards=torch.tensor([3.0, 4.0], dtype=torch.float32, device=trainer._device),
        loop_similarities=torch.tensor([5.0, 6.0], dtype=torch.float32, device=trainer._device),
        done_flags=torch.tensor([True, False], dtype=torch.bool, device=trainer._device),
        actor_indices=torch.tensor([0, 1], dtype=torch.int64, device=trainer._device),
    )

    payload = trainer._materialize_step_telemetry_host_batch(
        scalar_batch=scalar_batch,
        action_tensor=torch.tensor(
            [[7.0, 0.0, 0.0, 8.0], [9.0, 0.0, 0.0, 10.0]],
            dtype=torch.float32,
            device=trainer._device,
        ),
        reward_component_tensor=torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=torch.float32,
            device=trainer._device,
        ),
        actor_indices=torch.tensor([0, 1], dtype=torch.int64, device=trainer._device),
    )

    assert payload.raw_reward_tensor is not None
    assert payload.shaped_reward_tensor is not None
    assert payload.intrinsic_reward_tensor is not None
    assert payload.loop_similarity_tensor is not None
    assert payload.done_tensor is not None
    assert payload.forward_velocity_tensor is not None
    assert payload.yaw_velocity_tensor is not None
    assert payload.reward_component_tensor is not None
    assert payload.raw_reward_tensor.device.type == "cpu"
    assert payload.forward_velocity_tensor.device.type == "cpu"
    assert payload.reward_component_tensor.device.type == "cpu"
    assert torch.allclose(payload.raw_reward_tensor, torch.tensor([0.5, 1.5], dtype=torch.float32))
    assert torch.allclose(
        payload.shaped_reward_tensor, torch.tensor([1.0, 2.0], dtype=torch.float32)
    )
    assert torch.allclose(
        payload.intrinsic_reward_tensor, torch.tensor([3.0, 4.0], dtype=torch.float32)
    )
    assert torch.allclose(
        payload.loop_similarity_tensor, torch.tensor([5.0, 6.0], dtype=torch.float32)
    )
    assert torch.equal(payload.done_tensor, torch.tensor([True, False], dtype=torch.bool))
    assert torch.allclose(
        payload.forward_velocity_tensor, torch.tensor([7.0, 9.0], dtype=torch.float32)
    )
    assert torch.allclose(
        payload.yaw_velocity_tensor, torch.tensor([8.0, 10.0], dtype=torch.float32)
    )
    assert torch.allclose(
        payload.reward_component_tensor,
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32),
    )


def test_extract_completed_episode_host_batch_returns_none_when_no_actor_finished() -> None:
    """Canonical trainer should skip sparse episode extraction when nothing finished."""
    trainer = _make_trainer()

    payload = trainer._extract_completed_episode_host_batch(
        actor_indices=torch.tensor([], dtype=torch.int64, device=trainer._device),
        episode_returns=torch.tensor([1.0, 2.0], dtype=torch.float32, device=trainer._device),
        episode_lengths=torch.tensor([3, 4], dtype=torch.int32, device=trainer._device),
    )

    assert payload is None


def test_extract_completed_episode_host_batch_packs_done_events_once() -> None:
    """Canonical trainer should batch sparse completed-episode host extraction in one packed copy."""
    trainer = _make_trainer()

    payload = trainer._extract_completed_episode_host_batch(
        actor_indices=torch.tensor([1, 2], dtype=torch.int64, device=trainer._device),
        episode_returns=torch.tensor([1.0, 2.5, 3.5], dtype=torch.float32, device=trainer._device),
        episode_lengths=torch.tensor([10, 20, 30], dtype=torch.int32, device=trainer._device),
    )

    assert payload is not None
    assert payload.actor_id_tensor.device.type == "cpu"
    assert payload.episode_return_tensor.device.type == "cpu"
    assert payload.episode_length_tensor.device.type == "cpu"
    assert torch.equal(payload.actor_id_tensor, torch.tensor([1, 2], dtype=torch.int64))
    assert torch.allclose(
        payload.episode_return_tensor, torch.tensor([2.5, 3.5], dtype=torch.float32)
    )
    assert torch.equal(payload.episode_length_tensor, torch.tensor([20, 30], dtype=torch.int32))


def test_train_stops_after_requested_bounded_actor_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical trainer should stop after the requested total actor-step budget."""

    cfg = ActorConfig(
        sub_address="tcp://127.0.0.1:19100",
        pub_address="tcp://127.0.0.1:19101",
        step_endpoint="tcp://127.0.0.1:19102",
        mode="step",
        azimuth_bins=8,
        elevation_bins=4,
        embedding_dim=32,
        n_actors=4,
        rollout_length=32,
        ppo_epochs=1,
        minibatch_size=8,
        bptt_len=4,
        enable_episodic_memory=False,
        enable_reward_shaping=False,
        emit_observation_stream=False,
        emit_training_telemetry=False,
        emit_perf_telemetry=False,
        rollout_overlap_groups=1,
        stagger_ppo=False,
    )
    trainer = PpoTrainer(cfg)

    class FakeTensorStepBatch:
        def __init__(self, device: torch.device, actor_indices: torch.Tensor) -> None:
            actor_count = int(actor_indices.shape[0])
            self.observation_tensor = torch.ones(
                (actor_count, 3, 8, 4), dtype=torch.float32, device=device
            )
            self.reward_tensor = torch.ones((actor_count,), dtype=torch.float32, device=device)
            self.done_tensor = torch.zeros((actor_count,), dtype=torch.bool, device=device)
            self.truncated_tensor = torch.zeros((actor_count,), dtype=torch.bool, device=device)
            self.episode_id_tensor = torch.zeros((actor_count,), dtype=torch.int64, device=device)
            self.env_id_tensor = actor_indices.clone()
            self.reward_component_tensor = None
            self.published_observations: dict[int, DistanceMatrix] = {}

    class FakeRuntime:
        def __init__(self, device: torch.device) -> None:
            self._device = device
            self.step_calls = 0

        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            del episode_id, actor_id, materialize
            return torch.ones((3, 8, 4), dtype=torch.float32, device=self._device), None

        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            actor_indices: torch.Tensor | None = None,
            scratch_slot: int = 0,
            publish_actor_ids: tuple[int, ...] = (),
            materialize_results: bool = False,
        ) -> tuple[FakeTensorStepBatch, tuple[object, ...]]:
            del publish_actor_ids, scratch_slot
            assert actor_indices is not None
            assert materialize_results is False
            actor_count = int(actor_indices.shape[0])
            assert action_tensor.shape == (actor_count, 4)
            self.step_calls += 1
            return FakeTensorStepBatch(self._device, actor_indices), ()

        def perf_snapshot(self) -> object:
            return type(
                "Snapshot",
                (),
                {
                    "sps": 0.0,
                    "last_batch_step_ms": 0.0,
                    "ema_batch_step_ms": 0.0,
                    "avg_batch_step_ms": 0.0,
                    "avg_actor_step_ms": 0.0,
                    "total_batches": self.step_calls,
                    "total_actor_steps": self.step_calls * 4,
                },
            )()

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime(trainer._device)  # type: ignore[assignment]

    def fake_forward(
        obs_tensor: torch.Tensor,
        hidden: torch.Tensor | None = None,
        aux_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
        del hidden, aux_tensor
        actor_count = obs_tensor.shape[0]
        actions = torch.zeros((actor_count, 4), dtype=torch.float32, device=trainer._device)
        log_probs = torch.zeros((actor_count,), dtype=torch.float32, device=trainer._device)
        values = torch.zeros((actor_count,), dtype=torch.float32, device=trainer._device)
        embeddings = torch.zeros(
            (actor_count, trainer._config.embedding_dim),
            dtype=torch.float32,
            device=trainer._device,
        )
        return actions, log_probs, values, None, embeddings

    update_calls: list[int] = []

    monkeypatch.setattr(trainer._rollout_policy, "forward", fake_forward)
    monkeypatch.setattr(
        trainer._rnd,
        "intrinsic_reward",
        lambda z_t: torch.zeros((z_t.shape[0],), dtype=torch.float32, device=trainer._device),
    )
    monkeypatch.setattr(trainer, "_publish_dashboard_observations", lambda observations: None)
    monkeypatch.setattr(trainer, "_should_publish_dashboard_observation", lambda: False)
    monkeypatch.setattr(
        trainer, "_publish_update_telemetry", lambda step_id, reward_ema, metrics: None
    )
    monkeypatch.setattr(
        trainer,
        "_run_ppo_update",
        lambda **kwargs: update_calls.append(len(kwargs["multi_buffer"])),
    )

    metrics = trainer.train(total_steps=8, log_every=0)

    assert metrics.total_steps == 8
    assert trainer._runtime.step_calls == 2  # type: ignore[union-attr]
    assert update_calls == [8]


def test_train_records_coarse_resource_metrics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Canonical trainer metrics should include coarse process/CUDA resource snapshots."""

    run_root = tmp_path / "run"
    metrics_root = run_root / "metrics"
    manifest_root = run_root / "manifests"
    log_root = run_root / "logs"
    for directory in (metrics_root, manifest_root, log_root):
        directory.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("NAVI_RUN_ID", "test-trainer-metrics")
    monkeypatch.setenv("NAVI_RUN_ROOT", str(run_root))
    monkeypatch.setenv("NAVI_METRICS_ROOT", str(metrics_root))
    monkeypatch.setenv("NAVI_MANIFEST_ROOT", str(manifest_root))
    monkeypatch.setenv("NAVI_LOG_ROOT", str(log_root))
    monkeypatch.setenv("NAVI_REPO_ROOT", str(Path(__file__).resolve().parents[4]))
    monkeypatch.setenv("NAVI_RUN_STARTED_AT", "2026-03-18T00:00:00+00:00")

    cfg = ActorConfig(
        sub_address="tcp://127.0.0.1:19200",
        pub_address="tcp://127.0.0.1:19201",
        step_endpoint="tcp://127.0.0.1:19202",
        mode="step",
        azimuth_bins=8,
        elevation_bins=4,
        embedding_dim=32,
        n_actors=4,
        rollout_length=32,
        ppo_epochs=1,
        minibatch_size=8,
        bptt_len=4,
        enable_episodic_memory=False,
        enable_reward_shaping=False,
        emit_observation_stream=False,
        emit_training_telemetry=False,
        emit_perf_telemetry=True,
        stagger_ppo=False,
    )
    trainer = PpoTrainer(cfg)

    class FakeTensorStepBatch:
        def __init__(self, device: torch.device, actor_indices: torch.Tensor) -> None:
            actor_count = int(actor_indices.shape[0])
            self.observation_tensor = torch.ones(
                (actor_count, 3, 8, 4), dtype=torch.float32, device=device
            )
            self.reward_tensor = torch.ones((actor_count,), dtype=torch.float32, device=device)
            self.done_tensor = torch.zeros((actor_count,), dtype=torch.bool, device=device)
            self.truncated_tensor = torch.zeros((actor_count,), dtype=torch.bool, device=device)
            self.episode_id_tensor = torch.zeros((actor_count,), dtype=torch.int64, device=device)
            self.env_id_tensor = actor_indices.clone()
            self.reward_component_tensor = None
            self.published_observations: dict[int, DistanceMatrix] = {}

    class FakeRuntime:
        def __init__(self, device: torch.device) -> None:
            self._device = device
            self.step_calls = 0

        def reset_tensor(
            self,
            episode_id: int,
            *,
            actor_id: int = 0,
            materialize: bool = False,
        ) -> tuple[torch.Tensor, DistanceMatrix | None]:
            del episode_id, actor_id, materialize
            return torch.ones((3, 8, 4), dtype=torch.float32, device=self._device), None

        def batch_step_tensor_actions(
            self,
            action_tensor: torch.Tensor,
            step_id: int,
            *,
            actor_indices: torch.Tensor | None = None,
            scratch_slot: int = 0,
            publish_actor_ids: tuple[int, ...] = (),
            materialize_results: bool = False,
        ) -> tuple[FakeTensorStepBatch, tuple[object, ...]]:
            del step_id, publish_actor_ids, scratch_slot
            assert actor_indices is not None
            assert materialize_results is False
            actor_count = int(actor_indices.shape[0])
            assert action_tensor.shape == (actor_count, 4)
            self.step_calls += 1
            return FakeTensorStepBatch(self._device, actor_indices), ()

        def perf_snapshot(self) -> object:
            return type(
                "Snapshot",
                (),
                {
                    "sps": 42.0,
                    "last_batch_step_ms": 2.5,
                    "ema_batch_step_ms": 2.0,
                    "avg_batch_step_ms": 2.25,
                    "avg_actor_step_ms": 0.56,
                    "total_batches": self.step_calls,
                    "total_actor_steps": self.step_calls * 4,
                },
            )()

        def close(self) -> None:
            return None

    trainer._runtime = FakeRuntime(trainer._device)  # type: ignore[assignment]

    def fake_forward(
        obs_tensor: torch.Tensor,
        hidden: torch.Tensor | None = None,
        aux_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, torch.Tensor]:
        del hidden, aux_tensor
        actor_count = obs_tensor.shape[0]
        actions = torch.zeros((actor_count, 4), dtype=torch.float32, device=trainer._device)
        log_probs = torch.zeros((actor_count,), dtype=torch.float32, device=trainer._device)
        values = torch.zeros((actor_count,), dtype=torch.float32, device=trainer._device)
        embeddings = torch.zeros(
            (actor_count, trainer._config.embedding_dim),
            dtype=torch.float32,
            device=trainer._device,
        )
        return actions, log_probs, values, None, embeddings

    monkeypatch.setattr(trainer._rollout_policy, "forward", fake_forward)
    monkeypatch.setattr(
        trainer._rnd,
        "intrinsic_reward",
        lambda z_t: torch.zeros((z_t.shape[0],), dtype=torch.float32, device=trainer._device),
    )
    monkeypatch.setattr(trainer, "_publish_dashboard_observations", lambda observations: None)
    monkeypatch.setattr(trainer, "_should_publish_dashboard_observation", lambda: False)
    monkeypatch.setattr(
        trainer, "_publish_update_telemetry", lambda step_id, reward_ema, metrics: None
    )
    monkeypatch.setattr(
        trainer,
        "_run_ppo_update",
        lambda **kwargs: setattr(
            trainer,
            "_last_opt_metrics",
            type(
                "Metrics",
                (),
                {
                    "n_updates": 1,
                    "policy_eval_ms_total": 1.0,
                    "backward_ms_total": 1.0,
                    "optimizer_step_ms_total": 1.0,
                    "rnd_step_ms_total": 0.0,
                    "epoch_total_ms": 3.0,
                },
            )(),
        ),
    )

    metrics = trainer.train(total_steps=8, log_every=4)

    assert metrics.total_steps == 8
    metrics_path = metrics_root / "actor_training.jsonl"
    assert metrics_path.exists()
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
    streams = {record["stream"] for record in records}
    assert "training_perf" in streams
    assert "runtime_perf" in streams
    assert "training_summary" in streams

    perf_payload = next(record["payload"] for record in records if record["stream"] == "training_perf")
    assert perf_payload["operation"] == "rollout_heartbeat"
    assert "proc_rss_mb" in perf_payload
    assert "cuda_available" in perf_payload

    summary_payload = next(
        record["payload"] for record in records if record["stream"] == "training_summary"
    )
    assert summary_payload["operation"] == "training_summary"
    assert summary_payload["total_steps"] == 8
