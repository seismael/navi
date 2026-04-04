"""Canonical SDF/DAG observation convention tests."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest
import torch

from navi_contracts.testing.oracle_house import house_metric_distances, house_observation
from navi_environment.backends.sdfdag_backend import (
    SdfDagBackend,
    _forward_structure_reward,
    _inspection_reward,
    _observation_profile,
    _obstacle_clearance_reward,
    _proximity_penalty,
    _select_spawn_yaw_from_observation,
    _spawn_candidate_score,
    _starvation_penalty,
    _structure_band_reward,
    _validate_unit_direction_tensor,
    build_spherical_ray_directions,
)


def test_build_spherical_ray_directions_uses_forward_at_azimuth_zero() -> None:
    dirs = build_spherical_ray_directions(8, 3)

    forward = dirs[1]
    rear = dirs[(4 * 3) + 1]

    np.testing.assert_allclose(forward, np.array([0.0, 0.0, -1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(rear, np.array([0.0, 0.0, 1.0], dtype=np.float32), atol=1e-6)


def test_build_spherical_ray_directions_span_up_and_down() -> None:
    dirs = build_spherical_ray_directions(8, 3)

    up = dirs[0]
    down = dirs[2]

    np.testing.assert_allclose(up, np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(down, np.array([0.0, -1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_kinematics_forward_matches_ray_forward_at_all_yaws() -> None:
    """Contract: movement forward and view forward must point the same direction.

    The yaw rotation R_y(θ) applied to local az=0 direction (0, 0, -1) gives
    world forward (-sin(θ), 0, -cos(θ)).  The kinematics must move the agent
    in this same direction when the forward action is applied.
    """
    for yaw_val in [0.0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2]:
        yaw = torch.tensor([yaw_val])
        # Compute view forward from ray rotation: R_y(yaw) * (0, 0, -1)
        view_fwd_x = -math.sin(yaw_val)
        view_fwd_z = -math.cos(yaw_val)

        # Compute kinematic forward by stepping with a pure forward action
        depth = torch.full((1, 8, 3), 1.0)
        pos = torch.zeros(1, 3)
        new_pos, _, _, _ = torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 3), torch.zeros(1, 3)
        # Use the same function as the backend
        from navi_environment.backends.sdfdag_backend import _step_kinematics_tensor

        new_pos, _, _, _ = _step_kinematics_tensor(
            depth,
            actions_linear=torch.tensor([[1.0, 0.0, 0.0]]),
            actions_angular=torch.tensor([[0.0, 0.0, 0.0]]),
            previous_linear=torch.zeros(1, 3),
            previous_angular=torch.zeros(1, 3),
            positions=pos,
            yaws=yaw,
            max_distance=30.0,
            speed_fwd=1.0,
            speed_vert=0.5,
            speed_lat=0.5,
            speed_yaw=1.0,
            smoothing=0.0,
            dt=1.0,
        )
        kin_fwd_x = float(new_pos[0, 0])
        kin_fwd_z = float(new_pos[0, 2])
        kin_norm = math.hypot(kin_fwd_x, kin_fwd_z)
        if kin_norm < 1e-9:
            continue
        kin_fwd_x /= kin_norm
        kin_fwd_z /= kin_norm

        np.testing.assert_allclose(
            [kin_fwd_x, kin_fwd_z],
            [view_fwd_x, view_fwd_z],
            atol=1e-5,
            err_msg=f"Kinematics/view mismatch at yaw={yaw_val:.3f}",
        )


def test_obstacle_clearance_reward_is_positive_when_moving_away_near_geometry() -> None:
    reward = _obstacle_clearance_reward(
        0.2,
        0.8,
        proximity_window=3.0,
        reward_scale=0.6,
    )

    assert reward > 0.0


def test_obstacle_clearance_reward_is_zero_when_both_clearances_are_far() -> None:
    reward = _obstacle_clearance_reward(
        4.0,
        4.4,
        proximity_window=3.0,
        reward_scale=0.6,
    )

    assert reward == 0.0


def test_observation_profile_tracks_starvation_and_near_geometry() -> None:
    metric_depth = np.array([[30.0, 0.6], [3.0, 15.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, True]], dtype=np.bool_)

    starvation_ratio, proximity_ratio, structure_band_ratio, forward_structure_ratio = (
        _observation_profile(
            metric_depth,
            valid,
            max_distance=20.0,
            proximity_distance_threshold=1.0,
            structure_band_min_distance=1.5,
            structure_band_max_distance=10.0,
        )
    )

    # 30.0 >= 20.0 - 1e-3 → saturated (1 of 4 pixels starved)
    assert math.isclose(starvation_ratio, 0.25)
    assert math.isclose(proximity_ratio, 0.25)
    assert math.isclose(structure_band_ratio, 0.25)
    assert math.isclose(forward_structure_ratio, 0.25)


def test_select_spawn_yaw_rotates_structure_into_forward_sector() -> None:
    metric_depth = np.ones((12, 4), dtype=np.float32) * 10.0
    valid = np.zeros((12, 4), dtype=np.bool_)
    metric_depth[4:7, :] = 3.0
    valid[4:7, :] = True

    yaw = _select_spawn_yaw_from_observation(
        metric_depth,
        valid,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )

    # The yaw rotation maps local az `a` to world az `a - yaw`.
    # Equivalently, world az `w` appears at local az `w + yaw`.
    # Rolling the world observation by +shift simulates what the agent sees.
    azimuth_bins = metric_depth.shape[0]
    shift = round((yaw / (2.0 * math.pi)) * azimuth_bins) % azimuth_bins
    rotated_depth = np.roll(metric_depth, shift, axis=0)
    rotated_valid = np.roll(valid, shift, axis=0)
    before_forward = _observation_profile(
        metric_depth,
        valid,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )[3]
    after_forward = _observation_profile(
        rotated_depth,
        rotated_valid,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )[3]

    assert after_forward > before_forward
    assert after_forward > 0.0


def test_spawn_candidate_score_prefers_structure_over_empty_space() -> None:
    structured_depth = np.full((12, 4), 10.0, dtype=np.float32)
    structured_valid = np.zeros((12, 4), dtype=np.bool_)
    structured_depth[3:9, :] = 4.0
    structured_valid[3:9, :] = True
    empty_depth = np.full((12, 4), 10.0, dtype=np.float32)
    empty_valid = np.zeros((12, 4), dtype=np.bool_)

    structured_score = _spawn_candidate_score(
        structured_depth,
        structured_valid,
        max_distance=15.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )
    empty_score = _spawn_candidate_score(
        empty_depth,
        empty_valid,
        max_distance=15.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )

    assert structured_score > empty_score


def test_reset_uses_precomputed_spawn_yaw() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._initial_resets_remaining = 0
    backend._maybe_rotate_scene = lambda: False  # type: ignore[method-assign]
    backend._spawn_positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    backend._spawn_yaws = torch.tensor([1.25], dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._episode_ids = torch.zeros((1,), dtype=torch.int64)
    backend._actor_steps = torch.zeros((1,), dtype=torch.int32)
    backend._prev_depth_tensors = torch.zeros((1, 2, 2), dtype=torch.float32)
    backend._prev_min_distances = torch.zeros((1,), dtype=torch.float32)
    backend._prev_structure_band_ratios = torch.zeros((1,), dtype=torch.float32)
    backend._prev_forward_structure_ratios = torch.zeros((1,), dtype=torch.float32)
    backend._episode_returns = torch.zeros((1,), dtype=torch.float32)
    backend._visit_grid = torch.zeros((1, 1, 1, 1), dtype=torch.int16)
    backend._heading_visited = torch.zeros((1, 16), dtype=torch.bool)
    backend._prev_linear_vels = torch.zeros((1, 3), dtype=torch.float32)
    backend._prev_angular_vels = torch.zeros((1, 3), dtype=torch.float32)
    backend._needs_reset_mask = torch.zeros((1,), dtype=torch.bool)
    backend._max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._cast_actor_batch_tensors = lambda actor_ids: (  # type: ignore[method-assign]
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 2, 2), dtype=torch.int32),
        torch.zeros((1, 2, 2), dtype=torch.bool),
        torch.zeros((1, 2, 2), dtype=torch.float32),
    )
    backend._consume_actor_observation = lambda **kwargs: (  # type: ignore[method-assign]
        None,
        np.zeros((2, 2), dtype=np.float32),
        torch.zeros((2, 2), dtype=torch.float32),
    )
    backend._materialize_observation = lambda **kwargs: "ok"  # type: ignore[method-assign]

    result = backend.reset(episode_id=7, actor_id=0)

    assert result == "ok"
    assert torch.allclose(backend._actor_positions[0], torch.tensor([1.0, 2.0, 3.0]))
    assert float(backend._actor_yaws[0]) == pytest.approx(1.25)


def test_reset_tensor_uses_tensor_ratios_when_materialization_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = object.__new__(SdfDagBackend)
    backend._initial_resets_remaining = 0
    backend._maybe_rotate_scene = lambda: False  # type: ignore[method-assign]
    backend._spawn_positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    backend._spawn_yaws = torch.tensor([0.25], dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._episode_ids = torch.zeros((1,), dtype=torch.int64)
    backend._actor_steps = torch.zeros((1,), dtype=torch.int32)
    backend._prev_depth_tensors = torch.zeros((1, 2, 2), dtype=torch.float32)
    backend._prev_min_distances = torch.zeros((1,), dtype=torch.float32)
    backend._prev_structure_band_ratios = torch.zeros((1,), dtype=torch.float32)
    backend._prev_forward_structure_ratios = torch.zeros((1,), dtype=torch.float32)
    backend._episode_returns = torch.zeros((1,), dtype=torch.float32)
    backend._visit_grid = torch.zeros((1, 1, 1, 1), dtype=torch.int16)
    backend._heading_visited = torch.zeros((1, 16), dtype=torch.bool)
    backend._prev_linear_vels = torch.zeros((1, 3), dtype=torch.float32)
    backend._prev_angular_vels = torch.zeros((1, 3), dtype=torch.float32)
    backend._needs_reset_mask = torch.zeros((1,), dtype=torch.bool)
    backend._max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._cast_actor_batch_tensors = lambda actor_ids: (  # type: ignore[method-assign]
        torch.tensor([[[0.2, 0.3], [0.4, 0.5]]], dtype=torch.float32),
        torch.zeros((1, 2, 2), dtype=torch.int32),
        torch.ones((1, 2, 2), dtype=torch.bool),
        torch.tensor([[[0.2, 0.3], [0.4, 0.5]]], dtype=torch.float32),
    )
    backend._consume_actor_observation = lambda **kwargs: (  # type: ignore[method-assign]
        torch.ones((3, 2, 2), dtype=torch.float32),
        None,
        torch.zeros((2, 2), dtype=torch.float32),
    )
    backend._compute_observation_ratios = lambda *, clamped_metric, valid_batch: (  # type: ignore[method-assign]
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.3], dtype=torch.float32),
        torch.tensor([0.4], dtype=torch.float32),
    )
    backend._materialize_observation = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("not used")
    )  # type: ignore[method-assign]

    monkeypatch.setattr(
        "navi_environment.backends.sdfdag_backend._observation_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected numpy profile call")
        ),
    )

    obs_tensor, published = backend.reset_tensor(episode_id=7, actor_id=0, materialize=False)

    assert published is None
    assert obs_tensor.shape == (3, 2, 2)
    assert float(backend._prev_structure_band_ratios[0]) == pytest.approx(0.3)
    assert float(backend._prev_forward_structure_ratios[0]) == pytest.approx(0.4)


def test_select_publish_rows_returns_only_selected_actor_rows() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch

    rows, actor_ids = backend._select_publish_rows(
        actor_indices=torch.tensor([3, 1, 2, 0], dtype=torch.int64),
        publish_actor_ids=(2, 0),
    )

    assert rows == [2, 3]
    assert actor_ids == [2, 0]


def _make_tensor_action_backend_stub(*, n_actors: int = 4) -> SdfDagBackend:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._config = type("Config", (), {"sdf_max_steps": 32})()
    backend._device = torch.device("cpu")
    backend._n_actors = n_actors
    backend._az_bins = 2
    backend._el_bins = 2
    backend._n_rays = 4
    backend._max_distance = 10.0
    backend._max_steps_per_episode = 5
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._collision_clearance = 0.15
    backend._speed_limiter_distance = 0.8
    backend._needs_reset_mask = torch.zeros((n_actors,), dtype=torch.bool)
    backend._actor_positions = torch.zeros((n_actors, 3), dtype=torch.float32)
    backend._actor_yaws = torch.zeros((n_actors,), dtype=torch.float32)
    backend._episode_ids = torch.arange(100, 100 + n_actors, dtype=torch.int64)
    backend._actor_steps = torch.zeros((n_actors,), dtype=torch.int64)
    backend._episode_returns = torch.zeros((n_actors,), dtype=torch.float32)
    backend._prev_min_distances = torch.zeros((n_actors,), dtype=torch.float32)
    backend._prev_structure_band_ratios = torch.zeros((n_actors,), dtype=torch.float32)
    backend._prev_forward_structure_ratios = torch.zeros((n_actors,), dtype=torch.float32)
    backend._ray_dirs_local = torch.zeros((backend._n_rays, 3), dtype=torch.float32)
    backend._bbox_min = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    backend._bbox_max = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    backend._record_perf_sample = lambda **kwargs: None  # type: ignore[method-assign]
    backend._advance_reset_bookkeeping = lambda reset_count: None  # type: ignore[method-assign]
    backend._reset_tensor_actor_batch = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected reset path")
    )  # type: ignore[method-assign]
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 8})()  # type: ignore[method-assign]

    def cast_rays(*args: object, **kwargs: object) -> None:
        del kwargs
        out_distances = cast("torch.Tensor", args[3])
        out_semantics = cast("torch.Tensor", args[4])
        out_distances.fill_(2.5)
        out_semantics.zero_()

    backend._torch_sdf = type("TorchSdfStub", (), {"cast_rays": staticmethod(cast_rays)})()

    def scratch_slot_views(
        *, scratch_slot: int, actor_count: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        origins = torch.zeros((actor_count, backend._n_rays, 3), dtype=torch.float32)
        dirs_world = torch.zeros((actor_count, backend._n_rays, 3), dtype=torch.float32)
        out_distances = torch.zeros((actor_count, backend._n_rays), dtype=torch.float32)
        out_semantics = torch.zeros((actor_count, backend._n_rays), dtype=torch.int32)
        return origins, dirs_world, out_distances, out_semantics

    backend._scratch_slot_views = scratch_slot_views  # type: ignore[method-assign]

    def step_kinematics_indexed(
        *, actor_indices: torch.Tensor, actions_linear: torch.Tensor, actions_angular: torch.Tensor
    ) -> None:
        backend._actor_positions[actor_indices, 0] = (
            actor_indices.to(dtype=torch.float32) + actions_linear[:, 0]
        )
        backend._actor_yaws[actor_indices] = actions_angular[:, 2]

    backend._step_kinematics_indexed = step_kinematics_indexed  # type: ignore[method-assign]

    def compute_reward_batch(**kwargs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actor_indices = cast("torch.Tensor", kwargs["actor_indices"])
        rewards = actor_indices.to(dtype=torch.float32) + 0.5
        components = torch.zeros((int(actor_indices.shape[0]), 9), dtype=torch.float32)
        return rewards, components

    backend._compute_reward_batch = compute_reward_batch  # type: ignore[method-assign]

    def consume_observation_batch(
        *,
        actor_indices: torch.Tensor,
        depth_batch: torch.Tensor,
        semantic_batch: torch.Tensor,
        valid_batch: torch.Tensor,
        current_clearances: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_count = int(actor_indices.shape[0])
        obs_batch = torch.zeros(
            (actor_count, 3, backend._az_bins, backend._el_bins), dtype=torch.float32
        )
        obs_batch[:, 0, :, :] = actor_indices.to(dtype=torch.float32).view(-1, 1, 1)
        delta_batch = torch.zeros(
            (actor_count, backend._az_bins, backend._el_bins), dtype=torch.float32
        )
        return obs_batch, delta_batch

    backend._consume_observation_batch = consume_observation_batch  # type: ignore[method-assign]
    backend._materialize_selected_observations = lambda **kwargs: None  # type: ignore[method-assign]
    backend._steps_in_scene = 0
    return backend


def test_batch_step_tensor_actions_preserves_subset_actor_ids() -> None:
    backend = _make_tensor_action_backend_stub()

    step_batch, ordered_results = backend.batch_step_tensor_actions(
        torch.tensor([[1.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.2]], dtype=torch.float32),
        step_id=17,
        actor_indices=torch.tensor([2, 0], dtype=torch.int64),
        publish_actor_ids=(),
        materialize_results=False,
    )

    assert ordered_results == ()
    assert torch.equal(step_batch.env_id_tensor, torch.tensor([2, 0], dtype=torch.int64))
    assert torch.equal(step_batch.episode_id_tensor, torch.tensor([102, 100], dtype=torch.int64))
    assert torch.allclose(step_batch.reward_tensor, torch.tensor([2.5, 0.5], dtype=torch.float32))
    assert torch.allclose(
        step_batch.observation_tensor[:, 0, 0, 0], torch.tensor([2.0, 0.0], dtype=torch.float32)
    )
    assert float(backend._actor_positions[2, 0]) == pytest.approx(3.0)
    assert float(backend._actor_positions[0, 0]) == pytest.approx(2.0)
    assert float(backend._actor_positions[1, 0]) == pytest.approx(0.0)
    assert float(backend._actor_positions[3, 0]) == pytest.approx(0.0)


def test_batch_step_tensor_actions_skips_publish_materialization_when_unrequested() -> None:
    backend = _make_tensor_action_backend_stub()
    backend._select_publish_rows = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected publish-row selection")
    )  # type: ignore[method-assign]
    backend._materialize_selected_observations = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("unexpected observation materialization")
    )  # type: ignore[method-assign]

    step_batch, ordered_results = backend.batch_step_tensor_actions(
        torch.tensor([[0.5, 0.0, 0.0, 0.0]], dtype=torch.float32),
        step_id=3,
        actor_indices=torch.tensor([1], dtype=torch.int64),
        publish_actor_ids=(),
        materialize_results=False,
    )

    assert ordered_results == ()
    assert step_batch.published_observations == {}
    assert torch.equal(step_batch.env_id_tensor, torch.tensor([1], dtype=torch.int64))


def test_starvation_penalty_triggers_only_beyond_threshold() -> None:
    assert _starvation_penalty(0.7, ratio_threshold=0.8, penalty_scale=1.5) < 0.0
    assert _starvation_penalty(0.9, ratio_threshold=0.8, penalty_scale=1.5) < 0.0


def test_proximity_penalty_scales_with_near_field_ratio() -> None:
    assert _proximity_penalty(0.0, penalty_scale=0.8) == 0.0
    assert _proximity_penalty(0.5, penalty_scale=0.8) == -0.4


def test_structure_band_reward_scales_with_mid_range_geometry() -> None:
    assert _structure_band_reward(0.0, reward_scale=0.35) == 0.0
    assert _structure_band_reward(0.5, reward_scale=0.35) == pytest.approx(0.175)


def test_forward_structure_reward_scales_with_forward_visibility() -> None:
    assert _forward_structure_reward(0.0, reward_scale=0.2) == 0.0
    assert _forward_structure_reward(0.5, reward_scale=0.2) == pytest.approx(0.1)


def test_inspection_reward_requires_structure_and_tracks_information_gain() -> None:
    assert (
        _inspection_reward(
            0.10,
            0.35,
            previous_forward_structure_ratio=0.05,
            current_forward_structure_ratio=0.20,
            reward_scale=0.25,
            activation_threshold=0.05,
        )
        > 0.0
    )
    assert (
        _inspection_reward(
            0.35,
            0.10,
            previous_forward_structure_ratio=0.20,
            current_forward_structure_ratio=0.05,
            reward_scale=0.25,
            activation_threshold=0.05,
        )
        < 0.0
    )


def test_validate_unit_direction_tensor_rejects_unnormalized_vectors() -> None:
    ray_dirs = torch.tensor([[1.2, 0.0, 0.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="normalized within tolerance"):
        _validate_unit_direction_tensor(torch, ray_dirs, name="ray_dirs")


def test_scene_rotation_waits_for_configured_step_budget() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._scene_pool = ["scene_a.gmdag", "scene_b.gmdag"]
    backend._scene_pool_idx = 0
    backend._steps_in_scene = 63999
    backend._scene_steps_budget = 32000
    backend._n_actors = 2
    backend._device = torch.device("cpu")
    backend._torch = torch
    backend._spawn_positions = torch.zeros((2, 3), dtype=torch.float32)
    backend._needs_reset_mask = torch.zeros((2,), dtype=torch.bool)

    def _load_scene(_scene_path: str) -> list[tuple[float, float, float]]:
        spawns = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]
        for i, spawn in enumerate(spawns):
            backend._spawn_positions[i].copy_(torch.tensor(spawn))
        return spawns

    backend._load_scene = _load_scene  # type: ignore[assignment]

    # Budget not yet met
    assert not backend._maybe_rotate_scene()

    # Push past budget
    backend._steps_in_scene = 64000
    backend._maybe_rotate_scene()

    assert backend._scene_pool_idx == 1
    assert backend._steps_in_scene == 0
    assert torch.allclose(backend._spawn_positions[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(backend._spawn_positions[1], torch.tensor([4.0, 5.0, 6.0]))


def test_cast_actor_batch_tensors_passes_environment_horizon() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def __init__(self) -> None:
            self.args: tuple[object, ...] | None = None

        def cast_rays(self, *args: object, **kwargs: object) -> None:
            self.args = args
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.zero_()
            out_semantics.zero_()

    fake_sdf = _FakeSdf()
    backend._torch = torch
    backend._torch_sdf = fake_sdf
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 17.5
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._bbox_min = [0.0, 0.0, 0.0]
    backend._bbox_max = [1.0, 1.0, 1.0]
    backend._az_bins = 1
    backend._el_bins = 1
    backend._n_rays = 1
    backend._device = torch.device("cpu")
    backend._ray_dirs_local = torch.zeros((1, 3), dtype=torch.float32)
    backend._ray_origins = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._ray_dirs_world = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._out_distances = torch.empty((1, 1), dtype=torch.float32)
    backend._out_semantics = torch.empty((1, 1), dtype=torch.int32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 512})()  # type: ignore[method-assign]

    backend._cast_actor_batch_tensors((0,))

    assert fake_sdf.args is not None
    assert fake_sdf.args[6] == 17.5


def test_cast_actor_batch_tensors_clamps_and_marks_values_beyond_fixed_horizon() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object, **kwargs: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[5.0, 25.0]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[1, 2]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._bbox_min = [0.0, 0.0, 0.0]
    backend._bbox_max = [1.0, 1.0, 1.0]
    backend._az_bins = 2
    backend._el_bins = 1
    backend._n_rays = 2
    backend._device = torch.device("cpu")
    backend._ray_dirs_local = torch.zeros((2, 3), dtype=torch.float32)
    backend._ray_origins = torch.empty((1, 2, 3), dtype=torch.float32)
    backend._ray_dirs_world = torch.empty((1, 2, 3), dtype=torch.float32)
    backend._out_distances = torch.empty((1, 2), dtype=torch.float32)
    backend._out_semantics = torch.empty((1, 2), dtype=torch.int32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 512})()  # type: ignore[method-assign]

    depth_2d, semantic_2d, valid_2d, _metric = backend._cast_actor_batch_tensors((0,))

    assert depth_2d.shape == (1, 2, 1)
    assert semantic_2d.shape == (1, 2, 1)
    assert valid_2d.shape == (1, 2, 1)
    expected_depth = torch.tensor(
        [math.log1p(5.0) / math.log1p(10.0), 1.0], dtype=torch.float32
    )
    assert torch.allclose(depth_2d[0, :, 0], expected_depth)
    # With sensor-validity semantics: both rays are finite → both valid.
    # The beyond-horizon ray (25.0) is clamped to max_distance (depth=1.0)
    # but still valid since it is a finite reading.
    assert torch.equal(valid_2d[0, :, 0], torch.tensor([True, True]))


def test_cast_actor_batch_tensors_treats_exact_horizon_as_valid() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object, **kwargs: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[10.0]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[3]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._bbox_min = [0.0, 0.0, 0.0]
    backend._bbox_max = [1.0, 1.0, 1.0]
    backend._az_bins = 1
    backend._el_bins = 1
    backend._n_rays = 1
    backend._device = torch.device("cpu")
    backend._ray_dirs_local = torch.zeros((1, 3), dtype=torch.float32)
    backend._ray_origins = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._ray_dirs_world = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._out_distances = torch.empty((1, 1), dtype=torch.float32)
    backend._out_semantics = torch.empty((1, 1), dtype=torch.int32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 512})()  # type: ignore[method-assign]

    depth_2d, semantic_2d, valid_2d, _metric = backend._cast_actor_batch_tensors((0,))

    assert torch.allclose(depth_2d[0, :, 0], torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(semantic_2d[0, :, 0], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(valid_2d[0, :, 0], torch.tensor([True]))


def test_cast_actor_batch_tensors_clamps_negative_inside_solid_distances_to_zero() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object, **kwargs: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[-0.25]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[9]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._bbox_min = [0.0, 0.0, 0.0]
    backend._bbox_max = [1.0, 1.0, 1.0]
    backend._az_bins = 1
    backend._el_bins = 1
    backend._n_rays = 1
    backend._device = torch.device("cpu")
    backend._ray_dirs_local = torch.zeros((1, 3), dtype=torch.float32)
    backend._ray_origins = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._ray_dirs_world = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._out_distances = torch.empty((1, 1), dtype=torch.float32)
    backend._out_semantics = torch.empty((1, 1), dtype=torch.int32)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 512})()  # type: ignore[method-assign]

    depth_2d, semantic_2d, valid_2d, _metric = backend._cast_actor_batch_tensors((0,))

    assert torch.allclose(depth_2d[0, :, 0], torch.tensor([0.0], dtype=torch.float32))
    assert torch.equal(semantic_2d[0, :, 0], torch.tensor([9], dtype=torch.int32))
    assert torch.equal(valid_2d[0, :, 0], torch.tensor([True]))


def test_cast_actor_batch_tensors_rejects_wrong_output_dtype() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object) -> None:
            raise AssertionError("cast_rays should not run when validation fails")

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 17.5
    backend._proximity_distance_threshold = 1.0
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 10.0
    backend._bbox_min = [0.0, 0.0, 0.0]
    backend._bbox_max = [1.0, 1.0, 1.0]
    backend._az_bins = 1
    backend._el_bins = 1
    backend._n_rays = 1
    backend._device = torch.device("cpu")
    backend._ray_dirs_local = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    backend._ray_origins = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._ray_dirs_world = torch.empty((1, 1, 3), dtype=torch.float32)
    backend._out_distances = torch.empty((1, 1), dtype=torch.float32)
    backend._out_semantics = torch.empty((1, 1), dtype=torch.int64)
    backend._actor_yaws = torch.zeros((1,), dtype=torch.float32)
    backend._actor_positions = torch.zeros((1, 3), dtype=torch.float32)
    backend._require_dag_tensor = lambda: torch.zeros((1,), dtype=torch.int64)  # type: ignore[method-assign]
    backend._require_asset = lambda: type("Asset", (), {"resolution": 512})()  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="out_semantics must have dtype"):
        backend._cast_actor_batch_tensors((0,))


def test_postprocess_cast_outputs_preserves_oracle_house_profile() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._max_distance = 10.0
    backend._az_bins = 12
    backend._el_bins = 3
    backend._structure_band_min_distance = 1.5
    backend._structure_band_max_distance = 8.0
    backend._proximity_distance_threshold = 0.5

    oracle = house_observation()
    metric = house_metric_distances(backend._max_distance)
    metric = np.where(oracle.valid, metric, backend._max_distance + 1.0)
    out_distances = torch.from_numpy(metric.reshape(1, -1))
    out_semantics = torch.from_numpy(oracle.semantic.reshape(1, -1))

    depth_batch, semantic_batch, valid_batch, min_distances, *_rest = (
        backend._postprocess_cast_outputs(
            out_distances,
            out_semantics,
        )
    )

    # All metric values are finite, so all rays are valid under the sensor-
    # validity semantics (finite = valid, regardless of distance).
    expected_depth = np.where(
        oracle.valid,
        np.log1p(oracle.depth * 10.0) / np.log1p(10.0),
        np.log1p(backend._max_distance) / np.log1p(backend._max_distance),
    ).astype(np.float32)
    np.testing.assert_allclose(depth_batch[0].numpy(), expected_depth, atol=1e-6)
    np.testing.assert_array_equal(semantic_batch[0].numpy(), oracle.semantic)
    # valid_batch = all True since every ray distance is finite.
    assert valid_batch[0].all().item()
    assert min_distances.shape == (1,)
    # min_distances considers all clamped ray distances (min across the grid).
    assert min_distances[0].item() == pytest.approx(float(np.min(np.clip(metric, 0.0, backend._max_distance))))


def test_materialize_step_result_rows_preserves_non_monotonic_env_order() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._episode_ids = torch.tensor([101, 202, 303], dtype=torch.int64)
    backend._episode_returns = torch.tensor([1.25, 2.5, 3.75], dtype=torch.float32)

    rows = backend._materialize_step_result_rows(
        local_rows=torch.tensor([1, 0], dtype=torch.int64),
        env_ids=torch.tensor([2, 0], dtype=torch.int64),
        actor_indices=torch.tensor([1, 2], dtype=torch.int64),
        rewards=torch.tensor([0.5, -0.25], dtype=torch.float32),
        truncated_mask=torch.tensor([True, False], dtype=torch.bool),
    )

    assert rows.shape == (2, 6)
    np.testing.assert_allclose(rows[:, 0], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 1], np.array([2.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 2], np.array([202.0, 303.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 3], np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 4], np.array([0.5, -0.25], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 5], np.array([2.5, 3.75], dtype=np.float32))


def test_materialize_reset_result_rows_preserves_selected_actor_order() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._episode_ids = torch.tensor([7, 8, 9, 10], dtype=torch.int64)

    rows = backend._materialize_reset_result_rows(
        local_rows=torch.tensor([0, 1], dtype=torch.int64),
        env_ids=torch.tensor([3, 1], dtype=torch.int64),
        actor_indices=torch.tensor([0, 2], dtype=torch.int64),
    )

    assert rows.shape == (2, 3)
    np.testing.assert_allclose(rows[:, 0], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 1], np.array([3.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(rows[:, 2], np.array([7.0, 9.0], dtype=np.float32))


def test_coerce_batch_actor_indices_accepts_explicit_subset_order() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._device = torch.device("cpu")
    backend._n_actors = 6

    indices = backend._coerce_batch_actor_indices(actor_count=3, actor_indices=[4, 1, 5])

    assert torch.equal(indices, torch.tensor([4, 1, 5], dtype=torch.int64))


def test_coerce_batch_actor_indices_rejects_duplicates() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._torch = torch
    backend._device = torch.device("cpu")
    backend._n_actors = 4

    with pytest.raises(ValueError, match="must be unique"):
        backend._coerce_batch_actor_indices(actor_count=2, actor_indices=[1, 1])


def test_materialize_observation_preserves_oracle_house_arrays() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._episode_ids = torch.tensor([17], dtype=torch.int64)
    backend.actor_pose = lambda actor_id: type(
        "Pose",
        (),
        {
            "x": float(actor_id),
            "y": 0.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "timestamp": 1.0,
        },
    )()  # type: ignore[method-assign]

    oracle = house_observation()
    delta = np.zeros_like(oracle.depth)
    observation = backend._materialize_observation(
        actor_id=0,
        step_id=9,
        depth_2d=oracle.depth,
        delta_2d=delta,
        semantic_2d=oracle.semantic,
        valid_2d=oracle.valid,
    )

    np.testing.assert_allclose(observation.depth[0], oracle.depth)
    np.testing.assert_allclose(observation.delta_depth[0], delta)
    np.testing.assert_array_equal(observation.semantic[0], oracle.semantic)
    np.testing.assert_array_equal(observation.valid_mask[0], oracle.valid)
    assert observation.episode_id == 17
    assert observation.step_id == 9
