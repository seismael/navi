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


def test_obstacle_clearance_reward_is_positive_when_moving_away_near_geometry() -> None:
    reward = _obstacle_clearance_reward(
        0.2,
        0.8,
        proximity_window=1.5,
        reward_scale=0.6,
    )

    assert reward > 0.0


def test_obstacle_clearance_reward_is_zero_when_both_clearances_are_far() -> None:
    reward = _obstacle_clearance_reward(
        2.0,
        2.4,
        proximity_window=1.5,
        reward_scale=0.6,
    )

    assert reward == 0.0


def test_observation_profile_tracks_starvation_and_near_geometry() -> None:
    depth = np.array([[1.0, 0.02], [0.10, 0.50]], dtype=np.float32)
    valid = np.array([[False, True], [True, True]], dtype=np.bool_)

    starvation_ratio, proximity_ratio, structure_band_ratio, forward_structure_ratio = _observation_profile(
        depth,
        valid,
        max_distance=30.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )

    assert math.isclose(starvation_ratio, 0.25)
    assert math.isclose(proximity_ratio, 0.25)
    assert math.isclose(structure_band_ratio, 0.25)
    assert math.isclose(forward_structure_ratio, 0.25)


def test_select_spawn_yaw_rotates_structure_into_forward_sector() -> None:
    depth = np.ones((12, 4), dtype=np.float32)
    valid = np.zeros((12, 4), dtype=np.bool_)
    depth[4:7, :] = 0.3
    valid[4:7, :] = True

    yaw = _select_spawn_yaw_from_observation(
        depth,
        valid,
        max_distance=10.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )

    azimuth_bins = depth.shape[0]
    shift = int(round((yaw / (2.0 * math.pi)) * azimuth_bins)) % azimuth_bins
    rotated_depth = np.roll(depth, -shift, axis=0)
    rotated_valid = np.roll(valid, -shift, axis=0)
    before_forward = _observation_profile(
        depth,
        valid,
        max_distance=10.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )[3]
    after_forward = _observation_profile(
        rotated_depth,
        rotated_valid,
        max_distance=10.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )[3]

    assert after_forward > before_forward
    assert after_forward > 0.0


def test_spawn_candidate_score_prefers_structure_over_empty_space() -> None:
    structured_depth = np.ones((12, 4), dtype=np.float32)
    structured_valid = np.zeros((12, 4), dtype=np.bool_)
    structured_depth[3:9, :] = 0.4
    structured_valid[3:9, :] = True
    empty_depth = np.ones((12, 4), dtype=np.float32)
    empty_valid = np.zeros((12, 4), dtype=np.bool_)

    structured_score = _spawn_candidate_score(
        structured_depth,
        structured_valid,
        max_distance=10.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )
    empty_score = _spawn_candidate_score(
        empty_depth,
        empty_valid,
        max_distance=10.0,
        proximity_distance_threshold=1.0,
        structure_band_min_distance=1.5,
        structure_band_max_distance=10.0,
    )

    assert structured_score > empty_score


def test_reset_uses_precomputed_spawn_yaw() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._initial_resets_remaining = 0
    backend._maybe_rotate_scene = lambda *, is_natural: False  # type: ignore[method-assign]
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
    backend._visit_grid = torch.zeros((1, 1, 1), dtype=torch.uint8)
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


def test_reset_tensor_uses_tensor_ratios_when_materialization_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = object.__new__(SdfDagBackend)
    backend._initial_resets_remaining = 0
    backend._maybe_rotate_scene = lambda *, is_natural: False  # type: ignore[method-assign]
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
    backend._visit_grid = torch.zeros((1, 1, 1), dtype=torch.uint8)
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
    )
    backend._consume_actor_observation = lambda **kwargs: (  # type: ignore[method-assign]
        torch.ones((3, 2, 2), dtype=torch.float32),
        None,
        torch.zeros((2, 2), dtype=torch.float32),
    )
    backend._compute_observation_ratios = lambda *, depth_batch, valid_batch: (  # type: ignore[method-assign]
        torch.tensor([0.1], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.3], dtype=torch.float32),
        torch.tensor([0.4], dtype=torch.float32),
    )
    backend._materialize_observation = lambda **kwargs: (_ for _ in ()).throw(AssertionError("not used"))  # type: ignore[method-assign]

    monkeypatch.setattr(
        "navi_environment.backends.sdfdag_backend._observation_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected numpy profile call")),
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
    assert _inspection_reward(
        0.10,
        0.35,
        previous_forward_structure_ratio=0.05,
        current_forward_structure_ratio=0.20,
        reward_scale=0.25,
        activation_threshold=0.05,
    ) > 0.0
    assert _inspection_reward(
        0.35,
        0.10,
        previous_forward_structure_ratio=0.20,
        current_forward_structure_ratio=0.05,
        reward_scale=0.25,
        activation_threshold=0.05,
    ) < 0.0


def test_validate_unit_direction_tensor_rejects_unnormalized_vectors() -> None:
    ray_dirs = torch.tensor([[1.2, 0.0, 0.0]], dtype=torch.float32)

    with pytest.raises(RuntimeError, match="normalized within tolerance"):
        _validate_unit_direction_tensor(torch, ray_dirs, name="ray_dirs")


def test_scene_rotation_waits_for_configured_episode_budget() -> None:
    backend = object.__new__(SdfDagBackend)
    backend._scene_pool = ["scene_a.gmdag", "scene_b.gmdag"]
    backend._scene_pool_idx = 0
    backend._episodes_in_scene = 31
    backend._scene_episodes_per_scene = 16
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

    backend._maybe_rotate_scene(is_natural=True)

    assert backend._scene_pool_idx == 1
    assert backend._episodes_in_scene == 0
    assert torch.allclose(backend._spawn_positions[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(backend._spawn_positions[1], torch.tensor([4.0, 5.0, 6.0]))


def test_cast_actor_batch_tensors_passes_environment_horizon() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def __init__(self) -> None:
            self.args: tuple[object, ...] | None = None

        def cast_rays(self, *args: object) -> None:
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
        def cast_rays(self, *args: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[5.0, 25.0]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[1, 2]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
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

    depth_2d, semantic_2d, valid_2d = backend._cast_actor_batch_tensors((0,))

    assert depth_2d.shape == (1, 2, 1)
    assert semantic_2d.shape == (1, 2, 1)
    assert valid_2d.shape == (1, 2, 1)
    assert torch.allclose(depth_2d[0, :, 0], torch.tensor([0.5, 1.0], dtype=torch.float32))
    assert torch.equal(valid_2d[0, :, 0], torch.tensor([True, False]))


def test_cast_actor_batch_tensors_treats_exact_horizon_as_valid() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[10.0]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[3]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
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

    depth_2d, semantic_2d, valid_2d = backend._cast_actor_batch_tensors((0,))

    assert torch.allclose(depth_2d[0, :, 0], torch.tensor([1.0], dtype=torch.float32))
    assert torch.equal(semantic_2d[0, :, 0], torch.tensor([3], dtype=torch.int32))
    assert torch.equal(valid_2d[0, :, 0], torch.tensor([True]))


def test_cast_actor_batch_tensors_clamps_negative_inside_solid_distances_to_zero() -> None:
    backend = object.__new__(SdfDagBackend)

    class _FakeSdf:
        def cast_rays(self, *args: object) -> None:
            out_distances = cast("torch.Tensor", args[3])
            out_semantics = cast("torch.Tensor", args[4])
            out_distances.copy_(torch.tensor([[-0.25]], dtype=torch.float32))
            out_semantics.copy_(torch.tensor([[9]], dtype=torch.int32))

    backend._torch = torch
    backend._torch_sdf = _FakeSdf()
    backend._config = type("Cfg", (), {"sdf_max_steps": 64})()
    backend._max_distance = 10.0
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

    depth_2d, semantic_2d, valid_2d = backend._cast_actor_batch_tensors((0,))

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

    oracle = house_observation()
    metric = house_metric_distances(backend._max_distance)
    metric = np.where(oracle.valid, metric, backend._max_distance + 1.0)
    out_distances = torch.from_numpy(metric.reshape(1, -1))
    out_semantics = torch.from_numpy(oracle.semantic.reshape(1, -1))

    depth_batch, semantic_batch, valid_batch, min_distances, *_rest = backend._postprocess_cast_outputs_impl(
        out_distances,
        out_semantics,
    )

    np.testing.assert_allclose(depth_batch[0].numpy(), oracle.depth)
    np.testing.assert_array_equal(semantic_batch[0].numpy(), oracle.semantic)
    np.testing.assert_array_equal(valid_batch[0].numpy(), oracle.valid)
    assert min_distances.shape == (1,)
    assert min_distances[0].item() == pytest.approx(float(np.min(metric[oracle.valid])))


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
    backend.actor_pose = lambda actor_id: type("Pose", (), {
        "x": float(actor_id),
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "timestamp": 1.0,
    })()  # type: ignore[method-assign]

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
