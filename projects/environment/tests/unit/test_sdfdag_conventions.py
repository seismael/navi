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
