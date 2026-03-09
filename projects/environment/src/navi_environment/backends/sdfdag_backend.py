"""CUDA-backed SDF/DAG simulator backend.

This backend preserves the canonical ``DistanceMatrix`` observation contract while
moving scene queries onto the internal ``torch-sdf`` CUDA runtime over compiled
``.gmdag`` assets produced by ``projects/voxel-dag``.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult
from navi_environment.backends.base import SimulatorBackend
from navi_environment.integration import GmDagAsset, load_gmdag_asset
from navi_environment.mjx_env import MjxEnvironment

if TYPE_CHECKING:
    from navi_environment.config import EnvironmentConfig

__all__: list[str] = ["SdfDagBackend", "SdfDagPerfSnapshot", "SdfDagTensorStepBatch"]

_LOG = logging.getLogger(__name__)

_COLLISION_CLEARANCE: float = 0.35
_EXPLORATION_REWARD: float = 0.3
_PROGRESS_REWARD_SCALE: float = 0.8
_COLLISION_PENALTY: float = -2.0
_MAX_STEPS_PER_EPISODE: int = 2_000
_SPAWN_CANDIDATES_PER_AXIS: int = 3
_SPAWN_HEIGHT_SAMPLES: int = 5
_PERF_EMA_ALPHA: float = 0.1


@dataclass
class _ActorState:
    """Mutable per-actor state over a shared DAG scene."""

    pose: RobotPose
    episode_id: int = 0
    step_count: int = 0
    prev_depth: np.ndarray | None = None
    prev_depth_tensor: Any | None = None
    episode_return: float = 0.0
    visited_cells: set[tuple[int, int]] = field(default_factory=set)
    needs_reset: bool = False
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class SdfDagPerfSnapshot:
    """Rolling batch-step metrics for the canonical SDF/DAG runtime."""

    total_batches: int
    total_actor_steps: int
    last_actor_count: int
    last_batch_step_ms: float
    ema_batch_step_ms: float
    avg_batch_step_ms: float
    avg_actor_step_ms: float
    sps: float


@dataclass(frozen=True)
class SdfDagTensorStepBatch:
    """Tensor-native canonical rollout batch with optional publish materialization."""

    observation_tensor: Any
    published_observations: dict[int, DistanceMatrix]


class SdfDagBackend(SimulatorBackend):
    """Simulator backend using GPU sphere tracing against a compiled `.gmdag`."""

    def __init__(self, config: EnvironmentConfig) -> None:
        if not config.gmdag_file and not config.scene_pool:
            msg = "SdfDagBackend requires --gmdag-file"
            raise ValueError(msg)

        self._config = config
        self._torch = self._import_required_module("torch")
        self._torch_sdf = self._import_required_module("torch_sdf")
        if not self._torch.cuda.is_available():
            msg = "CUDA is not available. The canonical sdfdag backend does not support CPU fallback."
            raise RuntimeError(msg)

        self._device = self._torch.device("cuda")
        self._az_bins = config.azimuth_bins
        self._el_bins = config.elevation_bins
        self._max_distance = config.max_distance
        self._n_actors = config.n_actors
        self._n_rays = self._az_bins * self._el_bins
        self._max_steps_per_episode = _MAX_STEPS_PER_EPISODE
        self._scene_pool: list[str] = list(config.scene_pool) if config.scene_pool else []
        self._scene_pool_idx: int = 0
        self._episodes_in_scene: int = 0
        self._initial_resets_remaining: int = self._n_actors
        self._asset: GmDagAsset | None = None
        self._dag_tensor: Any | None = None
        self._bbox_min: list[float] = []
        self._bbox_max: list[float] = []

        self._ray_dirs_local = self._build_ray_directions().to(self._device)
        self._ray_origins = self._torch.empty(
            (self._n_actors, self._n_rays, 3),
            device=self._device,
            dtype=self._torch.float32,
        )
        self._ray_dirs_world = self._torch.empty_like(self._ray_origins)
        self._out_distances = self._torch.empty(
            (self._n_actors, self._n_rays),
            device=self._device,
            dtype=self._torch.float32,
        )
        self._out_semantics = self._torch.empty(
            (self._n_actors, self._n_rays),
            device=self._device,
            dtype=self._torch.int32,
        )
        self._perf_total_batches = 0
        self._perf_total_actor_steps = 0
        self._perf_total_batch_seconds = 0.0
        self._perf_last_batch_seconds = 0.0
        self._perf_last_actor_count = 0
        self._perf_ema_batch_seconds = 0.0
        self._mjx_envs = {
            actor_id: MjxEnvironment(
                dt=config.physics_dt,
                speed_scales=(
                    config.drone_max_speed,
                    config.drone_climb_rate,
                    config.drone_strafe_speed,
                    config.drone_yaw_rate,
                ),
            )
            for actor_id in range(self._n_actors)
        }

        scene_path = self._scene_pool[0] if self._scene_pool else config.gmdag_file
        spawns = self._load_scene(scene_path)
        self._actors: dict[int, _ActorState] = {}
        for actor_id in range(self._n_actors):
            spawn = spawns[actor_id % len(spawns)]
            yaw = (2.0 * math.pi * actor_id) / max(self._n_actors, 1)
            self._actors[actor_id] = _ActorState(
                pose=RobotPose(
                    x=spawn[0],
                    y=spawn[1],
                    z=spawn[2],
                    roll=0.0,
                    pitch=0.0,
                    yaw=yaw,
                    timestamp=time.time(),
                ),
                spawn_position=spawn,
            )

        if self._scene_pool:
            _LOG.info(
                "SdfDagBackend scene pool: %d scenes, starting with %s",
                len(self._scene_pool),
                Path(scene_path).name,
            )

    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        is_natural = self._initial_resets_remaining <= 0
        if not is_natural:
            self._initial_resets_remaining -= 1

        if is_natural and self._scene_pool:
            self._episodes_in_scene += 1
            if self._episodes_in_scene >= self._n_actors:
                self._scene_pool_idx = (self._scene_pool_idx + 1) % len(self._scene_pool)
                spawns = self._load_scene(self._scene_pool[self._scene_pool_idx])
                for idx in range(self._n_actors):
                    self._actors[idx].spawn_position = spawns[idx % len(spawns)]
                self._episodes_in_scene = 0

        actor = self._actors[actor_id]
        spawn = actor.spawn_position
        yaw = (2.0 * math.pi * actor_id) / max(self._n_actors, 1)
        actor.pose = RobotPose(
            x=spawn[0],
            y=spawn[1],
            z=spawn[2],
            roll=0.0,
            pitch=0.0,
            yaw=yaw,
            timestamp=time.time(),
        )
        actor.episode_id = episode_id
        actor.step_count = 0
        actor.prev_depth = None
        actor.prev_depth_tensor = None
        actor.episode_return = 0.0
        actor.visited_cells.clear()
        actor.needs_reset = False
        self._mjx_envs[actor_id].reset_velocity()

        depth_batch, semantic_batch, valid_batch = self._cast_actor_batch_tensors((actor_id,))
        _obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
            actor_id=actor_id,
            step_id=0,
            depth_2d=depth_batch[0],
            semantic_2d=semantic_batch[0],
            valid_2d=valid_batch[0],
        )
        return self._materialize_observation(
            actor_id=actor_id,
            step_id=0,
            depth_2d=depth_cpu,
            delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
            semantic_2d=semantic_batch[0].detach().cpu().numpy().astype(np.int32, copy=False),
            valid_2d=valid_batch[0].detach().cpu().numpy().astype(np.bool_, copy=False),
        )

    def reset_tensor(
        self,
        episode_id: int,
        *,
        actor_id: int = 0,
        materialize: bool = False,
    ) -> tuple[Any, DistanceMatrix | None]:
        """Reset one actor and return a tensor-native observation for canonical training."""
        is_natural = self._initial_resets_remaining <= 0
        if not is_natural:
            self._initial_resets_remaining -= 1

        if is_natural and self._scene_pool:
            self._episodes_in_scene += 1
            if self._episodes_in_scene >= self._n_actors:
                self._scene_pool_idx = (self._scene_pool_idx + 1) % len(self._scene_pool)
                spawns = self._load_scene(self._scene_pool[self._scene_pool_idx])
                for idx in range(self._n_actors):
                    self._actors[idx].spawn_position = spawns[idx % len(spawns)]
                self._episodes_in_scene = 0

        actor = self._actors[actor_id]
        spawn = actor.spawn_position
        yaw = (2.0 * math.pi * actor_id) / max(self._n_actors, 1)
        actor.pose = RobotPose(
            x=spawn[0],
            y=spawn[1],
            z=spawn[2],
            roll=0.0,
            pitch=0.0,
            yaw=yaw,
            timestamp=time.time(),
        )
        actor.episode_id = episode_id
        actor.step_count = 0
        actor.prev_depth = None
        actor.prev_depth_tensor = None
        actor.episode_return = 0.0
        actor.visited_cells.clear()
        actor.needs_reset = False
        self._mjx_envs[actor_id].reset_velocity()

        depth_batch, semantic_batch, valid_batch = self._cast_actor_batch_tensors((actor_id,))
        obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
            actor_id=actor_id,
            step_id=0,
            depth_2d=depth_batch[0],
            semantic_2d=semantic_batch[0],
            valid_2d=valid_batch[0],
        )
        published = None
        if materialize:
            published = self._materialize_observation(
                actor_id=actor_id,
                step_id=0,
                depth_2d=depth_cpu,
                delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
                semantic_2d=semantic_batch[0].detach().cpu().numpy().astype(np.int32, copy=False),
                valid_2d=valid_batch[0].detach().cpu().numpy().astype(np.bool_, copy=False),
            )
        return obs_tensor, published

    def step(
        self, action: Action, step_id: int, *, actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        observations, results = self.batch_step((action,), step_id)
        return observations[0], results[0]

    def batch_step(
        self,
        actions: tuple[Action, ...],
        step_id: int,
    ) -> tuple[tuple[DistanceMatrix, ...], tuple[StepResult, ...]]:
        batch_started_at = time.perf_counter()
        observations_by_actor: dict[int, DistanceMatrix] = {}
        results_by_actor: dict[int, StepResult] = {}
        active_actor_ids: list[int] = []
        previous_poses: dict[int, RobotPose] = {}

        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            actor = self._actors[actor_id]
            if actor.needs_reset:
                obs = self.reset(actor.episode_id + 1, actor_id=actor_id)
                observations_by_actor[actor_id] = obs
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=False,
                    reward=0.0,
                    episode_return=0.0,
                    timestamp=time.time(),
                )
                continue

            previous_poses[actor_id] = actor.pose
            actor.pose = self._mjx_envs[actor_id].step_pose(
                actor.pose,
                action,
                time.time(),
                prev_depth=actor.prev_depth,
                max_distance=self._max_distance,
            )
            active_actor_ids.append(actor_id)

        if active_actor_ids:
            depth_batch, semantic_batch, valid_batch = self._cast_actor_batch(tuple(active_actor_ids))
            for batch_idx, actor_id in enumerate(active_actor_ids):
                actor = self._actors[actor_id]
                previous_pose = previous_poses[actor_id]
                depth_2d = depth_batch[batch_idx]
                semantic_2d = semantic_batch[batch_idx]
                valid_2d = valid_batch[batch_idx]

                min_distance = float(np.min(depth_2d)) * self._max_distance
                collision = min_distance < _COLLISION_CLEARANCE
                if collision:
                    actor.pose = previous_pose
                    corrected_depth, corrected_semantic, corrected_valid = self._cast_actor_batch(
                        (actor_id,),
                    )
                    depth_2d = corrected_depth[0]
                    semantic_2d = corrected_semantic[0]
                    valid_2d = corrected_valid[0]

                obs = self._build_observation(
                    actor_id=actor_id,
                    step_id=step_id,
                    depth_2d=depth_2d,
                    semantic_2d=semantic_2d,
                    valid_2d=valid_2d,
                )

                actor.step_count += 1
                truncated = actor.step_count >= self._max_steps_per_episode
                reward = self._compute_reward(
                    actor_id=actor_id,
                    previous_pose=previous_pose,
                    current_pose=actor.pose,
                    collision=collision,
                )
                actor.episode_return += reward
                actor.needs_reset = truncated

                observations_by_actor[actor_id] = obs
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=truncated,
                    reward=reward,
                    episode_return=actor.episode_return,
                    timestamp=time.time(),
                )

        ordered_observations: list[DistanceMatrix] = []
        ordered_results: list[StepResult] = []
        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            ordered_observations.append(observations_by_actor[actor_id])
            ordered_results.append(results_by_actor[actor_id])

        self._record_perf_sample(
            batch_seconds=time.perf_counter() - batch_started_at,
            actor_count=len(actions),
        )
        return tuple(ordered_observations), tuple(ordered_results)

    def batch_step_tensor(
        self,
        actions: tuple[Action, ...],
        step_id: int,
        *,
        publish_actor_ids: tuple[int, ...] = (),
    ) -> tuple[SdfDagTensorStepBatch, tuple[StepResult, ...]]:
        """Step actors and return a CUDA observation batch for canonical training."""
        batch_started_at = time.perf_counter()
        published_observations: dict[int, DistanceMatrix] = {}
        observation_tensors_by_actor: dict[int, Any] = {}
        results_by_actor: dict[int, StepResult] = {}
        active_actor_ids: list[int] = []
        previous_poses: dict[int, RobotPose] = {}
        publish_actor_set = set(publish_actor_ids)

        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            actor = self._actors[actor_id]
            if actor.needs_reset:
                obs_tensor, published = self.reset_tensor(
                    actor.episode_id + 1,
                    actor_id=actor_id,
                    materialize=actor_id in publish_actor_set,
                )
                observation_tensors_by_actor[actor_id] = obs_tensor
                if published is not None:
                    published_observations[actor_id] = published
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=False,
                    reward=0.0,
                    episode_return=0.0,
                    timestamp=time.time(),
                )
                continue

            previous_poses[actor_id] = actor.pose
            actor.pose = self._mjx_envs[actor_id].step_pose(
                actor.pose,
                action,
                time.time(),
                prev_depth=actor.prev_depth,
                max_distance=self._max_distance,
            )
            active_actor_ids.append(actor_id)

        if active_actor_ids:
            depth_batch, semantic_batch, valid_batch = self._cast_actor_batch_tensors(tuple(active_actor_ids))
            min_distances = (
                depth_batch.amin(dim=(1, 2)).mul(self._max_distance).detach().cpu().tolist()
            )

            for batch_idx, actor_id in enumerate(active_actor_ids):
                actor = self._actors[actor_id]
                previous_pose = previous_poses[actor_id]
                depth_2d = depth_batch[batch_idx]
                semantic_2d = semantic_batch[batch_idx]
                valid_2d = valid_batch[batch_idx]

                collision = float(min_distances[batch_idx]) < _COLLISION_CLEARANCE
                if collision:
                    actor.pose = previous_pose
                    corrected_depth, corrected_semantic, corrected_valid = self._cast_actor_batch_tensors(
                        (actor_id,),
                    )
                    depth_2d = corrected_depth[0]
                    semantic_2d = corrected_semantic[0]
                    valid_2d = corrected_valid[0]

                obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
                    actor_id=actor_id,
                    step_id=step_id,
                    depth_2d=depth_2d,
                    semantic_2d=semantic_2d,
                    valid_2d=valid_2d,
                )
                observation_tensors_by_actor[actor_id] = obs_tensor

                if actor_id in publish_actor_set:
                    published_observations[actor_id] = self._materialize_observation(
                        actor_id=actor_id,
                        step_id=step_id,
                        depth_2d=depth_cpu,
                        delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
                        semantic_2d=semantic_2d.detach().cpu().numpy().astype(np.int32, copy=False),
                        valid_2d=valid_2d.detach().cpu().numpy().astype(np.bool_, copy=False),
                    )

                actor.step_count += 1
                truncated = actor.step_count >= self._max_steps_per_episode
                reward = self._compute_reward(
                    actor_id=actor_id,
                    previous_pose=previous_pose,
                    current_pose=actor.pose,
                    collision=collision,
                )
                actor.episode_return += reward
                actor.needs_reset = truncated

                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=truncated,
                    reward=reward,
                    episode_return=actor.episode_return,
                    timestamp=time.time(),
                )

        ordered_observations: list[Any] = []
        ordered_results: list[StepResult] = []
        for idx, action in enumerate(actions):
            actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            ordered_observations.append(observation_tensors_by_actor[actor_id])
            ordered_results.append(results_by_actor[actor_id])

        self._record_perf_sample(
            batch_seconds=time.perf_counter() - batch_started_at,
            actor_count=len(actions),
        )
        if ordered_observations:
            obs_batch = self._torch.stack(ordered_observations, dim=0)
        else:
            obs_batch = self._torch.empty(
                (0, 3, self._az_bins, self._el_bins),
                device=self._device,
                dtype=self._torch.float32,
            )
        return (
            SdfDagTensorStepBatch(
                observation_tensor=obs_batch,
                published_observations=published_observations,
            ),
            tuple(ordered_results),
        )

    def batch_step_tensor_actions(
        self,
        action_tensor: Any,
        step_id: int,
        *,
        publish_actor_ids: tuple[int, ...] = (),
    ) -> tuple[SdfDagTensorStepBatch, tuple[StepResult, ...]]:
        """Step actors from a tensor/array command batch without Python Action objects."""
        action_rows = self._coerce_action_rows(action_tensor)
        batch_started_at = time.perf_counter()
        published_observations: dict[int, DistanceMatrix] = {}
        observation_tensors_by_actor: dict[int, Any] = {}
        results_by_actor: dict[int, StepResult] = {}
        active_actor_ids: list[int] = []
        previous_poses: dict[int, RobotPose] = {}
        publish_actor_set = set(publish_actor_ids)

        for actor_id in range(int(action_rows.shape[0])):
            actor = self._actors[actor_id]
            if actor.needs_reset:
                obs_tensor, published = self.reset_tensor(
                    actor.episode_id + 1,
                    actor_id=actor_id,
                    materialize=actor_id in publish_actor_set,
                )
                observation_tensors_by_actor[actor_id] = obs_tensor
                if published is not None:
                    published_observations[actor_id] = published
                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=False,
                    reward=0.0,
                    episode_return=0.0,
                    timestamp=time.time(),
                )
                continue

            previous_poses[actor_id] = actor.pose
            command = action_rows[actor_id]
            actor.pose = self._mjx_envs[actor_id].step_pose_commands(
                actor.pose,
                command[:3],
                np.array([0.0, 0.0, command[3]], dtype=np.float32),
                time.time(),
                prev_depth=actor.prev_depth,
                max_distance=self._max_distance,
            )
            active_actor_ids.append(actor_id)

        if active_actor_ids:
            depth_batch, semantic_batch, valid_batch = self._cast_actor_batch_tensors(tuple(active_actor_ids))
            min_distances = (
                depth_batch.amin(dim=(1, 2)).mul(self._max_distance).detach().cpu().tolist()
            )

            for batch_idx, actor_id in enumerate(active_actor_ids):
                actor = self._actors[actor_id]
                previous_pose = previous_poses[actor_id]
                depth_2d = depth_batch[batch_idx]
                semantic_2d = semantic_batch[batch_idx]
                valid_2d = valid_batch[batch_idx]

                collision = float(min_distances[batch_idx]) < _COLLISION_CLEARANCE
                if collision:
                    actor.pose = previous_pose
                    corrected_depth, corrected_semantic, corrected_valid = self._cast_actor_batch_tensors(
                        (actor_id,),
                    )
                    depth_2d = corrected_depth[0]
                    semantic_2d = corrected_semantic[0]
                    valid_2d = corrected_valid[0]

                obs_tensor, depth_cpu, delta_tensor = self._consume_actor_observation(
                    actor_id=actor_id,
                    step_id=step_id,
                    depth_2d=depth_2d,
                    semantic_2d=semantic_2d,
                    valid_2d=valid_2d,
                )
                observation_tensors_by_actor[actor_id] = obs_tensor

                if actor_id in publish_actor_set:
                    published_observations[actor_id] = self._materialize_observation(
                        actor_id=actor_id,
                        step_id=step_id,
                        depth_2d=depth_cpu,
                        delta_2d=delta_tensor.detach().cpu().numpy().astype(np.float32, copy=False),
                        semantic_2d=semantic_2d.detach().cpu().numpy().astype(np.int32, copy=False),
                        valid_2d=valid_2d.detach().cpu().numpy().astype(np.bool_, copy=False),
                    )

                actor.step_count += 1
                truncated = actor.step_count >= self._max_steps_per_episode
                reward = self._compute_reward(
                    actor_id=actor_id,
                    previous_pose=previous_pose,
                    current_pose=actor.pose,
                    collision=collision,
                )
                actor.episode_return += reward
                actor.needs_reset = truncated

                results_by_actor[actor_id] = StepResult(
                    step_id=step_id,
                    env_id=actor_id,
                    episode_id=actor.episode_id,
                    done=False,
                    truncated=truncated,
                    reward=reward,
                    episode_return=actor.episode_return,
                    timestamp=time.time(),
                )

        ordered_observations = [observation_tensors_by_actor[actor_id] for actor_id in range(int(action_rows.shape[0]))]
        ordered_results = [results_by_actor[actor_id] for actor_id in range(int(action_rows.shape[0]))]

        self._record_perf_sample(
            batch_seconds=time.perf_counter() - batch_started_at,
            actor_count=int(action_rows.shape[0]),
        )
        obs_batch = self._torch.stack(ordered_observations, dim=0)
        return (
            SdfDagTensorStepBatch(
                observation_tensor=obs_batch,
                published_observations=published_observations,
            ),
            tuple(ordered_results),
        )

    def close(self) -> None:
        self._dag_tensor = None  # type: ignore[assignment]

    @property
    def pose(self) -> RobotPose:
        return self._actors[0].pose

    @property
    def episode_id(self) -> int:
        return self._actors[0].episode_id

    def actor_pose(self, actor_id: int) -> RobotPose:
        return self._actors[actor_id].pose

    def actor_episode_id(self, actor_id: int) -> int:
        return self._actors[actor_id].episode_id

    def perf_snapshot(self) -> SdfDagPerfSnapshot:
        """Return rolling throughput metrics for coarse telemetry and benchmarks."""
        if (
            self._perf_total_batches == 0
            or self._perf_total_actor_steps == 0
            or self._perf_total_batch_seconds <= 0.0
        ):
            return SdfDagPerfSnapshot(
                total_batches=0,
                total_actor_steps=0,
                last_actor_count=self._perf_last_actor_count,
                last_batch_step_ms=0.0,
                ema_batch_step_ms=0.0,
                avg_batch_step_ms=0.0,
                avg_actor_step_ms=0.0,
                sps=0.0,
            )

        avg_batch_seconds = self._perf_total_batch_seconds / self._perf_total_batches
        avg_actor_seconds = self._perf_total_batch_seconds / self._perf_total_actor_steps
        sps = self._perf_total_actor_steps / self._perf_total_batch_seconds
        return SdfDagPerfSnapshot(
            total_batches=self._perf_total_batches,
            total_actor_steps=self._perf_total_actor_steps,
            last_actor_count=self._perf_last_actor_count,
            last_batch_step_ms=self._perf_last_batch_seconds * 1_000.0,
            ema_batch_step_ms=self._perf_ema_batch_seconds * 1_000.0,
            avg_batch_step_ms=avg_batch_seconds * 1_000.0,
            avg_actor_step_ms=avg_actor_seconds * 1_000.0,
            sps=sps,
        )

    @staticmethod
    def _import_required_module(name: str) -> Any:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            msg = f"Required dependency '{name}' is not installed for sdfdag backend"
            raise RuntimeError(msg) from exc

    def _record_perf_sample(self, *, batch_seconds: float, actor_count: int) -> None:
        self._perf_total_batches += 1
        self._perf_total_actor_steps += actor_count
        self._perf_total_batch_seconds += batch_seconds
        self._perf_last_batch_seconds = batch_seconds
        self._perf_last_actor_count = actor_count
        if self._perf_total_batches == 1:
            self._perf_ema_batch_seconds = batch_seconds
            return
        self._perf_ema_batch_seconds = (
            (_PERF_EMA_ALPHA * batch_seconds)
            + ((1.0 - _PERF_EMA_ALPHA) * self._perf_ema_batch_seconds)
        )

    def _require_asset(self) -> GmDagAsset:
        asset = self._asset
        if asset is None:
            msg = "SdfDagBackend asset is not loaded"
            raise RuntimeError(msg)
        return asset

    def _require_dag_tensor(self) -> Any:
        dag_tensor = self._dag_tensor
        if dag_tensor is None:
            msg = "SdfDagBackend DAG tensor is not loaded"
            raise RuntimeError(msg)
        return dag_tensor

    def _load_scene(self, scene_path: str) -> list[tuple[float, float, float]]:
        asset = load_gmdag_asset(Path(scene_path))
        dag_tensor = self._torch.from_numpy(asset.nodes.view(np.int64)).to(
            device=self._device,
            dtype=self._torch.int64,
        )

        self._asset = asset
        self._dag_tensor = dag_tensor
        self._bbox_min = list(asset.bbox_min)
        self._bbox_max = list(asset.bbox_max)

        _LOG.info(
            "Loaded gmdag asset %s (resolution=%d, nodes=%d)",
            asset.path,
            asset.resolution,
            asset.nodes.shape[0],
        )
        return self._find_spawns(self._n_actors)

    def _build_ray_directions(self) -> Any:
        az = np.linspace(0.0, 2.0 * math.pi, self._az_bins, endpoint=False, dtype=np.float32)
        el = np.linspace(math.pi / 2.0, -math.pi / 2.0, self._el_bins, endpoint=True, dtype=np.float32)
        az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
        az_flat = az_grid.reshape(-1)
        el_flat = el_grid.reshape(-1)
        cos_el = np.cos(el_flat)
        dirs = np.stack(
            [
                cos_el * np.sin(az_flat),
                np.sin(el_flat),
                -cos_el * np.cos(az_flat),
            ],
            axis=-1,
        ).astype(np.float32)
        return self._torch.from_numpy(dirs)

    def _cast_actor_batch(
        self,
        actor_ids: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth_t, semantic_t, valid_t = self._cast_actor_batch_tensors(actor_ids)
        depth_2d = (
            depth_t.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        semantic_2d = (
            semantic_t.detach().cpu().numpy().astype(np.int32, copy=False)
        )
        valid_2d = (
            valid_t.detach().cpu().numpy().astype(np.bool_, copy=False)
        )
        return depth_2d, semantic_2d, valid_2d

    def _cast_actor_batch_tensors(
        self,
        actor_ids: tuple[int, ...],
    ) -> tuple[Any, Any, Any]:
        actor_count = len(actor_ids)
        yaws = self._torch.tensor(
            [self._actors[actor_id].pose.yaw for actor_id in actor_ids],
            device=self._device,
            dtype=self._torch.float32,
        )
        positions = self._torch.tensor(
            [
                [self._actors[actor_id].pose.x, self._actors[actor_id].pose.y, self._actors[actor_id].pose.z]
                for actor_id in actor_ids
            ],
            device=self._device,
            dtype=self._torch.float32,
        )

        cos_yaw = self._torch.cos(yaws).unsqueeze(1)
        sin_yaw = self._torch.sin(yaws).unsqueeze(1)
        base_dirs = self._ray_dirs_local.unsqueeze(0).expand(actor_count, -1, -1)
        dirs_world = self._ray_dirs_world[:actor_count]
        dirs_world[..., 0] = base_dirs[..., 0] * cos_yaw + base_dirs[..., 2] * sin_yaw
        dirs_world[..., 1] = base_dirs[..., 1]
        dirs_world[..., 2] = -base_dirs[..., 0] * sin_yaw + base_dirs[..., 2] * cos_yaw

        origins = self._ray_origins[:actor_count]
        origins.copy_(positions.unsqueeze(1).expand(-1, self._n_rays, -1))

        out_distances = self._out_distances[:actor_count]
        out_semantics = self._out_semantics[:actor_count]
        self._torch_sdf.cast_rays(
            self._require_dag_tensor(),
            origins,
            dirs_world,
            out_distances,
            out_semantics,
            self._config.sdf_max_steps,
            self._bbox_min,
            self._bbox_max,
            self._require_asset().resolution,
        )

        distances = out_distances.clamp(min=0.0, max=self._max_distance)
        valid = out_distances <= self._max_distance
        depth_2d = distances.div(self._max_distance).reshape(actor_count, self._az_bins, self._el_bins)
        semantic_2d = out_semantics.reshape(actor_count, self._az_bins, self._el_bins)
        valid_2d = valid.reshape(actor_count, self._az_bins, self._el_bins)
        return depth_2d, semantic_2d, valid_2d

    def _coerce_action_rows(self, action_tensor: Any) -> np.ndarray:
        """Return a contiguous float32 command matrix with shape `(actors, 4)`."""
        if hasattr(action_tensor, "detach"):
            array = action_tensor.detach().cpu().numpy()
        else:
            array = np.asarray(action_tensor)
        rows = np.asarray(array, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < 4:
            msg = "action_tensor must have shape (actors, 4)"
            raise ValueError(msg)
        return rows[:, :4]

    def _consume_actor_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        depth_2d: Any,
        semantic_2d: Any,
        valid_2d: Any,
    ) -> tuple[Any, np.ndarray, Any]:
        """Update actor observation state and build the canonical trainer tensor."""
        del step_id
        actor = self._actors[actor_id]
        previous_depth = actor.prev_depth_tensor
        if previous_depth is None:
            delta_2d = self._torch.zeros_like(depth_2d)
        else:
            delta_2d = depth_2d - previous_depth

        actor.prev_depth_tensor = depth_2d.detach().clone()
        depth_cpu = depth_2d.detach().cpu().numpy().astype(np.float32, copy=False)
        actor.prev_depth = depth_cpu.copy()

        observation_tensor = self._torch.stack(
            (
                depth_2d,
                semantic_2d.to(dtype=self._torch.float32),
                valid_2d.to(dtype=self._torch.float32),
            ),
            dim=0,
        )
        return observation_tensor, depth_cpu, delta_2d

    def _materialize_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        depth_2d: np.ndarray,
        delta_2d: np.ndarray,
        semantic_2d: np.ndarray,
        valid_2d: np.ndarray,
    ) -> DistanceMatrix:
        actor = self._actors[actor_id]
        return DistanceMatrix(
            episode_id=actor.episode_id,
            env_ids=np.array([actor_id], dtype=np.int32),
            matrix_shape=depth_2d.shape,
            depth=depth_2d[np.newaxis, ...],
            delta_depth=delta_2d[np.newaxis, ...],
            semantic=semantic_2d[np.newaxis, ...],
            valid_mask=valid_2d[np.newaxis, ...],
            overhead=None,
            robot_pose=actor.pose,
            step_id=step_id,
            timestamp=time.time(),
        )

    def _build_observation(
        self,
        *,
        actor_id: int,
        step_id: int,
        depth_2d: np.ndarray,
        semantic_2d: np.ndarray,
        valid_2d: np.ndarray,
    ) -> DistanceMatrix:
        actor = self._actors[actor_id]
        delta = depth_2d - actor.prev_depth if actor.prev_depth is not None else np.zeros_like(depth_2d)
        actor.prev_depth = depth_2d.copy()
        actor.prev_depth_tensor = self._torch.from_numpy(actor.prev_depth).to(
            device=self._device,
            dtype=self._torch.float32,
        )
        return self._materialize_observation(
            actor_id=actor_id,
            step_id=step_id,
            depth_2d=depth_2d,
            delta_2d=delta,
            semantic_2d=semantic_2d,
            valid_2d=valid_2d,
        )

    def _compute_reward(
        self,
        *,
        actor_id: int,
        previous_pose: RobotPose,
        current_pose: RobotPose,
        collision: bool,
    ) -> float:
        actor = self._actors[actor_id]
        reward = 0.0
        cell = (int(math.floor(current_pose.x / 2.0)), int(math.floor(current_pose.z / 2.0)))
        if cell not in actor.visited_cells:
            actor.visited_cells.add(cell)
            reward += _EXPLORATION_REWARD

        dx = current_pose.x - previous_pose.x
        dy = current_pose.y - previous_pose.y
        dz = current_pose.z - previous_pose.z
        reward += _PROGRESS_REWARD_SCALE * float(math.sqrt(dx * dx + dy * dy + dz * dz))
        if collision:
            reward += _COLLISION_PENALTY
        return reward

    def _find_spawns(self, count: int) -> list[tuple[float, float, float]]:
        asset = self._require_asset()
        bmin = np.array(asset.bbox_min, dtype=np.float32)
        bmax = np.array(asset.bbox_max, dtype=np.float32)
        extent = bmax - bmin
        x_values = np.linspace(
            bmin[0] + 0.2 * extent[0],
            bmax[0] - 0.2 * extent[0],
            _SPAWN_CANDIDATES_PER_AXIS,
            dtype=np.float32,
        )
        z_values = np.linspace(
            bmin[2] + 0.2 * extent[2],
            bmax[2] - 0.2 * extent[2],
            _SPAWN_CANDIDATES_PER_AXIS,
            dtype=np.float32,
        )
        y_values = np.linspace(
            bmin[1] + 0.2 * extent[1],
            bmax[1] - 0.2 * extent[1],
            _SPAWN_HEIGHT_SAMPLES,
            dtype=np.float32,
        )

        candidates = np.array(
            [[x, y, z] for x in x_values for y in y_values for z in z_values],
            dtype=np.float32,
        )
        probe_dirs = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        origins = self._torch.from_numpy(np.repeat(candidates[:, None, :], probe_dirs.shape[0], axis=1)).to(
            self._device
        )
        dirs = self._torch.from_numpy(np.repeat(probe_dirs[None, :, :], candidates.shape[0], axis=0)).to(
            self._device
        )
        out_distances = self._torch.empty(
            (candidates.shape[0], probe_dirs.shape[0]),
            device=self._device,
            dtype=self._torch.float32,
        )
        out_semantics = self._torch.empty(
            (candidates.shape[0], probe_dirs.shape[0]),
            device=self._device,
            dtype=self._torch.int32,
        )
        self._torch_sdf.cast_rays(
            self._require_dag_tensor(),
            origins,
            dirs,
            out_distances,
            out_semantics,
            max(32, min(self._config.sdf_max_steps, 128)),
            self._bbox_min,
            self._bbox_max,
            asset.resolution,
        )
        distances = out_distances.detach().cpu().numpy().astype(np.float32, copy=False)
        scores: list[tuple[float, tuple[float, float, float]]] = []
        for idx, candidate in enumerate(candidates):
            side_clearance = float(np.min(distances[idx, [0, 1, 4, 5]]))
            ceiling_clearance = float(distances[idx, 2])
            floor_distance = float(distances[idx, 3])
            floor_score = 1.0 - min(abs(floor_distance - 1.5) / 1.5, 1.0)
            score = min(side_clearance, 5.0) + 0.5 * min(ceiling_clearance, 5.0) + floor_score
            scores.append((score, (float(candidate[0]), float(candidate[1]), float(candidate[2]))))

        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores:
            center = 0.5 * (bmin + bmax)
            return [(float(center[0]), float(center[1]), float(center[2]))]

        selected = [spawn for _, spawn in scores[: max(count, 1)]]
        _LOG.info("Selected %d spawn candidates for sdfdag backend", len(selected))
        return selected
