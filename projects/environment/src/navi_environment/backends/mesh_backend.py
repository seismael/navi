"""MeshSceneBackend — loads .glb/.obj meshes via trimesh for equirectangular raycasting.

Provides a ``SimulatorBackend`` that works with real scene meshes
(e.g. ReplicaCAD .glb files from Meta's Habitat ecosystem) without
requiring ``habitat-sim``.  Uses ``trimesh`` for mesh loading and
ray-mesh intersection.

Usage::

    navi-environment serve --backend mesh --habitat-scene path/to/scene.glb

Generates equirectangular ``(Az, El)`` depth + semantic observations
with the same canonical ``(1, Az, El)`` shape as every other backend.

Drone controls
--------------
The agent is modelled as a **2.5D auto-level quadcopter**: it can
fly forward/backward, strafe left/right, change altitude, and yaw.
Roll and pitch are zeroed (auto-stabilised).

Steering convention
~~~~~~~~~~~~~~~~~~~
The policy outputs **normalised steering** in [-1, 1].  The backend
multiplies by drone speed parameters to get physical velocity.
Forward speed is **dynamic**: it scales with the minimum depth in
the front hemisphere of the previous observation.

- ``linear[0]``  x ``drone_max_speed x speed_factor``  -> forward m/s
- ``linear[1]``  x ``drone_climb_rate x speed_factor``  -> vertical m/s
- ``linear[2]``  x ``drone_strafe_speed x speed_factor`` -> lateral m/s
- ``angular[2]`` x ``drone_yaw_rate``                   -> yaw rad/s

``speed_factor = clamp(min_front_depth / safe_threshold, 0.05, 1.0)``

In open space the drone reaches ``drone_max_speed``; near walls it
automatically slows to 5 % of max.  Yaw is never throttled so the
drone can always turn to escape.  Changing ``drone_max_speed`` does
**not** require retraining.

Displacement per tick = velocity x ``physics_dt`` (default 0.02 s = 50 Hz).
A first-order exponential smoothing filter simulates drone inertia.

Episode management
------------------
Episodes are truncated after ``_MAX_STEPS_PER_EPISODE`` steps.
A stuck detector monitors displacement over a sliding window;
if the drone fails to move for ``_STUCK_CONSECUTIVE`` steps it
gets an auto-nudge toward the clearest direction.

Scene pool cycling
------------------
When ``scene_pool`` is set in the config, the backend automatically
cycles through scenes.  Every ``n_actors`` natural episode completions
(collisions or truncations) it loads the next ``.glb`` from the pool,
recomputes spawns, and resets all actors.  This gives maximum
environment diversity during training without restarting the server.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from navi_contracts import (
    Action,
    DistanceMatrix,
    RobotPose,
    StepResult,
)
from navi_environment.backends.base import SimulatorBackend

if TYPE_CHECKING:
    from navi_environment.config import EnvironmentConfig

__all__: list[str] = ["MeshSceneBackend"]

_log = logging.getLogger(__name__)

# ── Reward constants ─────────────────────────────────────────────────
_EXPLORATION_REWARD: float = 0.3
_COLLISION_PENALTY: float = -5.0
_VOID_TAX: float = -0.02        # Penalty for staring into the void
_STRUCTURE_WEIGHT: float = 0.1  # Weight for depth variance (structure attraction)
_CIRCLING_PENALTY: float = -0.5
_CIRCLING_WINDOW: int = 200  # physics ticks (~4 s at dt=0.02)
_PROXIMITY_CLEAR_BONUS: float = 0.15

# ── Episode limits ───────────────────────────────────────────────────
_MAX_STEPS_PER_EPISODE: int = 2_000
_STUCK_CONSECUTIVE: int = 8

# ── Drone physics ────────────────────────────────────────────────────
_SPAWN_ALTITUDE: float = 1.5   # height above floor for spawn points (m)
_MIN_ALTITUDE: float = 0.3     # min clearance above floor (m)
_MAX_ALTITUDE: float = 15.0    # max height above floor (m)
_COLLISION_CLEARANCE: float = 0.4
_NUDGE_DISTANCE: float = 0.6

# ── Dynamic speed scaling ────────────────────────────────────────────
# Forward (and lateral/climb) speed scales with proximity to obstacles.
# The front hemisphere (±45° azimuth) of the previous depth map
# determines a speed factor in [_MIN_SPEED_FACTOR, 1.0].
# Threshold is in **metres** — depth is un-normalised before comparison.
_SAFE_DEPTH_THRESHOLD: float = 3.0   # physical metres
_MIN_SPEED_FACTOR: float = 0.05      # never below 5 % of max speed

# ── Semantic IDs for mesh faces ──────────────────────────────────────
_SEM_FLOOR: int = 2
_SEM_WALL: int = 1
_SEM_CEILING: int = 3
_SEM_OBSTACLE: int = 6


@dataclass
class _ActorState:
    """Per-actor mutable state within a shared scene."""

    pose: RobotPose
    episode_id: int = 0
    step_count: int = 0
    prev_depth: NDArray[np.float32] | None = None
    episode_return: float = 0.0
    visited_cells: set[tuple[int, int]] = field(default_factory=set)
    yaw_history: list[float] = field(default_factory=list)
    pos_history: list[tuple[float, float]] = field(default_factory=list)
    stuck_counter: int = 0
    floor_y: float = 0.0
    needs_reset: bool = False
    scene_changed: bool = False  # True = forced reset from scene switch
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    prev_linear: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    prev_angular: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    # Pre-allocated raycasting buffers (populated in backend __init__)
    ray_origins: NDArray[np.float32] | None = None
    ray_dirs_rotated: NDArray[np.float32] | None = None
    dist_flat: NDArray[np.float32] | None = None
    sem_flat: NDArray[np.int32] | None = None
    valid_flat: NDArray[np.bool_] | None = None


class MeshSceneBackend(SimulatorBackend):
    """SimulatorBackend backed by trimesh mesh raycasting.

    Loads a triangle mesh from a ``.glb``, ``.obj``, or ``.ply`` file
    and performs equirectangular raycasting to produce spherical depth
    and semantic observations identical in shape to those from
    ``VoxelBackend`` or ``HabitatBackend``.

    The agent moves as a 2.5D auto-levelling quadcopter with
    altitude-hold and stuck-escape logic.
    """

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config
        self._az_bins = config.azimuth_bins
        self._el_bins = config.elevation_bins
        self._max_distance = config.max_distance
        self._n_actors = config.n_actors
        self._dt: float = config.physics_dt
        self._smoothing: float = 0.3  # velocity EMA momentum
        self._steps_per_decision: int = config.steps_per_decision

        # Drone speed scales (normalised steering → m/s / rad/s)
        self._drone_max_speed: float = config.drone_max_speed
        self._drone_climb: float = config.drone_climb_rate
        self._drone_strafe: float = config.drone_strafe_speed
        self._drone_yaw: float = config.drone_yaw_rate

        # Scene pool for per-episode cycling
        self._scene_pool: list[str] = list(config.scene_pool) if config.scene_pool else []
        self._scene_pool_idx: int = 0
        self._episodes_in_scene: int = 0
        self._total_scenes_visited: int = 0
        # The environment server calls reset() once per actor during start().
        # Those initial resets must NOT count as episode completions.
        self._initial_resets_remaining: int = self._n_actors

        # Determine first scene to load
        scene_path = (
            self._scene_pool[0] if self._scene_pool
            else config.habitat_scene
        )
        if not scene_path:
            msg = "MeshSceneBackend requires --habitat-scene path or a scene_pool"
            raise ValueError(msg)
        self._current_scene_path: str = scene_path

        # Pre-compute ray directions (scene-independent)
        self._ray_dirs = self._build_ray_directions()

        # Load the first scene mesh
        self._mesh: Any = None
        self._face_semantics: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self._load_scene(scene_path)

        # Find valid spawn positions (one per actor if possible)
        spawns = self._find_spawns(self._n_actors)

        # Per-actor state (N actors share the same mesh)
        self._actors: dict[int, _ActorState] = {}
        n_rays = self._az_bins * self._el_bins
        for i in range(self._n_actors):
            sp = spawns[i % len(spawns)]
            sx, sy, sz = sp
            # Diversify yaw when actors share a spawn
            yaw_offset = (2.0 * math.pi * i) / max(self._n_actors, 1)
            # Pre-allocate raycasting buffers for this actor
            ray_origins = np.zeros((n_rays, 3), dtype=np.float32)
            ray_dirs_rotated = np.zeros((n_rays, 3), dtype=np.float32)
            dist_flat = np.full(n_rays, self._max_distance, dtype=np.float32)
            sem_flat = np.zeros(n_rays, dtype=np.int32)
            valid_flat = np.zeros(n_rays, dtype=np.bool_)
            self._actors[i] = _ActorState(
                pose=RobotPose(
                    x=sx, y=sy, z=sz,
                    roll=0.0, pitch=0.0, yaw=yaw_offset,
                    timestamp=0.0,
                ),
                floor_y=sy - _SPAWN_ALTITUDE,
                spawn_position=sp,
                ray_origins=ray_origins,
                ray_dirs_rotated=ray_dirs_rotated,
                dist_flat=dist_flat,
                sem_flat=sem_flat,
                valid_flat=valid_flat,
            )

        if self._scene_pool:
            _log.info(
                "Scene pool: %d scenes, starting with %s",
                len(self._scene_pool), Path(scene_path).name,
            )

    # ------------------------------------------------------------------
    # SimulatorBackend API
    # ------------------------------------------------------------------

    def reset(self, episode_id: int = 0, *, actor_id: int = 0) -> DistanceMatrix:
        """Reset actor to a valid spawn point.

        Optionally cycles the scene if ``n_actors`` episodes have finished.
        """
        # 1. Natural scene cycling
        # We only cycle when a "natural" episode ends (collision or truncation).
        # Initial resets during server startup are ignored.
        is_natural = self._initial_resets_remaining <= 0
        if not is_natural:
            self._initial_resets_remaining -= 1

        if is_natural and self._scene_pool:
            self._episodes_in_scene += 1
            if self._episodes_in_scene >= self._n_actors:
                # Time to cycle scene!
                self._scene_pool_idx = (self._scene_pool_idx + 1) % len(self._scene_pool)
                self._load_scene(self._scene_pool[self._scene_pool_idx])
                # All actors need a new spawn on the new mesh
                spawns = self._find_spawns(self._n_actors)
                for i in range(self._n_actors):
                    self._actors[i].spawn_position = spawns[i % len(spawns)]
                    self._actors[i].scene_changed = True
                self._episodes_in_scene = 0

        # 2. Reset mutable actor state
        a = self._actors[actor_id]
        sx, sy, sz = a.spawn_position
        a.pose = RobotPose(
            x=sx, y=sy, z=sz,
            roll=0.0, pitch=0.0,
            yaw=(2.0 * math.pi * actor_id) / max(self._n_actors, 1),
            timestamp=time.time(),
        )
        a.episode_id = episode_id
        a.step_count = 0
        a.prev_depth = None
        a.episode_return = 0.0
        a.visited_cells.clear()
        a.yaw_history.clear()
        a.pos_history.clear()
        a.stuck_counter = 0
        a.needs_reset = False
        a.scene_changed = False
        a.prev_linear.fill(0)
        a.prev_angular.fill(0)

        return self._build_observation(step_id=0, actor_id=actor_id)

    def step(
        self, action: Action, step_id: int, *, actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        """Apply 2.5D kinematics, detect collision, and return observation."""
        a = self._actors[actor_id]
        if a.needs_reset:
            obs = self.reset(a.episode_id + 1, actor_id=actor_id)
            # Return dummy result; actual first step happens next call
            return obs, StepResult(
                step_id=step_id, env_id=actor_id, done=False,
                episode_id=a.episode_id,
                truncated=False, reward=0.0, episode_return=0.0,
                timestamp=time.time(),
            )

        # 1. Physics: update pose
        self._step_actor_kinematics(action, actor_id)

        # 2. Perception: cast rays and build result
        depth_2d, semantic_2d, valid_2d = self._cast_rays(actor_id=actor_id)
        return self._build_post_step(actor_id, depth_2d, semantic_2d, valid_2d, step_id)

    def _build_post_step(
        self, actor_id: int, depth_2d: np.ndarray, semantic_2d: np.ndarray,
        valid_2d: np.ndarray, step_id: int
    ) -> tuple[DistanceMatrix, StepResult]:
        """Common logic for collision detection, reward and results building."""
        a = self._actors[actor_id]

        # Delta-depth
        delta = depth_2d - a.prev_depth if a.prev_depth is not None else np.zeros_like(depth_2d)
        a.prev_depth = depth_2d.copy()

        # Overhead minimap
        overhead = self._build_overhead(depth_2d, valid_2d, actor_id=actor_id) if self._config.compute_overhead else None

        obs = DistanceMatrix(
            episode_id=a.episode_id,
            env_ids=np.array([actor_id], dtype=np.int32),
            matrix_shape=depth_2d.shape,
            depth=depth_2d[np.newaxis, ...],
            delta_depth=delta[np.newaxis, ...],
            semantic=semantic_2d[np.newaxis, ...],
            valid_mask=valid_2d[np.newaxis, ...],
            overhead=overhead,
            robot_pose=a.pose,
            step_id=step_id,
            timestamp=time.time(),
        )

        # 3. Collision & Stuck Detection
        collision = bool(np.any(depth_2d * self._max_distance < _COLLISION_CLEARANCE))

        # Stuck detection
        moved = False
        if len(a.pos_history) > 1:
            dist = math.sqrt((a.pose.x - a.pos_history[-1][0])**2 + (a.pose.z - a.pos_history[-1][1])**2)
            if dist > 0.05:
                moved = True

        if not moved:
            a.stuck_counter += 1
        else:
            a.stuck_counter = 0

        if a.stuck_counter >= _STUCK_CONSECUTIVE:
            self._escape_stuck(actor_id, depth_2d)
            a.stuck_counter = 0

        # 4. Reward & Done
        a.step_count += 1
        truncated = a.step_count >= _MAX_STEPS_PER_EPISODE
        # GHOST-MATRIX PERSISTENCE: Collision no longer ends the episode.
        # This forces the agent to learn to escape geometry using its temporal context.
        done = truncated or a.scene_changed

        reward = self._compute_reward(actor_id, depth_2d, valid_2d, collision)
        a.episode_return += reward
        a.needs_reset = done

        if collision:
            _log.debug("Actor %d collided. Applying nudge.", actor_id)
            # Collision Nudge: Move 0.5m backwards to free the agent
            import dataclasses
            a.pose = dataclasses.replace(
                a.pose,
                x=a.pose.x - 0.5 * math.cos(a.pose.yaw),
                z=a.pose.z - 0.5 * math.sin(a.pose.yaw)
            )

        now = time.time()
        # Update timestamp by creating new pose (frozen)
        a.pose = RobotPose(
            x=a.pose.x, y=a.pose.y, z=a.pose.z,
            roll=a.pose.roll, pitch=a.pose.pitch, yaw=a.pose.yaw,
            timestamp=now,
        )

        result = StepResult(
            step_id=step_id,
            env_id=actor_id,
            episode_id=a.episode_id,
            done=done,
            truncated=truncated,
            reward=reward,
            episode_return=a.episode_return,
            timestamp=now,
        )
        return obs, result

    def close(self) -> None:
        """Release resources."""
        self._mesh = None

    @property
    def episode_id(self) -> int:
        """Current episode ID (from actor 0)."""
        return self._actors[0].episode_id

    @property
    def pose(self) -> RobotPose:
        """Current robot pose (from actor 0)."""
        return self._actors[0].pose

    def actor_pose(self, actor_id: int) -> RobotPose:
        """Return pose for a specific actor."""
        return self._actors[actor_id].pose

    def actor_episode_id(self, actor_id: int) -> int:
        """Return episode ID for a specific actor."""
        return self._actors[actor_id].episode_id

    # ------------------------------------------------------------------
    # Batched stepping (parallel raycasting)
    # ------------------------------------------------------------------

    def batch_step(
        self,
        actions: tuple[Action, ...],
        step_id: int,
    ) -> tuple[tuple[DistanceMatrix, ...], tuple[StepResult, ...]]:
        """Step all actors with parallel kinematics and batched raycasting."""
        n = len(actions)

        # Resolve actor IDs
        actor_ids: list[int] = []
        for idx, action in enumerate(actions):
            aid = int(action.env_ids[0]) if len(action.env_ids) > 0 else idx
            actor_ids.append(aid)

        # Phase 1: Sequential resets & Kinematics (single-threaded loop is faster for simple math)
        for idx in range(n):
            aid = actor_ids[idx]
            a = self._actors[aid]
            if a.needs_reset:
                a.episode_id += 1
                self.reset(a.episode_id, actor_id=aid)

            # Kinematics
            self._step_actor_kinematics(actions[idx], aid)

        # Phase 2: BATCHED RAYCASTING
        n_rays_per_actor = self._az_bins * self._el_bins
        total_rays = n * n_rays_per_actor

        all_origins = np.empty((total_rays, 3), dtype=np.float32)
        all_dirs = np.empty((total_rays, 3), dtype=np.float32)

        # Vectorized generation of origins and ray directions
        poses = np.array([[self._actors[aid].pose.x, self._actors[aid].pose.y, self._actors[aid].pose.z] for aid in actor_ids], dtype=np.float32)
        yaws = np.array([self._actors[aid].pose.yaw for aid in actor_ids], dtype=np.float32)

        all_origins = np.repeat(poses, n_rays_per_actor, axis=0)

        cos_y = np.cos(yaws)
        sin_y = np.sin(yaws)

        rot = np.zeros((n, 3, 3), dtype=np.float32)
        rot[:, 0, 0] = cos_y
        rot[:, 0, 2] = sin_y
        rot[:, 1, 1] = 1.0
        rot[:, 2, 0] = -sin_y
        rot[:, 2, 2] = cos_y

        all_dirs = np.einsum('rj,ncj->nrc', self._ray_dirs, rot).reshape(total_rays, 3)

        try:
            locations, index_ray, index_tri = self._mesh.ray.intersects_location(
                all_origins, all_dirs, multiple_hits=False,
            )
        except Exception:
            _log.warning("Batched raycasting failed")
            locations, index_ray, index_tri = np.empty((0,3)), np.empty(0, dtype=int), np.empty(0, dtype=int)

        all_dists = np.full(total_rays, self._max_distance, dtype=np.float32)
        all_sems = np.zeros(total_rays, dtype=np.int32)
        all_valid = np.zeros(total_rays, dtype=bool)

        if len(index_ray) > 0:
            diffs = locations - all_origins[index_ray]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1)).astype(np.float32)
            in_range = dists <= self._max_distance
            all_dists[index_ray[in_range]] = dists[in_range]
            all_sems[index_ray[in_range]] = self._face_semantics[index_tri[in_range]]
            all_valid[index_ray[in_range]] = True

        # Phase 3: Sequential results building
        observations: list[DistanceMatrix] = []
        results: list[StepResult] = []

        for idx, aid in enumerate(actor_ids):
            start, end = idx * n_rays_per_actor, (idx + 1) * n_rays_per_actor
            d_2d = all_dists[start:end].reshape(self._az_bins, self._el_bins)
            s_2d = all_sems[start:end].reshape(self._az_bins, self._el_bins)
            v_2d = all_valid[start:end].reshape(self._az_bins, self._el_bins)

            dm, res = self._build_post_step(aid, d_2d, s_2d, v_2d, step_id)
            observations.append(dm)
            results.append(res)

        return tuple(observations), tuple(results)

    # ------------------------------------------------------------------
    # Ray directions
    # ------------------------------------------------------------------

    def _build_ray_directions(self) -> NDArray[np.float32]:
        """Pre-build unit ray directions for equirectangular sampling.

        Returns array of shape ``(az_bins * el_bins, 3)`` in world-frame
        (before yaw rotation — rotated per-frame in ``_cast_rays``).
        """
        az = np.linspace(0, 2 * math.pi, self._az_bins, endpoint=False)
        el = np.linspace(
            math.pi / 2, -math.pi / 2, self._el_bins, endpoint=True,
        )
        az_grid, el_grid = np.meshgrid(az, el, indexing="ij")
        az_flat = az_grid.ravel()
        el_flat = el_grid.ravel()

        # Spherical to Cartesian (Y-up)
        cos_el = np.cos(el_flat)
        return np.stack([
            cos_el * np.sin(az_flat),   # X
            np.sin(el_flat),             # Y (up)
            -cos_el * np.cos(az_flat),  # -Z (forward)
        ], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Raycasting
    # ------------------------------------------------------------------


    def _cast_rays(
        self, *, actor_id: int = 0,
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.bool_]]:
        """Cast equirectangular rays and return ``(depth, semantic, valid)``.

        All returned arrays have shape ``(az_bins, el_bins)``.
        """
        a = self._actors[actor_id]
        pos = np.array([a.pose.x, a.pose.y, a.pose.z], dtype=np.float32)
        yaw = a.pose.yaw

        # Rotate ray directions by agent yaw (around Y axis)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        rot = np.array([
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ], dtype=np.float32)

        # Use pre-allocated buffers
        dirs_rotated = a.ray_dirs_rotated
        origins = a.ray_origins
        n_rays = self._az_bins * self._el_bins

        if dirs_rotated is None or origins is None:
            # Fallback for unexpected state
            dirs_rotated = np.dot(self._ray_dirs, rot.T).astype(np.float32)
            origins = np.tile(pos, (n_rays, 1)).astype(np.float32)
        else:
            # Update dirs_rotated and origins in-place
            np.dot(self._ray_dirs, rot.T, out=dirs_rotated)
            origins[:] = pos

        # trimesh ray intersection via mesh.ray (uses rtree spatial index)
        try:
            locations, index_ray, index_tri = self._mesh.ray.intersects_location(
                origins, dirs_rotated, multiple_hits=False,
            )
        except Exception:
            _log.warning("Raycasting failed, returning empty observation")
            # Fallback: return empty
            depth = np.ones((self._az_bins, self._el_bins), dtype=np.float32)
            semantic = np.zeros((self._az_bins, self._el_bins), dtype=np.int32)
            valid = np.zeros((self._az_bins, self._el_bins), dtype=np.bool_)
            return depth, semantic, valid

        # Use pre-allocated result buffers
        dist_flat = a.dist_flat
        sem_flat = a.sem_flat
        valid_flat = a.valid_flat
        if dist_flat is not None:
            dist_flat.fill(self._max_distance)
        if sem_flat is not None:
            sem_flat.fill(0)
        if valid_flat is not None:
            valid_flat.fill(False)

        if len(index_ray) > 0:
            # Compute distances
            diffs = locations - origins[index_ray]
            distances = np.sqrt(np.sum(diffs * diffs, axis=1)).astype(np.float32)

            # Filter to max distance
            in_range = distances <= self._max_distance
            ray_idx = index_ray[in_range]
            tri_idx = index_tri[in_range]
            dists = distances[in_range]

            # For duplicate rays (shouldn't happen with multiple_hits=False),
            # keep nearest
            if len(ray_idx) > 0 and dist_flat is not None and sem_flat is not None and valid_flat is not None:
                dist_flat[ray_idx] = dists
                sem_flat[ray_idx] = self._face_semantics[tri_idx]
                valid_flat[ray_idx] = True

        # Normalize depth to [0, 1]
        depth_norm = np.clip(dist_flat / self._max_distance, 0.0, 1.0) if dist_flat is not None else np.ones(self._az_bins * self._el_bins, dtype=np.float32)

        depth_2d = depth_norm.reshape(self._az_bins, self._el_bins) if dist_flat is not None else np.ones((self._az_bins, self._el_bins), dtype=np.float32)
        semantic_2d = sem_flat.reshape(self._az_bins, self._el_bins) if sem_flat is not None else np.zeros((self._az_bins, self._el_bins), dtype=np.int32)
        valid_2d = valid_flat.reshape(self._az_bins, self._el_bins) if valid_flat is not None else np.zeros((self._az_bins, self._el_bins), dtype=np.bool_)

        return depth_2d, semantic_2d, valid_2d

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self, step_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Cast rays and pack into a DistanceMatrix."""
        depth_2d, semantic_2d, valid_2d = self._cast_rays(actor_id=actor_id)
        dm, _res = self._build_post_step(actor_id, depth_2d, semantic_2d, valid_2d, step_id)
        return dm

    def _build_overhead(
        self,
        _depth_2d: NDArray[np.float32],
        _valid_2d: NDArray[np.bool_],
        *,
        actor_id: int = 0,
    ) -> NDArray[np.uint8]:
        """Build a top-down minimap centred on the drone (not implemented)."""
        return np.zeros((128, 128, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # Physics & Motion
    # ------------------------------------------------------------------

    def _step_actor_kinematics(self, action: Action, actor_id: int) -> None:
        """Update actor pose from action (internal physics)."""
        a = self._actors[actor_id]

        # Dynamic speed factor from previous depth
        # Front hemisphere: indices around bin 0 (±45°)
        # bins = 128 -> bin 0..16 and 112..127
        speed_factor = self._compute_speed_factor(a.prev_depth)

        fwd_v = action.linear_velocity[0, 0] * self._drone_max_speed * speed_factor
        vert_v = action.linear_velocity[0, 1] * self._drone_climb * speed_factor
        lat_v = action.linear_velocity[0, 2] * self._drone_strafe * speed_factor
        yaw_rate = action.angular_velocity[0, 2] * self._drone_yaw

        # Body to World frame conversion (Y-up)
        yaw = a.pose.yaw
        cy, sy = math.cos(yaw), math.sin(yaw)

        # In sim coords: Z is forward (inverted in trimesh logic usually, but let's follow the standard)
        # MeshSceneBackend convention: -Z is Forward.
        vx = fwd_v * sy + lat_v * cy
        vz = -fwd_v * cy + lat_v * sy
        vy = vert_v

        # Inertia (first order smoothing)
        dt = self._dt * self._steps_per_decision
        a.prev_linear = (1.0 - self._smoothing) * a.prev_linear + self._smoothing * np.array([vx, vy, vz], dtype=np.float32)
        a.prev_angular[2] = (1.0 - self._smoothing) * a.prev_angular[2] + self._smoothing * yaw_rate

        # Integrate
        new_x = a.pose.x + a.prev_linear[0] * dt
        new_y = a.pose.y + a.prev_linear[1] * dt
        new_z = a.pose.z + a.prev_linear[2] * dt
        new_yaw = a.pose.yaw + a.prev_angular[2] * dt

        # Altitude constraints
        new_y = max(a.floor_y + _MIN_ALTITUDE, min(a.floor_y + _MAX_ALTITUDE, new_y))

        # Update pose (create new RobotPose as it is frozen)
        a.pose = RobotPose(
            x=new_x, y=new_y, z=new_z,
            roll=0.0, pitch=0.0, yaw=new_yaw,
            timestamp=time.time(),
        )
        a.pos_history.append((new_x, new_z))
        if len(a.pos_history) > 50:
            a.pos_history.pop(0)

    def _compute_speed_factor(
        self, prev_depth: NDArray[np.float32] | None,
    ) -> float:
        """Compute a proximity-based speed factor in [MIN, 1.0]."""
        if prev_depth is None:
            return 1.0

        az_bins = prev_depth.shape[0]
        # Front 90° (±45°)
        n = az_bins // 8
        front_indices = list(range(0, n)) + list(range(az_bins - n, az_bins))

        # Minimum depth in front hemisphere (un-normalised metres)
        front_depths = prev_depth[front_indices, :] * self._max_distance
        min_front = float(np.min(front_depths))

        factor = min_front / _SAFE_DEPTH_THRESHOLD
        return max(_MIN_SPEED_FACTOR, min(1.0, factor))

    def _compute_reward(
        self, actor_id: int, depth_2d: NDArray[np.float32],
        valid_2d: NDArray[np.bool_], collision: bool
    ) -> float:
        """Compute Information Foraging reward."""
        if collision:
            return _COLLISION_PENALTY

        a = self._actors[actor_id]

        # 1. Existential Tax
        reward = _VOID_TAX

        # 2. Structure Attraction (Depth Variance)
        # Higher variance means more geometric complexity
        if np.any(valid_2d):
            v = float(np.var(depth_2d[valid_2d]))
            reward += v * _STRUCTURE_WEIGHT

        # 3. Exploration Reward (Grid-based)
        gx, gz = int(a.pose.x / 2.0), int(a.pose.z / 2.0)
        cell = (gx, gz)
        if cell not in a.visited_cells and np.min(depth_2d) < 0.95:
            # Only reward exploration if scanning a structure
            reward += _EXPLORATION_REWARD
            a.visited_cells.add(cell)

        # 4. Anti-Circling Penalty
        a.yaw_history.append(a.pose.yaw)
        if len(a.yaw_history) > _CIRCLING_WINDOW:
            a.yaw_history.pop(0)
            # Net displacement over the position history
            if len(a.pos_history) > 10:
                dx = a.pose.x - a.pos_history[0][0]
                dz = a.pose.z - a.pos_history[0][1]
                net_displacement = math.sqrt(dx*dx + dz*dz)
                if net_displacement < 1.0:
                    reward += _CIRCLING_PENALTY

        return reward

    def _escape_stuck(self, actor_id: int, depth_2d: NDArray[np.float32]) -> None:
        """Nudge the drone in the clearest direction."""
        a = self._actors[actor_id]
        # Find azimuth with max depth
        max_idx = np.argmax(np.mean(depth_2d, axis=1))
        angle = (max_idx / self._az_bins) * 2.0 * math.pi

        new_x = a.pose.x + _NUDGE_DISTANCE * math.sin(angle)
        new_z = a.pose.z - _NUDGE_DISTANCE * math.cos(angle)

        a.pose = RobotPose(
            x=new_x, y=a.pose.y, z=new_z,
            roll=a.pose.roll, pitch=a.pose.pitch, yaw=a.pose.yaw,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Internal mesh loading
    # ------------------------------------------------------------------

    def _load_scene(self, scene_path: str) -> None:
        """Load a triangle mesh and pre-calculate face semantics."""
        import trimesh
        _log.info("Loading scene: %s", scene_path)
        try:
            self._mesh = trimesh.load(scene_path, force="mesh")
        except Exception as e:
            _log.error("Failed to load scene %s: %s", scene_path, e)
            self._cycle_to_next_scene_recursively()
            return

        n_faces = len(self._mesh.faces)
        if n_faces > 500_000:
            _log.warning("Scene %s is too complex (%d faces > 500k limit). Skipping to prevent simulation hang.", Path(scene_path).stem, n_faces)
            self._cycle_to_next_scene_recursively()
            return

        # Simple semantic heuristic if not present
        # floor = lowest Y, ceiling = highest Y, walls = everything else
        self._face_semantics = np.full(n_faces, _SEM_WALL, dtype=np.int32)

        # Find floor (heuristic: normals pointing up + low Y)
        normals = self._mesh.face_normals
        centers = self._mesh.triangles_center

        is_floor = (normals[:, 1] > 0.9) & (centers[:, 1] < np.percentile(centers[:, 1], 20))
        self._face_semantics[is_floor] = _SEM_FLOOR

        is_ceiling = (normals[:, 1] < -0.9) & (centers[:, 1] > np.percentile(centers[:, 1], 80))
        self._face_semantics[is_ceiling] = _SEM_CEILING

        _log.info("Scene loaded: %s (%d faces, %d vertices)",
                 Path(scene_path).stem, n_faces, len(self._mesh.vertices))

    def _cycle_to_next_scene_recursively(self) -> None:
        """Helper to cycle past broken or extremely dense meshes."""
        if not self._scene_pool:
            _log.error("Dense mesh encountered but NO SCENE POOL available to cycle. Forced to continue with high latency.")
            return
        self._scene_pool_idx = (self._scene_pool_idx + 1) % len(self._scene_pool)
        next_path = self._scene_pool[self._scene_pool_idx]
        self._load_scene(next_path)

    def _find_spawns(self, n: int) -> list[tuple[float, float, float]]:
        """Find N valid spawn points on the floor."""
        # Find floor faces
        floor_indices = np.where(self._face_semantics == _SEM_FLOOR)[0]
        if len(floor_indices) == 0:
            # Fallback to any faces
            floor_indices = np.arange(len(self._face_semantics))

        # Sample points on floor
        sampled = self._mesh.triangles_center[floor_indices]
        if len(sampled) == 0:
            return [(0.0, 2.0, 0.0)] * n

        # Shuffle and return top N
        indices = np.arange(len(sampled))
        np.random.shuffle(indices)

        res: list[tuple[float, float, float]] = []
        for i in range(min(n, len(indices))):
            p = sampled[indices[i]]
            res.append((float(p[0]), float(p[1]) + _SPAWN_ALTITUDE, float(p[2])))

        while len(res) < n:
            res.append(res[0])
        return res
