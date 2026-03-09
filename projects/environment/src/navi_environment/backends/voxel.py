"""Voxel-based simulator backend — wraps the existing procedural pipeline.

Extracts the voxel world generation, kinematic stepping, sliding window,
spherical raycast, and reward computation that were previously inline in
``EnvironmentServer`` into a self-contained ``SimulatorBackend``.

Produces ``DistanceMatrix`` observations in canonical ``(1, Az, El)`` shape
with ``matrix_shape = (azimuth_bins, elevation_bins)`` and depth normalised
to ``[0, 1]``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from navi_contracts import Action, DistanceMatrix, RobotPose, StepResult
from navi_environment.backends.base import SimulatorBackend
from navi_environment.distance_matrix_v2 import DistanceMatrixBuilder
from navi_environment.frustum import FrustumLoader
from navi_environment.lookahead import LookAheadBuffer
from navi_environment.matrix import SparseVoxelGrid
from navi_environment.mjx_env import MjxEnvironment
from navi_environment.sliding_window import SlidingWindow

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from navi_environment.config import EnvironmentConfig
    from navi_environment.generators.base import AbstractWorldGenerator

__all__: list[str] = ["VoxelBackend"]

_LOGGER = logging.getLogger(__name__)

# ── Reward constants ─────────────────────────────────────────────────
_SEM_TARGET: int = 10
_TARGET_DISCOVERY_RADIUS: float = 3.0
_TARGET_MAX_REWARD: float = 5.0
_EXPLORATION_REWARD: float = 0.3
_COLLISION_PENALTY: float = -2.0
_CIRCLING_PENALTY: float = -0.5
_PROGRESS_REWARD_SCALE: float = 0.8
_CIRCLING_WINDOW: int = 200  # physics ticks (~4 s at dt=0.02)


@dataclass
class _VoxelActorState:
    """Per-actor mutable state within a shared voxel world."""

    pose: RobotPose
    episode_id: int = 0
    step_id: int = 0
    episode_step: int = 0
    episode_return: float = 0.0
    visited_cells: set[tuple[int, int]] = field(default_factory=set)
    discovered_targets: set[tuple[int, int, int]] = field(default_factory=set)
    yaw_history: list[float] = field(default_factory=list)
    pos_history: list[tuple[float, float]] = field(default_factory=list)
    needs_reset: bool = False
    prev_depth: np.ndarray | None = None  # per-actor delta-depth state


class VoxelBackend(SimulatorBackend):
    """Procedural voxel world with kinematic pose stepping.

    Delegates world geometry to an :class:`AbstractWorldGenerator`,
    physics to :class:`MjxEnvironment` (body-frame kinematic stepping),
    depth observation to :class:`DistanceMatrixBuilder` + :class:`RaycastEngine`,
    and maintains sliding-window / frustum / lookahead caches.
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        generator: AbstractWorldGenerator,
    ) -> None:
        _LOGGER.info("Initializing VoxelBackend with %d actors.", config.n_actors)
        self._config = config
        self._generator = generator
        self._n_actors = config.n_actors

        # Core data structures (shared across all actors)
        self._grid = SparseVoxelGrid(chunk_size=config.chunk_size)
        self._window = SlidingWindow(self._grid, radius=config.window_radius)
        self._frustum = FrustumLoader(
            chunk_size=config.chunk_size,
            half_angle_deg=45.0,
            near=1,
            far=config.lookahead_margin,
        )
        self._lookahead = LookAheadBuffer(capacity=128)

        self._mjx_env = MjxEnvironment(
            dt=config.physics_dt,
            speed_scales=(
                config.drone_max_speed,
                config.drone_climb_rate,
                config.drone_strafe_speed,
                config.drone_yaw_rate,
            ),
        )
        self._distance_builder = DistanceMatrixBuilder(
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_distance=config.max_distance,
        )
        self._max_steps_per_episode = config.max_steps_per_episode

        # Per-actor state
        spawn = generator.spawn_position()
        self._actors: dict[int, _VoxelActorState] = {}
        for i in range(self._n_actors):
            # Diversify starting yaw so actors explore different directions
            yaw_offset = (2.0 * np.pi * i) / max(self._n_actors, 1)
            self._actors[i] = _VoxelActorState(
                pose=RobotPose(
                    x=spawn[0], y=spawn[1], z=spawn[2],
                    roll=0.0, pitch=0.0, yaw=float(yaw_offset),
                    timestamp=time.time(),
                ),
            )

        # Seed the window around the spawn
        self._window.shift(
            spawn[0], spawn[1], spawn[2],
            self._generator.generate_chunk,
        )
        _LOGGER.debug("VoxelBackend initialized successfully.")

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Reset the voxel world and return the initial observation."""
        a = self._actors[actor_id]
        a.episode_id = episode_id
        a.episode_return = 0.0
        a.visited_cells.clear()
        a.discovered_targets.clear()
        a.yaw_history.clear()
        a.pos_history.clear()
        a.step_id = 0
        a.episode_step = 0
        a.needs_reset = False
        a.prev_depth = None
        self._mjx_env.reset_velocity()

        spawn = self._generator.spawn_position()
        a.pose = RobotPose(
            x=spawn[0], y=spawn[1], z=spawn[2],
            roll=0.0, pitch=0.0, yaw=0.0,
            timestamp=time.time(),
        )
        _LOGGER.debug("Actor %d reset to %s (Episode %d)", actor_id, spawn, episode_id)
        self._window.shift(
            a.pose.x, a.pose.y, a.pose.z,
            self._generator.generate_chunk,
        )
        return self._build_distance_matrix(
            step_id=0, timestamp=time.time(), actor_id=actor_id,
        )

    def step(
        self, action: Action, step_id: int, *, actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        """Apply action, slide window, compute rewards, return observation."""
        a = self._actors[actor_id]
        if a.needs_reset:
            self.reset(a.episode_id + 1, actor_id=actor_id)

        now = time.time()
        previous_pose = a.pose
        new_pose = self._apply_action(action, now, actor_id=actor_id)

        # Collision detection (compensate for dynamic speed factor)
        proposed_motion = float(np.sqrt(
            (new_pose.x - previous_pose.x) ** 2
            + (new_pose.z - previous_pose.z) ** 2
        ))
        linear_cmd = (
            float(action.linear_velocity[0, 0])
            if action.linear_velocity.ndim == 2
            else float(action.linear_velocity[0])
        )
        speed_factor = self._mjx_env._compute_speed_factor(
            a.prev_depth, self._config.max_distance,
        )
        effective_speed = abs(linear_cmd) * self._mjx_env._speed_fwd * speed_factor
        expected_motion = max(1e-3, effective_speed) * self._mjx_env.dt
        collided = linear_cmd > 0.5 and expected_motion > 1e-6 and proposed_motion / expected_motion < 0.15

        if collided:
            _LOGGER.debug(
                "Actor %d collision detected at (%0.2f, %0.2f, %0.2f)",
                actor_id,
                new_pose.x,
                new_pose.y,
                new_pose.z,
            )

        a.pose = new_pose
        a.episode_step += 1

        # Slide the voxel window
        wedge = self._window.shift(
            new_pose.x, new_pose.y, new_pose.z,
            self._get_chunk,
        )
        wedge_voxels = wedge.new_voxels
        if wedge_voxels.shape[0] == 0:
            wedge_voxels = self._local_snapshot(radius_chunks=1)

        # Predictive prefetching
        velocity = np.array([
            new_pose.x - previous_pose.x,
            new_pose.y - previous_pose.y,
            new_pose.z - previous_pose.z,
        ], dtype=np.float32)
        frustum_coords = self._frustum.compute_frustum(
            self._window.center_chunk, velocity,
        )
        self._lookahead.prefetch(frustum_coords, self._generator.generate_chunk)

        _ = wedge_voxels
        _ = wedge.culled_count
        obs = self._build_distance_matrix(
            step_id=step_id, timestamp=now, actor_id=actor_id,
        )

        # ── Reward computation ───────────────────────────────────────
        reward = self._compute_reward(
            previous_pose, new_pose, collided, actor_id=actor_id,
        )

        a.episode_return += reward
        a.step_id = step_id

        # GHOST-MATRIX PERSISTENCE: Collided no longer ends the episode.
        # This allows Mamba memory to persist and learn recovery maneuvers.
        done = False
        truncated = a.episode_step >= self._max_steps_per_episode
        if truncated:
            _LOGGER.info("Actor %d episode truncated: steps=%d, reward=%0.2f",
                        actor_id, a.episode_step, a.episode_return)
            a.needs_reset = True

        if collided:
            _LOGGER.debug(
                "Actor %d in persistent collision standoff. Applying nudge.",
                actor_id,
            )
            a.pose = replace(
                a.pose,
                x=a.pose.x - 0.5 * math.cos(a.pose.yaw),
                z=a.pose.z - 0.5 * math.sin(a.pose.yaw),
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
        """No external resources to release for voxel backend."""

    @property
    def pose(self) -> RobotPose:
        """Current robot pose (actor 0 by default)."""
        return self._actors[0].pose

    @property
    def episode_id(self) -> int:
        """Current episode counter (actor 0 by default)."""
        return self._actors[0].episode_id

    def actor_pose(self, actor_id: int) -> RobotPose:
        """Current robot pose for a specific actor."""
        return self._actors[actor_id].pose

    def actor_episode_id(self, actor_id: int) -> int:
        """Current episode counter for a specific actor."""
        return self._actors[actor_id].episode_id

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        previous_pose: RobotPose,
        new_pose: RobotPose,
        collided: bool,
        *,
        actor_id: int = 0,
    ) -> float:
        """Compute structured reward from the step transition."""
        a = self._actors[actor_id]
        reward = 0.0

        # 1) Target discovery
        reward += self._compute_target_reward(new_pose, actor_id=actor_id)

        # 2) Exploration — new 2x2 floor cells
        cell = (int(np.floor(new_pose.x / 2.0)), int(np.floor(new_pose.z / 2.0)))
        if cell not in a.visited_cells:
            a.visited_cells.add(cell)
            reward += _EXPLORATION_REWARD

        # 3) Progress — forward translation
        dx = new_pose.x - previous_pose.x
        dz = new_pose.z - previous_pose.z
        dy = new_pose.y - previous_pose.y
        progress = float(np.sqrt(dx * dx + dz * dz + dy * dy))
        reward += _PROGRESS_REWARD_SCALE * progress

        # 4) Collision penalty
        if collided:
            reward += _COLLISION_PENALTY

        # 5) Anti-circling — penalise sustained spinning in place.
        #    Uses *net* angular displacement (not absolute travel) so
        #    normal steering micro-corrections don't trigger the penalty.
        a.yaw_history.append(float(new_pose.yaw))
        a.pos_history.append((float(new_pose.x), float(new_pose.z)))
        if len(a.yaw_history) > _CIRCLING_WINDOW:
            a.yaw_history = a.yaw_history[-_CIRCLING_WINDOW:]
            a.pos_history = a.pos_history[-_CIRCLING_WINDOW:]
        if len(a.yaw_history) >= _CIRCLING_WINDOW:
            net_yaw = abs(a.yaw_history[-1] - a.yaw_history[0])
            pos_travel = sum(
                float(np.sqrt(
                    (a.pos_history[i][0] - a.pos_history[i - 1][0]) ** 2
                    + (a.pos_history[i][1] - a.pos_history[i - 1][1]) ** 2
                ))
                for i in range(1, len(a.pos_history))
            )
            if net_yaw > 2.0 * np.pi and pos_travel < 1.0:
                reward += _CIRCLING_PENALTY

        return reward

    def _compute_target_reward(
        self, pose: RobotPose, *, actor_id: int = 0,
    ) -> float:
        """Check proximity to target voxels (semantic ID = 10)."""
        a = self._actors[actor_id]
        voxels = self._local_snapshot(radius_chunks=1)
        if voxels.shape[0] == 0:
            return 0.0

        sem = np.rint(voxels[:, 4]).astype(np.int32)
        target_mask = sem == _SEM_TARGET
        if not np.any(target_mask):
            return 0.0

        target_voxels = voxels[target_mask, :3]
        robot_pos = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        distances: NDArray[np.floating[Any]] = np.linalg.norm(
            target_voxels - robot_pos[None, :], axis=1,
        )
        min_dist = float(np.min(distances))

        if min_dist > _TARGET_DISCOVERY_RADIUS:
            return 0.0

        closest_idx = int(np.argmin(distances))
        target_cell = (
            int(np.floor(float(target_voxels[closest_idx, 0]))),
            int(np.floor(float(target_voxels[closest_idx, 1]))),
            int(np.floor(float(target_voxels[closest_idx, 2]))),
        )

        if target_cell in a.discovered_targets:
            return 0.1

        a.discovered_targets.add(target_cell)
        proximity_factor = 1.0 - (min_dist / _TARGET_DISCOVERY_RADIUS)
        return _TARGET_MAX_REWARD * proximity_factor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_action(
        self, action: Action, timestamp: float, *, actor_id: int = 0,
    ) -> RobotPose:
        """Apply a velocity-based action to the current pose."""
        a = self._actors[actor_id]
        stepped_pose = self._mjx_env.step_pose(
            a.pose, action, timestamp,
            prev_depth=a.prev_depth,
            max_distance=self._config.max_distance,
        )
        proposed_x = stepped_pose.x
        proposed_y = stepped_pose.y
        proposed_z = stepped_pose.z

        constrained_x, constrained_y, constrained_z = self._constrain_translation(
            proposed_x, proposed_y, proposed_z,
            self._config.barrier_distance,
            self._config.collision_probe_radius,
            actor_id=actor_id,
        )

        return RobotPose(
            x=constrained_x, y=constrained_y, z=constrained_z,
            roll=stepped_pose.roll, pitch=stepped_pose.pitch, yaw=stepped_pose.yaw,
            timestamp=timestamp,
        )

    def _constrain_translation(
        self,
        x: float, y: float, z: float,
        barrier_distance: float,
        probe_radius: float,
        *,
        actor_id: int = 0,
    ) -> tuple[float, float, float]:
        """Clamp motion to keep minimum standoff from occupied voxels."""
        if barrier_distance <= 0.0:
            return x, y, z
        a = self._actors[actor_id]
        cs = self._grid.chunk_size
        probe_chunks = max(1, int(np.ceil((probe_radius + barrier_distance + 1.0) / cs)))
        cx = int(np.floor(x / cs))
        cy = int(np.floor(y / cs))
        cz = int(np.floor(z / cs))

        voxels = self._grid.query_region(
            (cx - probe_chunks, cy - probe_chunks, cz - probe_chunks),
            (cx + probe_chunks, cy + probe_chunks, cz + probe_chunks),
        )
        if voxels.shape[0] == 0:
            return x, y, z

        positions = voxels[:, :3]
        density = voxels[:, 3]
        occupied = positions[density > 0.0]
        if occupied.shape[0] == 0:
            return x, y, z

        target = np.array([x, y, z], dtype=np.float32)
        offsets = target[None, :] - occupied
        distances = np.linalg.norm(offsets, axis=1)
        nearest_idx = int(np.argmin(distances))
        nearest_dist = float(distances[nearest_idx])
        if nearest_dist >= barrier_distance:
            return x, y, z

        nearest = occupied[nearest_idx]
        direction = target - nearest
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            fallback = np.array([
                a.pose.x - float(nearest[0]),
                a.pose.y - float(nearest[1]),
                a.pose.z - float(nearest[2]),
            ], dtype=np.float32)
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm < 1e-6:
                return a.pose.x, a.pose.y, a.pose.z
            direction = fallback / fallback_norm
        else:
            direction = direction / norm

        constrained = nearest + direction * barrier_distance
        current = np.array([a.pose.x, a.pose.y, a.pose.z], dtype=np.float32)
        proposed = np.array([x, y, z], dtype=np.float32)
        if float(np.linalg.norm(constrained - current)) > float(np.linalg.norm(proposed - current)):
            return a.pose.x, a.pose.y, a.pose.z
        return float(constrained[0]), float(constrained[1]), float(constrained[2])

    def _get_chunk(self, cx: int, cy: int, cz: int) -> np.ndarray:
        """Get a chunk from lookahead cache or generate on demand."""
        cached = self._lookahead.get((cx, cy, cz))
        if cached is not None:
            return cached
        return self._generator.generate_chunk(cx, cy, cz)

    def _build_distance_matrix(
        self, step_id: int, timestamp: float, *, actor_id: int = 0,
    ) -> DistanceMatrix:
        """Build a DistanceMatrix v2 from local voxels around the robot."""
        a = self._actors[actor_id]
        voxels = self._local_snapshot(radius_chunks=2)

        # Swap per-actor prev_depth into the shared builder so
        # delta-depth is computed correctly for each actor.
        self._distance_builder._prev_depth = a.prev_depth
        dm = self._distance_builder.build(
            voxels=voxels,
            pose=a.pose,
            step_id=step_id,
            timestamp=timestamp,
            episode_id=a.episode_id,
            env_id=actor_id,
        )
        a.prev_depth = self._distance_builder._prev_depth
        return dm

    def _local_snapshot(self, radius_chunks: int) -> np.ndarray:
        """Query a local voxel snapshot around the current center chunk."""
        cx, cy, cz = self._window.center_chunk
        return self._grid.query_region(
            (cx - radius_chunks, cy - radius_chunks, cz - radius_chunks),
            (cx + radius_chunks, cy + radius_chunks, cz + radius_chunks),
        )
