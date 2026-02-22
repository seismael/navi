"""MeshSceneBackend — loads .glb/.obj meshes via trimesh for equirectangular raycasting.

Provides a ``SimulatorBackend`` that works with real scene meshes
(e.g. ReplicaCAD .glb files from Meta's Habitat ecosystem) without
requiring ``habitat-sim``.  Uses ``trimesh`` for mesh loading and
ray-mesh intersection.

Usage::

    navi-section-manager serve --backend mesh --habitat-scene path/to/scene.glb

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
multiplies by drone speed parameters (``drone_speed``, etc.) to get
physical velocity:

- ``linear[0]``  × ``drone_speed``        → forward  (body +X) m/s
- ``linear[1]``  × ``drone_climb_rate``   → vertical (world +Y) m/s
- ``linear[2]``  × ``drone_strafe_speed`` → lateral  (body strafe-right) m/s
- ``angular[2]`` × ``drone_yaw_rate``     → yaw rate (rad/s)

Changing ``drone_speed`` changes how fast the drone flies **without
retraining**.  The same trained model works at any speed.

Displacement per tick = velocity × ``physics_dt`` (default 0.02 s = 50 Hz).
A first-order exponential smoothing filter simulates drone inertia.

Episode management
------------------
Episodes are truncated after ``_MAX_STEPS_PER_EPISODE`` steps.
A stuck detector monitors displacement over a sliding window;
if the drone fails to move for ``_STUCK_CONSECUTIVE`` steps it
gets an auto-nudge toward the clearest direction.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from navi_contracts import (
    Action,
    DistanceMatrix,
    RobotPose,
    StepResult,
)
from navi_section_manager.backends.base import SimulatorBackend

if TYPE_CHECKING:
    from navi_section_manager.config import SectionManagerConfig

__all__: list[str] = ["MeshSceneBackend"]

_log = logging.getLogger(__name__)

# ── Reward constants ─────────────────────────────────────────────────
_EXPLORATION_REWARD: float = 0.3
_COLLISION_PENALTY: float = -1.0
_PROGRESS_REWARD_SCALE: float = 0.8
_CIRCLING_PENALTY: float = -0.5
_CIRCLING_WINDOW: int = 200  # physics ticks (~4 s at dt=0.02)
_PROXIMITY_CLEAR_BONUS: float = 0.15

# ── Episode limits ───────────────────────────────────────────────────
_MAX_STEPS_PER_EPISODE: int = 50_000
_STUCK_CONSECUTIVE: int = 8

# ── Drone physics ────────────────────────────────────────────────────
_HOVER_ALTITUDE: float = 1.5
_ALTITUDE_SPRING: float = 0.3
_COLLISION_CLEARANCE: float = 0.4
_NUDGE_DISTANCE: float = 0.6

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
    spawn_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    prev_linear: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )
    prev_angular: NDArray[np.float32] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32),
    )


class MeshSceneBackend(SimulatorBackend):
    """SimulatorBackend backed by trimesh mesh raycasting.

    Loads a triangle mesh from a ``.glb``, ``.obj``, or ``.ply`` file
    and performs equirectangular raycasting to produce spherical depth
    and semantic observations identical in shape to those from
    ``VoxelBackend`` or ``HabitatBackend``.

    The agent moves as a 2.5D auto-levelling quadcopter with
    altitude-hold and stuck-escape logic.
    """

    def __init__(self, config: SectionManagerConfig) -> None:
        import trimesh  # type: ignore[import-untyped,import-not-found]

        self._config = config
        self._az_bins = config.azimuth_bins
        self._el_bins = config.elevation_bins
        self._max_distance = config.max_distance
        self._n_actors = config.n_actors
        self._dt: float = config.physics_dt
        self._smoothing: float = 0.3  # velocity EMA momentum
        self._steps_per_decision: int = config.steps_per_decision

        # Drone speed scales (normalised steering → m/s / rad/s)
        self._drone_speed: float = config.drone_speed
        self._drone_climb: float = config.drone_climb_rate
        self._drone_strafe: float = config.drone_strafe_speed
        self._drone_yaw: float = config.drone_yaw_rate

        # Load scene mesh
        scene_path = config.habitat_scene
        if not scene_path:
            msg = "MeshSceneBackend requires --habitat-scene path"
            raise ValueError(msg)

        loaded = trimesh.load(scene_path, force="scene")
        mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded

        if not isinstance(mesh, trimesh.Trimesh):
            msg = f"Could not load a triangle mesh from {scene_path}"
            raise TypeError(msg)

        self._mesh: Any = mesh

        # Assign semantic IDs based on face normals
        self._face_semantics = self._classify_faces(mesh)

        # Pre-compute ray directions for equirectangular sampling
        self._ray_dirs = self._build_ray_directions()

        # Find valid spawn positions (one per actor if possible)
        spawns = self._find_spawns(self._n_actors)

        # Per-actor state (N actors share the same mesh)
        self._actors: dict[int, _ActorState] = {}
        for i in range(self._n_actors):
            sp = spawns[i % len(spawns)]
            sx, sy, sz = sp
            # Diversify yaw when actors share a spawn
            yaw_offset = (2.0 * math.pi * i) / max(self._n_actors, 1)
            self._actors[i] = _ActorState(
                pose=RobotPose(
                    x=sx, y=sy, z=sz,
                    roll=0.0, pitch=0.0, yaw=yaw_offset,
                    timestamp=0.0,
                ),
                floor_y=sy - _HOVER_ALTITUDE,
                spawn_position=sp,
            )

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

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

    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Reset to spawn and return the initial observation."""
        a = self._actors[actor_id]
        a.episode_id = episode_id
        a.step_count = 0
        a.episode_return = 0.0
        a.visited_cells.clear()
        a.yaw_history.clear()
        a.pos_history.clear()
        a.prev_depth = None
        a.stuck_counter = 0
        a.needs_reset = False
        a.prev_linear = np.zeros(3, dtype=np.float32)
        a.prev_angular = np.zeros(3, dtype=np.float32)

        sx, sy, sz = a.spawn_position
        a.floor_y = sy - _HOVER_ALTITUDE
        a.pose = RobotPose(
            x=sx, y=sy, z=sz,
            roll=0.0, pitch=0.0, yaw=0.0,
            timestamp=time.time(),
        )
        return self._build_observation(step_id=0, actor_id=actor_id)

    def step(
        self, action: Action, step_id: int, *, actor_id: int = 0,
    ) -> tuple[DistanceMatrix, StepResult]:
        """Apply action, update pose, return observation + result.

        When ``steps_per_decision > 1`` the same action is applied for
        multiple physics sub-ticks before building the observation.
        This simulates the *command-hold* paradigm where the actor
        sets a desired velocity state and the flight controller
        maintains it at high frequency.
        """
        a = self._actors[actor_id]

        # Auto-reset on previous truncation / done
        if a.needs_reset:
            a.episode_id += 1
            self.reset(a.episode_id, actor_id=actor_id)

        prev_pose = a.pose
        collided = False

        # Run N physics sub-ticks with the same action (command-hold)
        for _sub in range(self._steps_per_decision):
            now = time.time()
            a.step_count += 1

            sub_prev = a.pose
            new_pose = self._apply_action(action, now, actor_id=actor_id)

            # Collision detection per sub-tick
            dx = new_pose.x - sub_prev.x
            dz = new_pose.z - sub_prev.z
            actual_motion = math.sqrt(dx * dx + dz * dz)

            lin_vel = action.linear_velocity
            cmd_fwd = float(lin_vel[0, 0]) if lin_vel.ndim == 2 else float(lin_vel[0])
            expected = max(1e-3, abs(cmd_fwd)) * self._dt
            if cmd_fwd > 0.5 and expected > 1e-6 and actual_motion / expected < 0.15:
                collided = True

            # Stuck detection + auto-nudge
            if actual_motion < 0.001:
                a.stuck_counter += 1
            else:
                a.stuck_counter = 0

            if a.stuck_counter >= _STUCK_CONSECUTIVE:
                new_pose = self._nudge_escape(new_pose, now)
                a.stuck_counter = 0

            a.pose = new_pose

            if collided or a.step_count >= _MAX_STEPS_PER_EPISODE:
                break

        obs = self._build_observation(step_id=step_id, actor_id=actor_id)
        reward = self._compute_reward(prev_pose, a.pose, collided, obs, actor_id=actor_id)
        a.episode_return += reward

        # Episode horizon → truncation
        truncated = a.step_count >= _MAX_STEPS_PER_EPISODE
        done = collided
        if truncated or done:
            a.needs_reset = True

        result = StepResult(
            step_id=step_id,
            env_id=actor_id,
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

        dirs_rotated = (rot @ self._ray_dirs.T).T  # (N, 3)
        origins = np.broadcast_to(pos, dirs_rotated.shape).copy()

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

        n_rays = self._az_bins * self._el_bins
        dist_flat = np.full(n_rays, self._max_distance, dtype=np.float32)
        sem_flat = np.zeros(n_rays, dtype=np.int32)
        valid_flat = np.zeros(n_rays, dtype=np.bool_)

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
            if len(ray_idx) > 0:
                dist_flat[ray_idx] = dists
                sem_flat[ray_idx] = self._face_semantics[tri_idx]
                valid_flat[ray_idx] = True

        # Normalize depth to [0, 1]
        depth_norm = np.clip(dist_flat / self._max_distance, 0.0, 1.0)

        depth_2d = depth_norm.reshape(self._az_bins, self._el_bins)
        semantic_2d = sem_flat.reshape(self._az_bins, self._el_bins)
        valid_2d = valid_flat.reshape(self._az_bins, self._el_bins)

        return depth_2d, semantic_2d, valid_2d

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(self, step_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Cast rays and pack into a DistanceMatrix."""
        a = self._actors[actor_id]
        depth_2d, semantic_2d, valid_2d = self._cast_rays(actor_id=actor_id)

        # Delta-depth (temporal change)
        delta = depth_2d - a.prev_depth if a.prev_depth is not None else np.zeros_like(depth_2d)
        a.prev_depth = depth_2d.copy()

        if self._config.compute_overhead:
            overhead = self._build_overhead(depth_2d, valid_2d, actor_id=actor_id)
        else:
            overhead = np.zeros((1, 1, 3), dtype=np.float32)

        return DistanceMatrix(
            episode_id=a.episode_id,
            env_ids=np.array([actor_id], dtype=np.int32),
            matrix_shape=(self._az_bins, self._el_bins),
            depth=depth_2d[np.newaxis, ...].astype(np.float32),
            delta_depth=delta[np.newaxis, ...].astype(np.float32),
            semantic=semantic_2d[np.newaxis, ...].astype(np.int32),
            valid_mask=valid_2d[np.newaxis, ...],
            overhead=overhead,
            robot_pose=a.pose,
            step_id=step_id,
            timestamp=time.time(),
        )

    def _build_overhead(
        self,
        depth: NDArray[np.float32],
        valid: NDArray[np.bool_],
        *,
        actor_id: int = 0,
    ) -> NDArray[np.float32]:
        """Build a 256x256 top-down minimap from depth scan."""
        a = self._actors[actor_id]
        size = 256
        overhead = np.zeros((size, size, 3), dtype=np.float32)

        # Use minimum depth per azimuth across all elevations
        band = np.where(valid, depth, 1.0).min(axis=1)  # (az_bins,)
        band_valid = np.any(valid, axis=1)

        if not np.any(band_valid):
            return overhead

        azimuths = np.linspace(0, 2 * math.pi, self._az_bins, endpoint=False)
        # Add agent yaw to get world-frame azimuths
        azimuths = azimuths + a.pose.yaw

        xs = band * np.cos(azimuths) * self._max_distance
        zs = band * np.sin(azimuths) * self._max_distance

        view_radius = min(self._max_distance, 18.0)
        scale = size / (2.0 * view_radius)
        px = (xs * scale / self._max_distance + size / 2.0).astype(np.int32)
        pz = (zs * scale / self._max_distance + size / 2.0).astype(np.int32)

        mask = band_valid & (px >= 1) & (px < size - 1) & (pz >= 1) & (pz < size - 1)
        px = px[mask]
        pz = pz[mask]
        depths = band[mask]

        # Turbo-style color: near = warm, far = cool
        t = 1.0 - np.clip(depths, 0.0, 1.0)
        r = np.clip(np.where(t > 0.5, 1.0, t * 2.0), 0.0, 1.0)
        g = np.clip(
            np.where(t < 0.25, t * 4.0, np.where(t > 0.75, (1.0 - t) * 4.0, 1.0)),
            0.0, 1.0,
        )
        b = np.clip(np.where(t < 0.5, (0.5 - t) * 2.0, 0.0), 0.0, 1.0)

        # Vectorised 3x3 block fill using fancy indexing
        offsets = np.array([-1, 0, 1], dtype=np.int32)
        oy, ox = np.meshgrid(offsets, offsets, indexing="ij")
        oy_flat = oy.ravel()
        ox_flat = ox.ravel()

        all_py = (pz[:, None] + oy_flat[None, :]).ravel()
        all_px = (px[:, None] + ox_flat[None, :]).ravel()
        all_b = np.repeat(b, 9)
        all_g = np.repeat(g, 9)
        all_r = np.repeat(r, 9)

        overhead[all_py, all_px, 0] = all_b
        overhead[all_py, all_px, 1] = all_g
        overhead[all_py, all_px, 2] = all_r

        # Robot marker — bright cyan
        c = size // 2
        overhead[c - 2: c + 3, c - 2: c + 3, :] = 0.0
        overhead[c - 2: c + 3, c - 2: c + 3, 1] = 1.0
        overhead[c - 2: c + 3, c - 2: c + 3, 2] = 1.0

        return overhead

    # ------------------------------------------------------------------
    # Face classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_faces(mesh: Any) -> NDArray[np.int32]:
        """Classify mesh faces into semantic IDs based on face normals.

        - Upward-facing (normal.y > 0.7) → FLOOR (2)
        - Downward-facing (normal.y < -0.7) → CEILING (3)
        - Mostly horizontal (abs(normal.y) <= 0.7) → WALL (1)
        """
        normals = np.asarray(mesh.face_normals, dtype=np.float32)
        n_faces = len(normals)
        sem = np.full(n_faces, _SEM_WALL, dtype=np.int32)

        up_mask = normals[:, 1] > 0.7
        down_mask = normals[:, 1] < -0.7

        sem[up_mask] = _SEM_FLOOR
        sem[down_mask] = _SEM_CEILING

        return sem

    # ------------------------------------------------------------------
    # Spawn + navigation
    # ------------------------------------------------------------------

    def _find_spawns(self, n: int) -> list[tuple[float, float, float]]:
        """Find up to *n* valid spawn positions inside the mesh.

        Strategy: try several candidate positions (offset from centre)
        and collect all that have clear space around them, up to *n*.
        """
        bounds = self._mesh.bounds  # (2, 3) — min, max
        cx = float((bounds[0, 0] + bounds[1, 0]) / 2.0)
        cz = float((bounds[0, 2] + bounds[1, 2]) / 2.0)
        sx = float(bounds[1, 0] - bounds[0, 0])
        sz = float(bounds[1, 2] - bounds[0, 2])
        above_y = float(bounds[1, 1]) + 5.0

        # Candidate positions: quadrant centres (avoid interior wall junctions)
        candidates = [
            (cx - sx * 0.25, cz - sz * 0.25),  # Room A
            (cx + sx * 0.25, cz + sz * 0.25),  # Room B
            (cx + sx * 0.25, cz - sz * 0.25),  # Room C
            (cx - sx * 0.25, cz + sz * 0.25),  # Hallway
            (cx, cz + sz * 0.15),               # Near centre
        ]

        found: list[tuple[float, float, float]] = []
        for trial_x, trial_z in candidates:
            origin = np.array([[trial_x, above_y, trial_z]], dtype=np.float64)
            direction = np.array([[0.0, -1.0, 0.0]], dtype=np.float64)
            try:
                locs, _, _ = self._mesh.ray.intersects_location(
                    origin, direction, multiple_hits=True,
                )
                if len(locs) > 0:
                    floor_y = float(locs[:, 1].min())
                    spawn_y = floor_y + 1.5
                    # Verify clearance — cast horizontal probe rays
                    probe_origin = np.array(
                        [[trial_x, spawn_y, trial_z]], dtype=np.float64,
                    )
                    clear = True
                    for dx, dz in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        probe_dir = np.array(
                            [[float(dx), 0.0, float(dz)]],
                            dtype=np.float64,
                        )
                        try:
                            pl, _, _ = self._mesh.ray.intersects_location(
                                probe_origin, probe_dir, multiple_hits=False,
                            )
                            if len(pl) > 0:
                                d = float(np.sqrt(np.sum((pl[0] - probe_origin[0]) ** 2)))
                                if d < 0.5:
                                    clear = False
                                    break
                        except Exception:
                            _log.debug("Clearance probe failed for (%s, %s)", dx, dz)
                    if clear:
                        found.append((trial_x, spawn_y, trial_z))
                        if len(found) >= n:
                            return found
            except Exception:
                _log.debug("Spawn candidate (%s, %s) failed", trial_x, trial_z)
                continue

        if found:
            return found

        # Last resort: first candidate at floor + 1.5
        return [(candidates[0][0], float(bounds[0, 1]) + 1.5, candidates[0][1])]

    def _apply_action(
        self, action: Action, timestamp: float, *, actor_id: int = 0,
    ) -> RobotPose:
        """Apply a normalised steering command to the current pose.

        The policy outputs steering in [-1, 1].  This method scales
        the steering by drone speed parameters to get physical
        velocity (m/s, rad/s), then integrates by ``dt``.

        - ``linear[0]`` = forward steering  → × ``drone_speed``
        - ``linear[1]`` = vertical steering → × ``drone_climb_rate``
        - ``linear[2]`` = lateral steering  → × ``drone_strafe_speed``
        - ``angular[2]`` = yaw steering     → × ``drone_yaw_rate``

        Includes altitude-hold spring and collision avoidance.
        """
        a = self._actors[actor_id]
        lin = action.linear_velocity
        ang = action.angular_velocity
        if lin.ndim == 2:
            lin = lin[0]
            ang = ang[0]

        # Scale normalised steering [-1, 1] → physical velocity
        fwd = float(lin[0]) * self._drone_speed
        vert = (float(lin[1]) if len(lin) > 1 else 0.0) * self._drone_climb
        lat = (float(lin[2]) if len(lin) > 2 else 0.0) * self._drone_strafe
        yaw_rate = (float(ang[2]) if len(ang) > 2 else 0.0) * self._drone_yaw

        # First-order exponential smoothing (momentum)
        sm = self._smoothing
        s_fwd = (1.0 - sm) * fwd + sm * float(a.prev_linear[0])
        s_vert = (1.0 - sm) * vert + sm * float(a.prev_linear[1])
        s_lat = (1.0 - sm) * lat + sm * float(a.prev_linear[2])
        s_yaw = (1.0 - sm) * yaw_rate + sm * float(a.prev_angular[2])
        a.prev_linear[:] = [s_fwd, s_vert, s_lat]
        a.prev_angular[:] = [0.0, 0.0, s_yaw]

        dt = self._dt
        new_yaw = a.pose.yaw + s_yaw * dt

        # Body-to-world rotation (same convention as MjxEnvironment)
        cos_y = math.cos(a.pose.yaw)
        sin_y = math.sin(a.pose.yaw)
        world_dx = (s_fwd * cos_y - s_lat * sin_y) * dt
        world_dz = (s_fwd * sin_y + s_lat * cos_y) * dt

        # Altitude hold: spring towards hover altitude + command offset
        target_y = a.floor_y + _HOVER_ALTITUDE + s_vert
        alt_error = target_y - a.pose.y
        world_dy = _ALTITUDE_SPRING * alt_error * dt

        # Proposed position
        prop_x = a.pose.x + world_dx
        prop_y = a.pose.y + world_dy
        prop_z = a.pose.z + world_dz

        # Collision avoidance probe
        motion_dist = math.sqrt(world_dx * world_dx + world_dz * world_dz)
        if motion_dist > 0.01:
            direction = np.array(
                [[world_dx / motion_dist, 0.0, world_dz / motion_dist]],
                dtype=np.float64,
            )
            origin = np.array(
                [[a.pose.x, a.pose.y, a.pose.z]],
                dtype=np.float64,
            )
            try:
                locs, _, _ = self._mesh.ray.intersects_location(
                    origin, direction, multiple_hits=False,
                )
                if len(locs) > 0:
                    hit_dist = float(
                        np.sqrt(np.sum((locs[0] - origin[0]) ** 2)),
                    )
                    if hit_dist < motion_dist + _COLLISION_CLEARANCE:
                        safe = max(0.0, hit_dist - _COLLISION_CLEARANCE)
                        ratio = safe / max(motion_dist, 1e-6)
                        prop_x = a.pose.x + world_dx * ratio
                        prop_z = a.pose.z + world_dz * ratio
            except Exception:
                _log.debug("Collision probe failed")

        return RobotPose(
            x=prop_x, y=prop_y, z=prop_z,
            roll=0.0, pitch=0.0, yaw=new_yaw,
            timestamp=timestamp,
        )

    def _nudge_escape(self, pose: RobotPose, timestamp: float) -> RobotPose:
        """Escape a stuck position by moving toward the clearest direction.

        Casts 8 radial probes, finds the direction with the most
        clearance, then applies a small displacement plus a random yaw
        perturbation to break symmetry.
        """
        best_dist = 0.0
        best_dx = 0.0
        best_dz = 0.0
        origin = np.array([[pose.x, pose.y, pose.z]], dtype=np.float64)

        angles = np.linspace(0, 2 * math.pi, 8, endpoint=False)
        for angle in angles:
            ddx = math.cos(angle)
            ddz = math.sin(angle)
            probe_dir = np.array([[ddx, 0.0, ddz]], dtype=np.float64)
            try:
                locs, _, _ = self._mesh.ray.intersects_location(
                    origin, probe_dir, multiple_hits=False,
                )
                d = (
                    float(np.sqrt(np.sum((locs[0] - origin[0]) ** 2)))
                    if len(locs) > 0
                    else self._max_distance
                )
            except Exception:
                d = self._max_distance
            if d > best_dist:
                best_dist = d
                best_dx = ddx
                best_dz = ddz

        nudge = min(_NUDGE_DISTANCE, best_dist * 0.5)
        new_x = pose.x + best_dx * nudge
        new_z = pose.z + best_dz * nudge
        rng = np.random.default_rng()
        new_yaw = pose.yaw + float(rng.uniform(-math.pi / 2, math.pi / 2))

        _log.debug(
            "Nudge escape: dir=(%.2f, %.2f) clearance=%.2f nudge=%.2f",
            best_dx, best_dz, best_dist, nudge,
        )
        return RobotPose(
            x=new_x, y=pose.y, z=new_z,
            roll=0.0, pitch=0.0, yaw=new_yaw,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        prev: RobotPose,
        curr: RobotPose,
        collided: bool,
        obs: DistanceMatrix,
        *,
        actor_id: int = 0,
    ) -> float:
        """Multi-component reward with exploration incentives.

        Components:
        1. Exploration -- new 2x2 floor cell        (+0.3)
        2. Progress -- displacement                  (+0.8 x dist)
        3. Collision penalty                        (-1.0)
        4. Anti-circling (spin in place)            (-0.5)
        5. Proximity clear — flying near obstacles
           without collision                        (+0.15)
        """
        a = self._actors[actor_id]
        reward = 0.0

        # 1) Exploration -- new 2x2 floor cell
        cell = (int(np.floor(curr.x / 2.0)), int(np.floor(curr.z / 2.0)))
        if cell not in a.visited_cells:
            a.visited_cells.add(cell)
            reward += _EXPLORATION_REWARD

        # 2) Progress — forward displacement
        dx = curr.x - prev.x
        dz = curr.z - prev.z
        dy = curr.y - prev.y
        progress = float(math.sqrt(dx * dx + dz * dz + dy * dy))
        reward += _PROGRESS_REWARD_SCALE * progress

        # 3) Collision penalty (softened so drone retries near walls)
        if collided:
            reward += _COLLISION_PENALTY

        # 4) Anti-circling
        a.yaw_history.append(float(curr.yaw))
        a.pos_history.append((float(curr.x), float(curr.z)))
        if len(a.yaw_history) > _CIRCLING_WINDOW:
            a.yaw_history = a.yaw_history[-_CIRCLING_WINDOW:]
            a.pos_history = a.pos_history[-_CIRCLING_WINDOW:]
        if len(a.yaw_history) >= _CIRCLING_WINDOW:
            yaw_travel = sum(
                abs(a.yaw_history[i] - a.yaw_history[i - 1])
                for i in range(1, len(a.yaw_history))
            )
            pos_travel = sum(
                float(math.sqrt(
                    (a.pos_history[i][0] - a.pos_history[i - 1][0]) ** 2
                    + (a.pos_history[i][1] - a.pos_history[i - 1][1]) ** 2,
                ))
                for i in range(1, len(a.pos_history))
            )
            if yaw_travel > 3.0 and pos_travel < 1.0:
                reward += _CIRCLING_PENALTY

        # 5) Proximity clear bonus — near obstacles but not colliding
        depth = obs.depth[0]  # (Az, El)
        valid = obs.valid_mask[0]
        if np.any(valid):
            min_depth = float(np.where(valid, depth, 1.0).min())
            if min_depth < 0.15 and not collided:
                reward += _PROXIMITY_CLEAR_BONUS

        return reward
