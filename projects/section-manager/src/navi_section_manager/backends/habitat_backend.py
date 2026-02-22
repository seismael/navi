"""HabitatBackend — wraps Meta's habitat-sim into Navi's SimulatorBackend interface.

Uses equirectangular depth + semantic sensors at the configured resolution
and delegates conversion to a ``HabitatAdapter`` (``DatasetAdapter`` Protocol)
which transposes raw ``(El, Az)`` observations to canonical ``(1, Az, El)``
shape, normalises depth to ``[0, 1]``, remaps semantic classes, and computes
delta-depth and valid-mask.  The training engine is never modified — the
adapter transforms *to* its format.

Requires ``habitat-sim`` to be installed (conda install -c aihabitat habitat-sim).
"""

from __future__ import annotations

import json
import math
import time
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
from navi_section_manager.backends.base import SimulatorBackend
from navi_section_manager.backends.habitat_adapter import HabitatAdapter
from navi_section_manager.backends.habitat_semantic_lut import HabitatSemanticLUT

if TYPE_CHECKING:
    from navi_section_manager.config import SectionManagerConfig

__all__: list[str] = ["HabitatBackend"]

# Reward constants (same tier system as VoxelBackend)
_EXPLORATION_REWARD: float = 0.3
_COLLISION_PENALTY: float = -2.0
_PROGRESS_REWARD_SCALE: float = 0.8
_GOAL_REACHED_REWARD: float = 10.0
_GOAL_RADIUS: float = 0.2
_CIRCLING_PENALTY: float = -0.5
_CIRCLING_WINDOW: int = 20


class HabitatBackend(SimulatorBackend):
    """SimulatorBackend backed by Meta's habitat-sim.

    Sensor setup:
    - **equirect_depth**: ``EquirectangularSensor`` with ``SensorType.DEPTH``
      at ``(elevation_bins, azimuth_bins)`` — produces float32 depth in metres.
    - **equirect_semantic**: ``EquirectangularSensor`` with ``SensorType.SEMANTIC``
      at the same resolution — produces uint32 instance IDs remapped via the
      semantic LUT to Navi's 0-10 range.

    Coordinate mapping:
    - Habitat: Y-up, forward = -Z.
    - Navi: Y-up, forward = -Z (compatible).
    - Agent rotation around Y-axis maps directly to Navi's yaw.
    """

    def __init__(self, config: SectionManagerConfig) -> None:
        # Lazy import — habitat-sim may not be installed
        import habitat_sim  # type: ignore[import-untyped,import-not-found]

        self._config = config
        self._habitat_sim = habitat_sim

        # Resolution for spherical observations
        self._az_bins = config.azimuth_bins
        self._el_bins = config.elevation_bins
        self._max_distance = config.max_distance

        # Build simulator configuration
        sim_cfg = self._build_sim_config(config)
        self._sim: Any = habitat_sim.Simulator(sim_cfg)

        # Semantic LUT
        self._semantic_lut = HabitatSemanticLUT()
        self._semantic_lut.build_from_scene(self._sim.semantic_scene)

        # Dataset adapter: Habitat raw obs → canonical (1, Az, El) arrays
        self._adapter = HabitatAdapter(
            az_bins=self._az_bins,
            el_bins=self._el_bins,
            max_distance=self._max_distance,
            semantic_lut=self._semantic_lut,
        )

        # Episode state
        self._episode_id: int = 0
        self._episode_step: int = 0
        self._max_steps_per_episode: int = config.max_steps_per_episode
        self._physics_dt: float = config.physics_dt

        # Drone speed scales (normalised steering → m/s / rad/s)
        self._drone_speed: float = config.drone_speed
        self._drone_climb: float = config.drone_climb_rate
        self._drone_strafe: float = config.drone_strafe_speed
        self._drone_yaw: float = config.drone_yaw_rate
        self._needs_reset: bool = False
        self._pose = RobotPose(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, timestamp=0.0)

        # Reward tracking
        self._episode_return: float = 0.0
        self._visited_cells: set[tuple[int, int]] = set()
        self._yaw_history: list[float] = []
        self._pos_history: list[tuple[float, float]] = []
        self._goal_position: NDArray[np.float32] | None = None

        # PointNav episodes loaded from dataset config (JSON)
        self._episodes: list[dict[str, Any]] = self._load_episodes(config)

    # ------------------------------------------------------------------
    # SimulatorBackend interface
    # ------------------------------------------------------------------

    @property
    def pose(self) -> RobotPose:
        """Current robot pose in world coordinates."""
        return self._pose

    @property
    def episode_id(self) -> int:
        """Current episode identifier."""
        return self._episode_id

    def reset(self, episode_id: int, *, actor_id: int = 0) -> DistanceMatrix:
        """Reset the simulator and return an initial observation."""
        self._episode_id = episode_id
        obs = self._sim.reset()

        # If we have dataset episodes, place the agent at the episode start
        if self._episodes:
            idx = episode_id % len(self._episodes)
            ep = self._episodes[idx]
            start_pos = ep.get("start_position")
            start_rot = ep.get("start_rotation")
            if start_pos is not None:
                agent = self._sim.get_agent(0)
                state = agent.get_state()
                state.position = np.array(start_pos, dtype=np.float32)
                if start_rot is not None:
                    # habitat uses [x, y, z, w] quaternion
                    qr = np.array(start_rot, dtype=np.float32)
                    state.rotation = self._habitat_sim.utils.common.quat_from_coeffs(qr)
                agent.set_state(state)
                obs = self._sim.get_sensor_observations()

        # Extract agent state
        agent_state = self._sim.get_agent(0).get_state()
        self._pose = self._agent_state_to_pose(agent_state)

        # Parse goal from episode if available
        self._goal_position = self._extract_goal()

        # Reset reward tracking
        self._episode_return = 0.0
        self._episode_step = 0
        self._needs_reset = False
        self._visited_cells.clear()
        self._yaw_history.clear()
        self._pos_history.clear()
        self._adapter.reset()

        return self._obs_to_distance_matrix(obs, step_id=0)

    def step(self, action: Action, step_id: int, *, actor_id: int = 0) -> tuple[DistanceMatrix, StepResult]:
        """Apply a continuous velocity action and return the new observation + result."""
        if self._needs_reset:
            self.reset(self._episode_id + 1)

        now = time.time()
        previous_pose = self._pose

        # Convert Navi Action (4-DOF: linear xyz + yaw rate) to Habitat velocity control
        habitat_action = self._action_to_habitat(action)
        obs = self._sim.step(habitat_action)

        # Update pose from agent state
        agent_state = self._sim.get_agent(0).get_state()
        self._pose = self._agent_state_to_pose(agent_state)
        self._episode_step += 1

        # Build observation
        dm = self._obs_to_distance_matrix(obs, step_id=step_id)

        # Compute reward
        reward = self._compute_reward(previous_pose, self._pose, obs)
        self._episode_return += reward

        # Check termination
        done = self._check_goal_reached()
        collided = (
            self._sim is not None
            and hasattr(self._sim, "previous_step_collided")
            and self._sim.previous_step_collided
        )
        done = done or collided
        truncated = self._episode_step >= self._max_steps_per_episode
        if done or truncated:
            self._needs_reset = True

        result = StepResult(
            step_id=step_id,
            env_id=actor_id,
            done=done,
            truncated=truncated,
            reward=reward,
            episode_return=self._episode_return,
            timestamp=now,
        )
        return dm, result

    def close(self) -> None:
        """Shut down the Habitat simulator."""
        if self._sim is not None:
            self._sim.close()
            self._sim = None

    # ------------------------------------------------------------------
    # Habitat configuration
    # ------------------------------------------------------------------

    def _build_sim_config(self, config: SectionManagerConfig) -> Any:
        """Build a habitat_sim.Configuration for the simulator."""
        habitat_sim = self._habitat_sim

        # Backend config
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = config.habitat_scene
        backend_cfg.enable_physics = True

        # Agent config
        agent_cfg = habitat_sim.agent.AgentConfiguration()

        # Equirectangular depth sensor
        depth_spec = habitat_sim.EquirectangularSensorSpec()
        depth_spec.uuid = "equirect_depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [config.elevation_bins, config.azimuth_bins]
        depth_spec.position = [0.0, 1.5, 0.0]  # 1.5m above floor
        depth_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR

        # Equirectangular semantic sensor
        sem_spec = habitat_sim.EquirectangularSensorSpec()
        sem_spec.uuid = "equirect_semantic"
        sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        sem_spec.resolution = [config.elevation_bins, config.azimuth_bins]
        sem_spec.position = [0.0, 1.5, 0.0]
        sem_spec.sensor_subtype = habitat_sim.SensorSubType.EQUIRECTANGULAR

        agent_cfg.sensor_specifications = [depth_spec, sem_spec]

        # Enable continuous velocity control
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=0.0),
            ),
            "velocity_control": habitat_sim.agent.ActionSpec(
                "velocity_control",
                habitat_sim.agent.ActuationSpec(amount=0.0),
            ),
        }

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        return cfg  # noqa: RET504

    # ------------------------------------------------------------------
    # Coordinate / action conversion
    # ------------------------------------------------------------------

    def _agent_state_to_pose(self, agent_state: Any) -> RobotPose:
        """Convert habitat agent state to Navi RobotPose."""
        pos = agent_state.position  # [x, y, z]
        rot = agent_state.rotation  # quaternion (w, x, y, z) or (x, y, z, w)

        # habitat-sim uses [w, x, y, z] quaternion ordering via magnum
        # Extract yaw (rotation around Y-axis) from quaternion
        qw = float(rot.scalar)
        qx = float(rot.vector[0])
        qy = float(rot.vector[1])
        qz = float(rot.vector[2])

        # Yaw from quaternion (Y-axis rotation)
        siny_cosp = 2.0 * (qw * qy + qx * qz)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qx * qx)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Pitch from quaternion
        sinp = 2.0 * (qw * qx - qz * qy)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))

        return RobotPose(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            roll=0.0,
            pitch=pitch,
            yaw=yaw,
            timestamp=time.time(),
        )

    def _action_to_habitat(self, action: Action) -> dict[str, Any]:
        """Convert Navi's 4-DOF Action to Habitat velocity control.

        Navi Action layout:
        - linear_velocity: (batch, 3) — [forward, strafe, vertical]
        - angular_velocity: (batch, 3) — [roll_rate, pitch_rate, yaw_rate]

        We use habitat-sim's velocity_control interface:
        - linear_velocity: [x, y, z] in agent-local frame
        - angular_velocity: [x, y, z] in agent-local frame (Y = yaw)
        """
        # Extract from batch dimension
        if action.linear_velocity.ndim == 2:
            lin = action.linear_velocity[0]  # (3,)
            ang = action.angular_velocity[0]  # (3,)
        else:
            lin = action.linear_velocity
            ang = action.angular_velocity

        forward = float(lin[0]) * self._drone_speed
        strafe = (float(lin[1]) if len(lin) > 1 else 0.0) * self._drone_strafe
        vertical = (float(lin[2]) if len(lin) > 2 else 0.0) * self._drone_climb
        yaw_rate = (float(ang[2]) if len(ang) > 2 else 0.0) * self._drone_yaw

        # Direct physics-based velocity control
        agent = self._sim.get_agent(0)
        agent_state = agent.get_state()

        # Build rotation matrix from current yaw to convert local→world velocities
        yaw = self._pose.yaw
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Agent-local forward = -Z in Habitat
        # Agent-local strafe = +X in Habitat
        world_vx = strafe * cos_y + forward * sin_y
        world_vz = -forward * cos_y + strafe * sin_y

        # Apply translation
        dt = self._physics_dt
        new_pos = np.array([
            agent_state.position[0] + world_vx * dt,
            agent_state.position[1] + vertical * dt,
            agent_state.position[2] + world_vz * dt,
        ])

        # Apply yaw rotation
        new_yaw = yaw + yaw_rate * dt

        # Convert yaw to quaternion around Y-axis
        half = new_yaw * 0.5
        quat = self._habitat_sim.utils.common.quat_from_angle_axis(
            new_yaw, np.array([0.0, 1.0, 0.0])
        )

        agent_state.position = new_pos
        agent_state.rotation = quat

        # Snap to navmesh to prevent falling through floors
        snapped_pos = self._sim.pathfinder.snap_point(new_pos)
        if not np.isnan(snapped_pos).any():
            agent_state.position = snapped_pos

        agent.set_state(agent_state)

        # Return a no-op action since we set state directly
        _ = half  # suppress unused
        return {"action": "move_forward", "action_args": {"amount": 0.0}}

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _obs_to_distance_matrix(
        self, obs: dict[str, NDArray[Any]], step_id: int,
    ) -> DistanceMatrix:
        """Convert Habitat sensor observations to Navi DistanceMatrix.

        Delegates to the ``HabitatAdapter`` which handles axis
        transposition ``(El, Az)`` → ``(Az, El)``, depth normalisation,
        semantic remapping, and env-dimension insertion.
        """
        canonical = self._adapter.adapt(obs, step_id)

        return DistanceMatrix(
            episode_id=self._episode_id,
            env_ids=np.array([0], dtype=np.int32),
            matrix_shape=(self._az_bins, self._el_bins),
            depth=canonical["depth"],
            delta_depth=canonical["delta_depth"],
            semantic=canonical["semantic"],
            valid_mask=canonical["valid_mask"],
            overhead=canonical["overhead"],
            robot_pose=self._pose,
            step_id=step_id,
            timestamp=time.time(),
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        previous_pose: RobotPose,
        current_pose: RobotPose,
        obs: dict[str, NDArray[Any]],
    ) -> float:
        """Compute structured reward (same tier system as VoxelBackend)."""
        reward = 0.0

        # 1) Exploration — visit new cells
        cell = (int(np.floor(current_pose.x / 2.0)), int(np.floor(current_pose.z / 2.0)))
        if cell not in self._visited_cells:
            self._visited_cells.add(cell)
            reward += _EXPLORATION_REWARD

        # 2) Progress — forward translation
        dx = current_pose.x - previous_pose.x
        dz = current_pose.z - previous_pose.z
        dy = current_pose.y - previous_pose.y
        progress = float(np.sqrt(dx * dx + dz * dz + dy * dy))
        reward += _PROGRESS_REWARD_SCALE * progress

        # 3) Collision detection via Habitat
        if (
            self._sim is not None
            and hasattr(self._sim, "previous_step_collided")
            and self._sim.previous_step_collided
        ):
            reward += _COLLISION_PENALTY

        # 4) Goal proximity (PointNav)
        if self._goal_position is not None:
            robot_pos = np.array([current_pose.x, current_pose.y, current_pose.z])
            dist_to_goal = float(np.linalg.norm(robot_pos - self._goal_position))
            # Geodesic progress reward (approached goal since last step)
            prev_pos = np.array([previous_pose.x, previous_pose.y, previous_pose.z])
            prev_dist = float(np.linalg.norm(prev_pos - self._goal_position))
            goal_progress = prev_dist - dist_to_goal
            reward += goal_progress * 2.0

        # 5) Anti-circling
        self._yaw_history.append(float(current_pose.yaw))
        self._pos_history.append((float(current_pose.x), float(current_pose.z)))
        if len(self._yaw_history) > _CIRCLING_WINDOW:
            self._yaw_history = self._yaw_history[-_CIRCLING_WINDOW:]
            self._pos_history = self._pos_history[-_CIRCLING_WINDOW:]
        if len(self._yaw_history) >= _CIRCLING_WINDOW:
            yaw_travel = sum(
                abs(self._yaw_history[i] - self._yaw_history[i - 1])
                for i in range(1, len(self._yaw_history))
            )
            pos_travel = sum(
                float(np.sqrt(
                    (self._pos_history[i][0] - self._pos_history[i - 1][0]) ** 2
                    + (self._pos_history[i][1] - self._pos_history[i - 1][1]) ** 2
                ))
                for i in range(1, len(self._pos_history))
            )
            if yaw_travel > 3.0 and pos_travel < 1.0:
                reward += _CIRCLING_PENALTY

        _ = obs
        return reward

    def _check_goal_reached(self) -> bool:
        """Check if the agent has reached the PointNav goal."""
        if self._goal_position is None:
            return False
        robot_pos = np.array([self._pose.x, self._pose.y, self._pose.z])
        dist = float(np.linalg.norm(robot_pos - self._goal_position))
        return dist < _GOAL_RADIUS

    def _extract_goal(self) -> NDArray[np.float32] | None:
        """Extract goal position from dataset episodes or Habitat config."""
        # First: try dataset episodes loaded from JSON
        if self._episodes:
            idx = self._episode_id % len(self._episodes)
            ep = self._episodes[idx]
            goals = ep.get("goals", [])
            if goals and "position" in goals[0]:
                return np.array(goals[0]["position"], dtype=np.float32)

        # Fallback: try Habitat-native episode config
        try:
            episode = self._sim.habitat_config.episode  # type: ignore[attr-defined]
            if hasattr(episode, "goals") and episode.goals:
                goal = episode.goals[0]
                return np.array(goal.position, dtype=np.float32)
        except (AttributeError, IndexError):
            pass
        return None

    @staticmethod
    def _load_episodes(config: SectionManagerConfig) -> list[dict[str, Any]]:
        """Load PointNav episodes from a dataset config JSON file.

        The JSON file should follow the Habitat PointNav dataset format::

            {
              "episodes": [
                {
                  "episode_id": "0",
                  "scene_id": "...",
                  "start_position": [x, y, z],
                  "start_rotation": [qx, qy, qz, qw],
                  "goals": [{"position": [x, y, z], "radius": 0.2}],
                  ...
                },
                ...
              ]
            }

        Returns an empty list if no config path is set or the file
        doesn't exist.
        """
        path_str = config.habitat_dataset_config
        if not path_str:
            return []
        path = Path(path_str)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            episodes: list[dict[str, Any]] = data.get("episodes", [])
            return episodes
        except (json.JSONDecodeError, KeyError, OSError):
            return []
