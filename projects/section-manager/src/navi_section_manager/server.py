"""ZMQ server for the Section Manager — PUB + REP sockets."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import zmq

from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    DistanceMatrix,
    RobotPose,
    StepRequest,
    StepResult,
    TelemetryEvent,
    deserialize,
    serialize,
)
from navi_section_manager.distance_matrix_v2 import DistanceMatrixBuilder
from navi_section_manager.frustum import FrustumLoader
from navi_section_manager.lookahead import LookAheadBuffer
from navi_section_manager.matrix import SparseVoxelGrid
from navi_section_manager.mjx_env import MjxEnvironment
from navi_section_manager.sliding_window import SlidingWindow

if TYPE_CHECKING:
    from navi_section_manager.config import SectionManagerConfig
    from navi_section_manager.generators.base import AbstractWorldGenerator

__all__: list[str] = ["SectionManagerServer"]

# Semantic IDs used for reward computation
_SEM_TARGET: int = 10
_TARGET_DISCOVERY_RADIUS: float = 3.0
_TARGET_MAX_REWARD: float = 5.0
_EXPLORATION_REWARD: float = 0.3
_COLLISION_PENALTY: float = -2.0
_CIRCLING_PENALTY: float = -0.5
_PROGRESS_REWARD_SCALE: float = 0.8
_CIRCLING_WINDOW: int = 20


class SectionManagerServer:
    """Section Manager ZMQ server.

    Operates in two modes:

    - **step** (training): Listens on a REP socket for ``StepRequest``
      messages.  Each request applies an ``Action``, slides the voxel
      window, and replies with ``StepResult``.  Publishes the resulting
            ``DistanceMatrix`` on the PUB socket.

    - **async** (inference): Subscribes to ``Action`` on a SUB socket,
            applies them continuously, and publishes ``DistanceMatrix``
      on the PUB socket.
    """

    def __init__(
        self,
        config: SectionManagerConfig,
        generator: AbstractWorldGenerator,
    ) -> None:
        self._config = config
        self._generator = generator

        # Core data structures
        self._grid = SparseVoxelGrid(chunk_size=config.chunk_size)
        self._window = SlidingWindow(self._grid, radius=config.window_radius)
        self._frustum = FrustumLoader(
            chunk_size=config.chunk_size,
            half_angle_deg=45.0,
            near=1,
            far=config.lookahead_margin,
        )
        self._lookahead = LookAheadBuffer(capacity=128)

        # Robot state
        spawn = generator.spawn_position()
        self._pose = RobotPose(
            x=spawn[0],
            y=spawn[1],
            z=spawn[2],
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
            timestamp=time.time(),
        )
        self._step_id = 0
        self._mjx_env = MjxEnvironment(dt=1.0)
        self._distance_builder = DistanceMatrixBuilder(
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_distance=config.max_distance,
        )

        # Episode tracking for reward computation
        self._episode_return: float = 0.0
        self._visited_cells: set[tuple[int, int]] = set()
        self._discovered_targets: set[tuple[int, int, int]] = set()
        self._yaw_history: list[float] = []
        self._pos_history: list[tuple[float, float]] = []

        # ZMQ
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._pub_socket: zmq.Socket[bytes] | None = None
        self._rep_socket: zmq.Socket[bytes] | None = None
        self._action_sub: zmq.Socket[bytes] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind ZMQ sockets and initialise the window around the spawn."""
        cfg = self._config

        # PUB socket — publishes DistanceMatrix
        # PUB socket — publishes DistanceMatrix
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(cfg.pub_address)

        if cfg.mode == "step":
            self._rep_socket = self._context.socket(zmq.REP)
            self._rep_socket.bind(cfg.rep_address)
        else:
            # Async mode: subscribe to Action topic
            self._action_sub = self._context.socket(zmq.SUB)
            self._action_sub.connect(cfg.action_sub_address)
            self._action_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_ACTION.encode("utf-8"))

        # Seed the window — generate all chunks around the spawn
        initial_wedge = self._window.shift(
            self._pose.x,
            self._pose.y,
            self._pose.z,
            self._generator.generate_chunk,
        )

        # Publish initial world snapshot so downstream services render immediately.
        # This avoids starting with empty spherical/environment views before movement.
        _ = initial_wedge
        initial_obs = self._build_distance_matrix(step_id=0, timestamp=time.time())
        self._publish(initial_obs)
        self._publish_telemetry(
            event_type="section_manager.startup",
            step_id=0,
            payload=np.array([self._pose.x, self._pose.y, self._pose.z, self._pose.yaw], dtype=np.float32),
        )

    def run(self) -> None:
        """Run the main loop (blocks until interrupted)."""
        self.start()
        try:
            if self._config.mode == "step":
                self._run_step_loop()
            else:
                self._run_async_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Shut down sockets."""
        for sock in (self._pub_socket, self._rep_socket, self._action_sub):
            if sock is not None:
                sock.close()
        self._context.term()

    # ------------------------------------------------------------------
    # Step mode (REQ/REP)
    # ------------------------------------------------------------------

    def _run_step_loop(self) -> None:
        """REP loop: receive StepRequest -> apply -> reply StepResult.

        Uses a poller with a 500 ms timeout so the current observation
        is re-published periodically while no ``StepRequest`` has
        arrived.  This prevents a startup deadlock: ZMQ PUB/SUB does
        not replay old messages, so late-connecting subscribers (Actor,
        Auditor) would miss the single initial publish and block
        forever.
        """
        assert self._rep_socket is not None
        poller = zmq.Poller()
        poller.register(self._rep_socket, zmq.POLLIN)

        idle_interval_ms = 500
        while True:
            events = dict(poller.poll(timeout=idle_interval_ms))
            if self._rep_socket not in events:
                # No StepRequest yet — re-publish current observation
                # so late subscribers can bootstrap.
                obs = self._build_distance_matrix(
                    step_id=self._step_id,
                    timestamp=time.time(),
                )
                self._publish(obs)
                continue

            _topic_or_data = self._rep_socket.recv()
            msg = deserialize(_topic_or_data)
            if isinstance(msg, StepRequest):
                result = self.step(msg)
                self._rep_socket.send(serialize(result))

    def step(self, request: StepRequest) -> StepResult:
        """Execute a single step: apply action, slide window, publish observation.

        Computes structured rewards:
        - **Target discovery** (max): large positive for approaching target voxels
        - **Exploration** (medium): positive for visiting new spatial cells
        - **Progress** (medium): positive for forward translation
        - **Collision** (negative): penalty when motion is blocked
        - **Circling** (negative): penalty when yaw spins with little translation

        Returns:
            ``StepResult`` with step acknowledgement data and computed reward.
        """
        action = request.action
        now = time.time()

        previous_pose = self._pose
        new_pose = self._apply_action(action, now)

        # Detect collision: proposed vs actual motion
        proposed_motion = np.sqrt(
            (new_pose.x - previous_pose.x) ** 2
            + (new_pose.z - previous_pose.z) ** 2
        )
        linear_cmd = float(action.linear_velocity[0, 0]) if action.linear_velocity.ndim == 2 else float(action.linear_velocity[0])
        expected_motion = max(1e-3, abs(linear_cmd))
        collided = linear_cmd > 0.15 and proposed_motion / expected_motion < 0.15

        self._pose = new_pose

        wedge = self._window.shift(
            new_pose.x,
            new_pose.y,
            new_pose.z,
            self._get_chunk,
        )
        wedge_voxels = wedge.new_voxels
        if wedge_voxels.shape[0] == 0:
            wedge_voxels = self._local_snapshot(radius_chunks=1)

        # Predictive prefetching
        velocity = np.array(
            [
                new_pose.x - previous_pose.x,
                new_pose.y - previous_pose.y,
                new_pose.z - previous_pose.z,
            ],
            dtype=np.float32,
        )
        frustum_coords = self._frustum.compute_frustum(
            self._window.center_chunk,
            velocity,
        )
        self._lookahead.prefetch(frustum_coords, self._generator.generate_chunk)

        _ = wedge_voxels
        _ = wedge.culled_count
        obs = self._build_distance_matrix(step_id=request.step_id, timestamp=now)
        self._publish(obs)
        self._publish_telemetry(
            event_type="section_manager.step",
            step_id=request.step_id,
            payload=np.array([new_pose.x, new_pose.y, new_pose.z, new_pose.yaw], dtype=np.float32),
        )

        # ── Reward computation ───────────────────────────────────────
        reward = 0.0

        # 1) Target discovery — highest reward tier
        target_reward = self._compute_target_reward(new_pose)
        reward += target_reward

        # 2) Exploration — medium reward for visiting new cells
        cell = (int(np.floor(new_pose.x / 2.0)), int(np.floor(new_pose.z / 2.0)))
        is_novel = cell not in self._visited_cells
        if is_novel:
            self._visited_cells.add(cell)
            reward += _EXPLORATION_REWARD

        # 3) Progress — reward forward translation
        dx = new_pose.x - previous_pose.x
        dz = new_pose.z - previous_pose.z
        dy = new_pose.y - previous_pose.y
        progress = float(np.sqrt(dx * dx + dz * dz + dy * dy))
        reward += _PROGRESS_REWARD_SCALE * progress

        # 4) Collision penalty
        if collided:
            reward += _COLLISION_PENALTY

        # 5) Anti-circling — penalise spinning in place
        self._yaw_history.append(float(new_pose.yaw))
        self._pos_history.append((float(new_pose.x), float(new_pose.z)))
        if len(self._yaw_history) > _CIRCLING_WINDOW:
            self._yaw_history = self._yaw_history[-_CIRCLING_WINDOW:]
            self._pos_history = self._pos_history[-_CIRCLING_WINDOW:]
        if len(self._yaw_history) >= _CIRCLING_WINDOW:
            yaw_travel = sum(
                abs(self._yaw_history[i] - self._yaw_history[i - 1])
                for i in range(1, len(self._yaw_history))
            )
            pos_travel = sum(
                np.sqrt(
                    (self._pos_history[i][0] - self._pos_history[i - 1][0]) ** 2
                    + (self._pos_history[i][1] - self._pos_history[i - 1][1]) ** 2
                )
                for i in range(1, len(self._pos_history))
            )
            if yaw_travel > 3.0 and pos_travel < 1.0:
                reward += _CIRCLING_PENALTY

        self._episode_return += reward
        self._step_id += 1

        return StepResult(
            step_id=request.step_id,
            done=False,
            truncated=False,
            reward=reward,
            episode_return=self._episode_return,
            timestamp=now,
        )

    # ------------------------------------------------------------------
    # Async mode (SUB Action → PUB DistanceMatrix)
    # ------------------------------------------------------------------

    def _run_async_loop(self) -> None:
        """SUB loop: receive Action, slide window, publish observations."""
        assert self._action_sub is not None
        poller = zmq.Poller()
        poller.register(self._action_sub, zmq.POLLIN)
        while True:
            events = dict(poller.poll(timeout=100))
            if self._action_sub not in events:
                continue

            frames = self._action_sub.recv_multipart()
            if len(frames) != 2:
                continue

            _topic, data = frames
            try:
                msg = deserialize(data)
            except Exception:
                continue

            if not isinstance(msg, Action):
                continue

            now = time.time()
            previous_pose = self._pose
            new_pose = self._apply_action(msg, now)
            self._pose = new_pose

            wedge = self._window.shift(
                new_pose.x,
                new_pose.y,
                new_pose.z,
                self._get_chunk,
            )
            wedge_voxels = wedge.new_voxels
            if wedge_voxels.shape[0] == 0:
                wedge_voxels = self._local_snapshot(radius_chunks=1)

            _ = previous_pose
            _ = wedge_voxels
            _ = wedge.culled_count
            obs = self._build_distance_matrix(step_id=self._step_id, timestamp=now)
            self._publish(obs)
            self._publish_telemetry(
                event_type="section_manager.async_step",
                step_id=self._step_id,
                payload=np.array([new_pose.x, new_pose.y, new_pose.z, new_pose.yaw], dtype=np.float32),
            )
            self._step_id += 1

    # ------------------------------------------------------------------
    # Reward helpers
    # ------------------------------------------------------------------

    def _compute_target_reward(self, pose: RobotPose) -> float:
        """Check proximity to target voxels (semantic ID = 10) and reward discovery."""
        voxels = self._local_snapshot(radius_chunks=1)
        if voxels.shape[0] == 0:
            return 0.0

        # Filter for target voxels
        sem = np.rint(voxels[:, 4]).astype(np.int32)
        target_mask = sem == _SEM_TARGET
        if not np.any(target_mask):
            return 0.0

        target_voxels = voxels[target_mask, :3]
        robot_pos = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
        distances = np.linalg.norm(target_voxels - robot_pos[None, :], axis=1)
        min_dist = float(np.min(distances))

        if min_dist > _TARGET_DISCOVERY_RADIUS:
            return 0.0

        # Check if closest target was already discovered
        closest_idx = int(np.argmin(distances))
        target_cell = (
            int(np.floor(float(target_voxels[closest_idx, 0]))),
            int(np.floor(float(target_voxels[closest_idx, 1]))),
            int(np.floor(float(target_voxels[closest_idx, 2]))),
        )

        if target_cell in self._discovered_targets:
            # Already discovered — small proximity reward
            return 0.1

        self._discovered_targets.add(target_cell)
        # Scale reward: closer = higher (max at distance 0)
        proximity_factor = 1.0 - (min_dist / _TARGET_DISCOVERY_RADIUS)
        return _TARGET_MAX_REWARD * proximity_factor

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action, timestamp: float) -> RobotPose:
        """Apply a velocity-based action to the current pose."""
        stepped_pose = self._mjx_env.step_pose(self._pose, action, timestamp)
        proposed_x = stepped_pose.x
        proposed_y = stepped_pose.y
        proposed_z = stepped_pose.z

        constrained_x, constrained_y, constrained_z = self._constrain_translation(
            proposed_x,
            proposed_y,
            proposed_z,
            self._config.barrier_distance,
            self._config.collision_probe_radius,
        )

        return RobotPose(
            x=constrained_x,
            y=constrained_y,
            z=constrained_z,
            roll=stepped_pose.roll,
            pitch=stepped_pose.pitch,
            yaw=stepped_pose.yaw,
            timestamp=timestamp,
        )

    def _constrain_translation(
        self,
        x: float,
        y: float,
        z: float,
        barrier_distance: float,
        probe_radius: float,
    ) -> tuple[float, float, float]:
        """Clamp motion to keep a minimum standoff from occupied voxels."""
        if barrier_distance <= 0.0:
            return x, y, z
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
            fallback = np.array(
                [
                    self._pose.x - float(nearest[0]),
                    self._pose.y - float(nearest[1]),
                    self._pose.z - float(nearest[2]),
                ],
                dtype=np.float32,
            )
            fallback_norm = float(np.linalg.norm(fallback))
            if fallback_norm < 1e-6:
                return self._pose.x, self._pose.y, self._pose.z
            direction = fallback / fallback_norm
        else:
            direction = direction / norm

        constrained = nearest + direction * barrier_distance
        current = np.array([self._pose.x, self._pose.y, self._pose.z], dtype=np.float32)
        proposed = np.array([x, y, z], dtype=np.float32)
        if np.linalg.norm(constrained - current) > np.linalg.norm(proposed - current):
            return self._pose.x, self._pose.y, self._pose.z
        return float(constrained[0]), float(constrained[1]), float(constrained[2])

    def _get_chunk(self, cx: int, cy: int, cz: int) -> np.ndarray:
        """Get a chunk from the lookahead buffer or generate on demand."""
        cached = self._lookahead.get((cx, cy, cz))
        if cached is not None:
            return cached
        return self._generator.generate_chunk(cx, cy, cz)

    def _publish(self, obs: DistanceMatrix) -> None:
        """Publish a DistanceMatrix on the PUB socket."""
        if self._pub_socket is None:
            return
        payload = serialize(obs)
        self._pub_socket.send_multipart(
            [
                TOPIC_DISTANCE_MATRIX.encode("utf-8"),
                payload,
            ]
        )

    def _publish_telemetry(self, event_type: str, step_id: int, payload: np.ndarray) -> None:
        """Publish a TelemetryEvent on the PUB socket."""
        if self._pub_socket is None:
            return

        event = TelemetryEvent(
            event_type=event_type,
            episode_id=0,
            env_id=0,
            step_id=step_id,
            payload=payload.astype(np.float32, copy=False),
            timestamp=time.time(),
        )
        self._pub_socket.send_multipart(
            [
                TOPIC_TELEMETRY_EVENT.encode("utf-8"),
                serialize(event),
            ]
        )

    def _build_distance_matrix(self, step_id: int, timestamp: float) -> DistanceMatrix:
        """Build a DistanceMatrix v2 observation from local voxels around the robot."""
        voxels = self._local_snapshot(radius_chunks=2)
        return self._distance_builder.build(
            voxels=voxels,
            pose=self._pose,
            step_id=step_id,
            timestamp=timestamp,
            episode_id=0,
            env_id=0,
        )

    def _local_snapshot(self, radius_chunks: int) -> np.ndarray:
        """Query a local voxel snapshot around the current center chunk."""
        cx, cy, cz = self._window.center_chunk
        return self._grid.query_region(
            (cx - radius_chunks, cy - radius_chunks, cz - radius_chunks),
            (cx + radius_chunks, cy + radius_chunks, cz + radius_chunks),
        )

    @property
    def pose(self) -> RobotPose:
        """Current robot pose."""
        return self._pose
