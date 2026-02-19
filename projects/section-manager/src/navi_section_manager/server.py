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

        Returns:
            ``StepResult`` with step acknowledgement data.
        """
        action = request.action
        now = time.time()

        previous_pose = self._pose
        new_pose = self._apply_action(action, now)
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

        self._step_id += 1
        return StepResult(
            step_id=request.step_id,
            done=False,
            truncated=False,
            reward=0.0,
            episode_return=0.0,
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
