"""ZMQ server for the Environment — PUB + REP sockets."""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

import numpy as np
import zmq

from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    BatchStepRequest,
    BatchStepResult,
    DistanceMatrix,
    StepRequest,
    TelemetryEvent,
    deserialize,
    serialize,
)

if TYPE_CHECKING:
    from navi_environment.backends.base import SimulatorBackend
    from navi_environment.config import EnvironmentConfig

__all__: list[str] = ["EnvironmentServer"]


class EnvironmentServer:
    """Environment ZMQ server.

    Operates in two modes:

    - **step** (training): Listens on a REP socket for ``StepRequest``
      messages.  Each request delegates to the backend's ``step()``,
      replies with ``StepResult``, and publishes the resulting
      ``DistanceMatrix`` on the PUB socket.

    - **async** (inference): Subscribes to ``Action`` on a SUB socket,
      applies them continuously, and publishes ``DistanceMatrix``
      on the PUB socket.

    The simulation logic is fully encapsulated in a
    :class:`SimulatorBackend` implementation (voxel, habitat, etc.).
    """

    def __init__(
        self,
        config: EnvironmentConfig,
        backend: SimulatorBackend,
    ) -> None:
        # ── Training-mode invariant ──────────────────────────────────
        # When training_mode is True, enforce the unthrottled REQ/REP
        # lock-step loop.  The simulation must never be artificially
        # delayed; wall-clock time is irrelevant during training.
        if config.training_mode and config.mode != "step":
            msg = (
                "training_mode=True requires mode='step' (REQ/REP). "
                f"Got mode='{config.mode}'.  Async mode is for "
                "inference/dashboard only."
            )
            raise ValueError(msg)

        self._config = config
        self._backend = backend
        self._step_id = 0
        self._last_obs: DistanceMatrix | None = None
        self._last_obs_bytes: bytes | None = None  # cached serialized DM
        self._initial_obs_bytes: list[bytes] = []   # per-actor initial obs
        self._running = False

        # ZMQ
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._pub_socket: zmq.Socket[bytes] | None = None
        self._rep_socket: zmq.Socket[bytes] | None = None
        self._action_sub: zmq.Socket[bytes] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind ZMQ sockets and publish initial observations for all actors."""
        cfg = self._config

        self._running = True
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(cfg.pub_address)

        if cfg.mode == "step":
            self._rep_socket = self._context.socket(zmq.REP)
            self._rep_socket.bind(cfg.rep_address)
        else:
            self._action_sub = self._context.socket(zmq.SUB)
            self._action_sub.connect(cfg.action_sub_address)
            self._action_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_ACTION.encode("utf-8"))

        # Reset and publish initial observation for each actor
        self._initial_obs_bytes = []
        for actor_id in range(cfg.n_actors):
            obs = self._backend.reset(episode_id=0, actor_id=actor_id)
            self._last_obs = obs
            self._publish(obs)
            self._initial_obs_bytes.append(self._last_obs_bytes)  # type: ignore[arg-type]
            pose = self._backend.actor_pose(actor_id) if hasattr(self._backend, "actor_pose") else self._backend.pose
            self._publish_telemetry(
                event_type="environment.startup",
                step_id=0,
                payload=np.array([
                    pose.x, pose.y, pose.z, pose.yaw,
                ], dtype=np.float32),
                actor_id=actor_id,
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
        """Signal the loop to stop, close sockets and backend.

        Idempotent — safe to call more than once or from any thread.
        """
        self._running = False
        if self._pub_socket is None and self._rep_socket is None and self._action_sub is None:
            return  # already stopped
        self._backend.close()
        for sock in (self._pub_socket, self._rep_socket, self._action_sub):
            if sock is not None:
                with contextlib.suppress(Exception):
                    sock.close()
        self._pub_socket = None
        self._rep_socket = None
        self._action_sub = None
        with contextlib.suppress(Exception):
            self._context.term()

    # ------------------------------------------------------------------
    # Step mode (REQ/REP)
    # ------------------------------------------------------------------

    def _run_step_loop(self) -> None:
        """REP loop: receive StepRequest -> delegate to backend -> reply.

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
        dm_topic = TOPIC_DISTANCE_MATRIX.encode("utf-8")
        while self._running:
            try:
                events = dict(poller.poll(timeout=idle_interval_ms))
            except zmq.ZMQError:
                break  # socket closed by stop()
            if self._rep_socket not in events:
                # Re-publish all initial actor observations for late
                # subscribers.  Only fires before the first step request.
                if (
                    self._step_id == 0
                    and self._initial_obs_bytes
                    and self._pub_socket is not None
                ):
                    for obs_bytes in self._initial_obs_bytes:
                        with contextlib.suppress(zmq.Again):
                            self._pub_socket.send_multipart(
                                [dm_topic, obs_bytes],
                                flags=zmq.NOBLOCK,
                            )
                continue

            try:
                data = self._rep_socket.recv()
                msg = deserialize(data)
                if isinstance(msg, BatchStepRequest):
                    observations, results = self._backend.batch_step(
                        msg.actions, msg.step_id,
                    )
                    batch_result = BatchStepResult(
                        results=results,
                        observations=observations,
                    )
                    self._rep_socket.send(serialize(batch_result))
                    # Publish each actor's observation + telemetry
                    for obs, action in zip(observations, msg.actions, strict=True):
                        self._publish(obs)
                        actor_id = (
                            int(action.env_ids[0])
                            if len(action.env_ids) > 0
                            else 0
                        )
                        pose = (
                            self._backend.actor_pose(actor_id)
                            if hasattr(self._backend, "actor_pose")
                            else self._backend.pose
                        )
                        self._publish_telemetry(
                            event_type="environment.step",
                            step_id=msg.step_id,
                            payload=np.array([
                                pose.x, pose.y, pose.z, pose.yaw,
                            ], dtype=np.float32),
                            actor_id=actor_id,
                        )
                    self._step_id = msg.step_id
                elif isinstance(msg, StepRequest):
                    # Extract actor_id from action env_ids (default 0)
                    actor_id = int(msg.action.env_ids[0]) if len(msg.action.env_ids) > 0 else 0
                    obs, result = self._backend.step(
                        msg.action, msg.step_id, actor_id=actor_id,
                    )
                    self._last_obs = obs
                    self._rep_socket.send(serialize(result))
                    self._publish(obs)
                    pose = (
                        self._backend.actor_pose(actor_id)
                        if hasattr(self._backend, "actor_pose")
                        else self._backend.pose
                    )
                    self._publish_telemetry(
                        event_type="environment.step",
                        step_id=msg.step_id,
                        payload=np.array([
                            pose.x, pose.y, pose.z, pose.yaw,
                        ], dtype=np.float32),
                        actor_id=actor_id,
                    )
                    self._step_id = msg.step_id
            except Exception as e:
                import traceback
                print(f"CRITICAL ERROR in Environment Step Loop: {e}", flush=True)
                traceback.print_exc()
                # To prevent total deadlock, we must reply or close. But REQ/REP is strictly ping-pong.
                # If we crashed before sending a reply, the Actor is permanently stuck.
                # We will just break to let the thread die and the system reboot if handled.
                break

    # ------------------------------------------------------------------
    # Async mode (SUB Action → PUB DistanceMatrix)
    # ------------------------------------------------------------------

    def _run_async_loop(self) -> None:
        """SUB loop: receive Action, delegate to backend, publish."""
        assert self._action_sub is not None
        poller = zmq.Poller()
        poller.register(self._action_sub, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(timeout=100))
            except zmq.ZMQError:
                break  # socket closed by stop()
            if self._action_sub not in events:
                continue

            frames = self._action_sub.recv_multipart()
            if len(frames) != 2:
                continue

            _topic, data = frames
            try:
                msg = deserialize(data)
            except Exception:  # noqa: BLE001, RUF100
                continue

            if not isinstance(msg, Action):
                continue

            obs, _result = self._backend.step(msg, self._step_id)
            self._publish(obs)
            self._publish_telemetry(
                event_type="environment.async_step",
                step_id=self._step_id,
                payload=np.array([
                    self._backend.pose.x,
                    self._backend.pose.y,
                    self._backend.pose.z,
                    self._backend.pose.yaw,
                ], dtype=np.float32),
            )
            self._step_id += 1

    # ------------------------------------------------------------------
    # Publishing helpers
    # ------------------------------------------------------------------

    def _publish(self, obs: DistanceMatrix) -> None:
        """Publish a DistanceMatrix on the PUB socket (non-blocking)."""
        if self._pub_socket is None:
            return
        obs_bytes = serialize(obs)
        self._last_obs_bytes = obs_bytes
        with contextlib.suppress(zmq.Again):
            self._pub_socket.send_multipart(
                [TOPIC_DISTANCE_MATRIX.encode("utf-8"), obs_bytes],
                flags=zmq.NOBLOCK,
            )

    def _publish_telemetry(
        self,
        event_type: str,
        step_id: int,
        payload: np.ndarray,
        *,
        actor_id: int = 0,
    ) -> None:
        """Publish a TelemetryEvent on the PUB socket (non-blocking)."""
        if self._pub_socket is None:
            return
        episode_id = (
            self._backend.actor_episode_id(actor_id)
            if hasattr(self._backend, "actor_episode_id")
            else self._backend.episode_id
        )
        event = TelemetryEvent(
            event_type=event_type,
            episode_id=episode_id,
            env_id=actor_id,
            step_id=step_id,
            payload=payload.astype(np.float32, copy=False),
            timestamp=time.time(),
        )
        with contextlib.suppress(zmq.Again):
            self._pub_socket.send_multipart(
                [
                    TOPIC_TELEMETRY_EVENT.encode("utf-8"),
                    serialize(event),
                ],
                flags=zmq.NOBLOCK,
            )

    @property
    def pose(self) -> DistanceMatrix | None:
        """Current robot pose (delegates to backend)."""
        return None

