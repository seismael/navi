"""Multi-stream asynchronous ZMQ ingestion engine for the RL dashboard.

Subscribes to ``distance_matrix_v2``, ``action_v2``, and
``telemetry_event_v2`` on two independent sockets (Section-Manager PUB
and Actor PUB), drains all queues without blocking, and maintains typed
ring-buffer state for the rendering layer.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import zmq

from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    DistanceMatrix,
    StepRequest,
    TelemetryEvent,
    deserialize,
    serialize,
)

__all__: list[str] = ["StreamEngine", "StreamState"]

_RING_LEN = 2000


@dataclass
class StreamState:
    """Snapshot of latest data across all ZMQ streams."""

    latest_matrix: DistanceMatrix | None = None
    latest_action: Action | None = None
    last_rx_time: float = 0.0
    pose_history: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=90),
    )

    # Training step telemetry ring buffers
    reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    advantage_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    collision_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    novelty_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    forward_cmd_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    yaw_cmd_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    grad_norm_fwd_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    grad_norm_yaw_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # Evaluation-window telemetry ring buffers
    eval_reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    eval_collision_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    eval_novelty_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    eval_coverage_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    eval_step_history: deque[int] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # Inference features
    latest_features: np.ndarray | None = None

    # Telemetry ring (raw events)
    telemetry_buffer: deque[TelemetryEvent] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )


class StreamEngine:
    """Non-blocking multi-socket ZMQ subscriber with per-topic routing.

    Subscribes to:
    * ``matrix_sub`` — Section Manager PUB (``distance_matrix_v2``,
      ``telemetry_event_v2``)
    * ``actor_sub``  — Actor/Trainer PUB (``action_v2``,
      ``telemetry_event_v2``)

    Optionally connects a REQ socket for manual teleop stepping.
    """

    def __init__(
        self,
        matrix_sub: str,
        actor_sub: str = "",
        step_endpoint: str = "",
    ) -> None:
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._poller = zmq.Poller()
        self.state = StreamState()

        # Section-Manager PUB socket
        self._sock_matrix = self._ctx.socket(zmq.SUB)
        self._sock_matrix.connect(matrix_sub)
        for topic in (TOPIC_DISTANCE_MATRIX, TOPIC_TELEMETRY_EVENT):
            self._sock_matrix.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        self._poller.register(self._sock_matrix, zmq.POLLIN)

        # Actor / Trainer PUB socket
        self._sock_actor: zmq.Socket[bytes] | None = None
        if actor_sub:
            self._sock_actor = self._ctx.socket(zmq.SUB)
            self._sock_actor.connect(actor_sub)
            for topic in (TOPIC_ACTION, TOPIC_TELEMETRY_EVENT):
                self._sock_actor.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            self._poller.register(self._sock_actor, zmq.POLLIN)

        # Manual teleop REQ socket
        self._sock_step: zmq.Socket[bytes] | None = None
        if step_endpoint:
            self._sock_step = self._ctx.socket(zmq.REQ)
            self._sock_step.connect(step_endpoint)

        self._step_counter = 0

    # ── public API ───────────────────────────────────────────────────

    def poll(self) -> None:
        """Non-blocking drain of all ZMQ queues, updating state buffers."""
        while True:
            socks = dict(self._poller.poll(0))
            if not socks:
                break

            if self._sock_matrix in socks:
                self._recv_from(self._sock_matrix)
            if self._sock_actor is not None and self._sock_actor in socks:
                self._recv_from(self._sock_actor)

    def send_step_request(
        self,
        linear_velocity: float,
        yaw_rate: float,
    ) -> None:
        """Send manual teleop StepRequest via REQ socket."""
        if self._sock_step is None:
            return
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array(
                [[linear_velocity, 0.0, 0.0]], dtype=np.float32,
            ),
            angular_velocity=np.array(
                [[0.0, 0.0, yaw_rate]], dtype=np.float32,
            ),
            policy_id="dashboard-manual",
            step_id=self._step_counter,
            timestamp=time.time(),
        )
        request = StepRequest(
            action=action,
            step_id=self._step_counter,
            timestamp=time.time(),
        )
        self._step_counter += 1
        self._sock_step.send(serialize(request))
        # Block until Section Manager replies (< 5 ms typical)
        _reply = self._sock_step.recv()

    @property
    def has_step_socket(self) -> bool:
        """Whether manual stepping is available."""
        return self._sock_step is not None

    def close(self) -> None:
        """Tear down all sockets."""
        self._sock_matrix.close()
        if self._sock_actor is not None:
            self._sock_actor.close()
        if self._sock_step is not None:
            self._sock_step.close()
        self._ctx.term()

    # ── internal routing ─────────────────────────────────────────────

    def _recv_from(self, sock: zmq.Socket[bytes]) -> None:
        """Drain one socket, dispatching by topic."""
        while True:
            try:
                topic_bytes, data = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            topic = topic_bytes.decode("utf-8")
            msg = deserialize(data)
            self._dispatch(topic, msg)

    def _dispatch(self, topic: str, msg: object) -> None:
        """Route a deserialized message to the appropriate state buffer."""
        if topic == TOPIC_DISTANCE_MATRIX and isinstance(msg, DistanceMatrix):
            self.state.latest_matrix = msg
            self.state.last_rx_time = time.time()
            pose = msg.robot_pose
            self.state.pose_history.append((pose.x, pose.z, pose.yaw))

        elif topic == TOPIC_ACTION and isinstance(msg, Action):
            self.state.latest_action = msg

        elif topic == TOPIC_TELEMETRY_EVENT and isinstance(msg, TelemetryEvent):
            self.state.telemetry_buffer.append(msg)
            self._route_telemetry(msg)

    def _route_telemetry(self, event: TelemetryEvent) -> None:
        """Parse telemetry event type and push into typed ring buffers."""
        et = event.event_type
        p = event.payload

        if et == "actor.training.step" and len(p) >= 8:
            self.state.reward_history.append(float(p[0]))
            self.state.advantage_history.append(float(p[1]))
            self.state.collision_history.append(float(p[2]))
            self.state.novelty_history.append(float(p[3]))
            self.state.forward_cmd_history.append(float(p[4]))
            self.state.yaw_cmd_history.append(float(p[5]))
            self.state.grad_norm_fwd_history.append(float(p[6]))
            self.state.grad_norm_yaw_history.append(float(p[7]))

        elif et == "actor.training.eval" and len(p) >= 4:
            self.state.eval_step_history.append(event.step_id)
            self.state.eval_reward_history.append(float(p[0]))
            self.state.eval_collision_history.append(float(p[1]))
            self.state.eval_novelty_history.append(float(p[2]))
            self.state.eval_coverage_history.append(float(p[3]))

        elif et == "actor.inference.features":
            self.state.latest_features = np.asarray(p, dtype=np.float32)
