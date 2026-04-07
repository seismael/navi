"""Multi-stream asynchronous ZMQ ingestion engine for the RL dashboard.

Subscribes to ``distance_matrix_v2``, ``action_v2``, and
``telemetry_event_v2`` on two independent sockets (Environment PUB
and Actor PUB), drains all queues without blocking, and maintains typed
ring-buffer state for the rendering layer.
"""

from __future__ import annotations

import logging
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
_LOG = logging.getLogger(__name__)


@dataclass
class StreamState:
    """Snapshot of latest data across all ZMQ streams."""

    latest_matrix: DistanceMatrix | None = None
    latest_action: Action | None = None
    last_rx_time: float = 0.0
    pose_history: deque[tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=90),
    )

    # Canonical training/inference ring buffers
    reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    collision_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    forward_cmd_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    yaw_cmd_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # Inference features
    latest_features: np.ndarray | None = None

    # Universal metric histories (available in all policy modes)
    front_depth_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    mean_depth_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    near_fraction_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # PPO per-step telemetry
    ppo_raw_reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_shaped_reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_intrinsic_reward_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_loop_similarity_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_loop_detected_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_beta_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_done_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # PPO per-update telemetry
    ppo_policy_loss_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_value_loss_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_entropy_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_kl_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_rnd_loss_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_clip_fraction_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_total_loss_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_reward_ema_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_lr_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    ppo_rnd_lr_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # Performance instrumentation ring buffers
    perf_sps_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_forward_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_batch_step_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_memory_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_transition_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_tick_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_zero_wait_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    perf_opt_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    env_perf_sps_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    env_perf_batch_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )
    env_perf_actor_ms_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )

    # Current scene name (updated by actor.scene events)
    current_scene_name: str = ""

    # Telemetry ring (raw events)
    telemetry_buffer: deque[TelemetryEvent] = field(
        default_factory=lambda: deque(maxlen=_RING_LEN),
    )


class StreamEngine:
    """Non-blocking multi-socket ZMQ subscriber with per-topic routing.

    Subscribes to:
    * ``matrix_sub`` — Environment PUB (``distance_matrix_v2``,
      ``telemetry_event_v2``)
    * ``actor_sub``  — Actor/Trainer PUB (``action_v2``,
      ``telemetry_event_v2``)

    Optionally connects a REQ socket for manual teleop stepping.
    """

    def __init__(
        self,
        matrix_sub: str = "",
        actor_sub: str = "",
        step_endpoint: str = "",
        n_actors: int = 0,
        selected_actor_id: int | None = 0,
    ) -> None:
        self._ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._poller = zmq.Poller()
        self._actor_states: dict[int, StreamState] = {}
        self._fleet_n_actors: int = 0
        self._selected_actor_id: int | None = selected_actor_id

        self._msg_total: int = 0
        self._drop_total: int = 0
        self._topic_counts: dict[str, int] = {
            TOPIC_DISTANCE_MATRIX: 0,
            TOPIC_ACTION: 0,
            TOPIC_TELEMETRY_EVENT: 0,
        }
        self._last_diag_t: float = time.monotonic()
        self._observation_updated: bool = False

        # Pre-populate actor states if requested
        if n_actors > 0:
            for i in range(n_actors):
                self._actor_states[i] = StreamState()

        # Environment PUB socket
        self._sock_matrix: zmq.Socket[bytes] | None = None
        if matrix_sub:
            self._sock_matrix = self._ctx.socket(zmq.SUB)
            self._sock_matrix.setsockopt(zmq.RCVHWM, 200)
            self._sock_matrix.setsockopt(zmq.LINGER, 0)
            self._sock_matrix.connect(matrix_sub)
            for topic in (TOPIC_DISTANCE_MATRIX, TOPIC_TELEMETRY_EVENT):
                self._sock_matrix.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            self._poller.register(self._sock_matrix, zmq.POLLIN)

        # Actor / Trainer PUB socket — telemetry + actions only (no observations)
        self._sock_actor: zmq.Socket[bytes] | None = None
        if actor_sub:
            self._sock_actor = self._ctx.socket(zmq.SUB)
            self._sock_actor.setsockopt(zmq.RCVHWM, 50)
            self._sock_actor.setsockopt(zmq.LINGER, 0)
            self._sock_actor.connect(actor_sub)
            for topic in (TOPIC_ACTION, TOPIC_TELEMETRY_EVENT):
                self._sock_actor.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
            self._poller.register(self._sock_actor, zmq.POLLIN)

        # Dedicated low-HWM observation socket — drain-to-latest pattern
        # prevents stale buffered observations from replaying after any
        # transient UI or GC pause.  We do NOT use zmq.CONFLATE because it
        # is incompatible with multi-part messages (corrupts [topic, payload]
        # framing, causing libzmq !_more assertion crash).
        self._sock_obs: zmq.Socket[bytes] | None = None
        if actor_sub:
            self._sock_obs = self._ctx.socket(zmq.SUB)
            self._sock_obs.setsockopt(zmq.RCVHWM, 4)
            self._sock_obs.setsockopt(zmq.LINGER, 0)
            self._sock_obs.connect(actor_sub)
            self._sock_obs.setsockopt(
                zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"),
            )
            self._poller.register(self._sock_obs, zmq.POLLIN)

        # Manual teleop REQ socket
        self._sock_step: zmq.Socket[bytes] | None = None
        if step_endpoint:
            self._sock_step = self._ctx.socket(zmq.REQ)
            # Set short timeouts to prevent UI freezes
            self._sock_step.setsockopt(zmq.RCVTIMEO, 500)
            self._sock_step.setsockopt(zmq.SNDTIMEO, 500)
            # REQ_RELAXED + REQ_CORRELATE allow the socket to recover if a request is lost
            self._sock_step.setsockopt(zmq.REQ_RELAXED, 1)
            self._sock_step.setsockopt(zmq.REQ_CORRELATE, 1)
            self._sock_step.connect(step_endpoint)

        self._step_counter = 0

    # ── public API ───────────────────────────────────────────────────

    def poll(self, max_messages: int = 50) -> int:
        """Non-blocking drain of ZMQ queues.

        The dedicated observation socket is drained first — all buffered
        frames are consumed but only the latest observation is dispatched.
        Telemetry and action sockets are then drained up to *max_messages*
        so status metrics stay current.

        Returns:
            Number of messages processed.
        """
        self._observation_updated = False
        count = 0

        # --- Drain observation socket, keep only the latest frame ---
        if self._sock_obs is not None:
            latest_topic: str | None = None
            latest_msg: object | None = None
            while True:
                try:
                    topic_bytes, data = self._sock_obs.recv_multipart(
                        flags=zmq.NOBLOCK,
                    )
                    latest_topic = topic_bytes.decode("utf-8")
                    latest_msg = deserialize(data)
                    count += 1
                except zmq.Again:
                    break
            if latest_topic is not None and latest_msg is not None:
                self._dispatch(latest_topic, latest_msg)
                self._msg_total += count
                if latest_topic in self._topic_counts:
                    self._topic_counts[latest_topic] += count

        # --- Environment and telemetry sockets: drain all available ---
        while count < max_messages:
            socks = dict(self._poller.poll(0))
            # Remove observation socket from this check — already drained
            if self._sock_obs is not None:
                socks.pop(self._sock_obs, None)
            if not socks:
                break

            if self._sock_matrix is not None and self._sock_matrix in socks:
                count += self._recv_from(self._sock_matrix, limit=max_messages - count)
            if self._sock_actor is not None and self._sock_actor in socks:
                count += self._recv_from(self._sock_actor, limit=max_messages - count)

        now = time.monotonic()
        if now - self._last_diag_t >= 5.0:
            _LOG.info(
                "auditor.stream poll total=%d dropped=%d obs_up=%s selected_actor=%s topics={dm:%d action:%d telem:%d}",
                self._msg_total,
                self._drop_total,
                self._observation_updated,
                str(self._selected_actor_id),
                self._topic_counts[TOPIC_DISTANCE_MATRIX],
                self._topic_counts[TOPIC_ACTION],
                self._topic_counts[TOPIC_TELEMETRY_EVENT],
            )
            self._last_diag_t = now
        return count

    def send_step_request(
        self,
        linear_velocity: float,
        yaw_rate: float,
        vertical_velocity: float = 0.0,
    ) -> None:
        """Send manual teleop StepRequest via REQ socket."""
        if self._sock_step is None:
            return
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.array(
                [[linear_velocity, vertical_velocity, 0.0]],
                dtype=np.float32,
            ),
            angular_velocity=np.array(
                [[0.0, 0.0, yaw_rate]],
                dtype=np.float32,
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
        try:
            self._sock_step.send(serialize(request))
            # Wait for reply with the timeout set in __init__
            _reply = self._sock_step.recv()
        except zmq.Again:
            # Environment is busy or offline, just drop this teleop step
            pass
        except Exception:  # noqa: S110
            # Log other ZMQ errors if necessary
            pass

    @property
    def has_step_socket(self) -> bool:
        """Whether manual stepping is available."""
        return self._sock_step is not None

    @property
    def observation_updated(self) -> bool:
        """Whether the latest ``poll()`` delivered a new observation frame."""
        return self._observation_updated

    def close(self) -> None:
        """Tear down all sockets."""
        if self._sock_matrix is not None:
            self._sock_matrix.close()
        if self._sock_actor is not None:
            self._sock_actor.close()
        if self._sock_obs is not None:
            self._sock_obs.close()
        if self._sock_step is not None:
            self._sock_step.close()
        self._ctx.term()

    # ── internal routing ─────────────────────────────────────────────

    def _recv_from(self, sock: zmq.Socket[bytes], limit: int = 10) -> int:
        """Drain one socket up to limit, dispatching by topic."""
        count = 0
        while count < limit:
            try:
                topic_bytes, data = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            topic = topic_bytes.decode("utf-8")
            msg = deserialize(data)
            self._dispatch(topic, msg)
            self._msg_total += 1
            if topic in self._topic_counts:
                self._topic_counts[topic] += 1
            count += 1
        return count

    @property
    def actor_states(self) -> dict[int, StreamState]:
        """Per-actor stream states keyed by actor ID."""
        return self._actor_states

    @property
    def n_actors(self) -> int:
        """Fleet actor count reported by the trainer via perf telemetry."""
        return self._fleet_n_actors if self._fleet_n_actors > 0 else len(self._actor_states)

    def _resolve_state(self, actor_id: int) -> StreamState:
        """Return the StreamState for *actor_id*, creating it if necessary."""
        if actor_id not in self._actor_states:
            self._actor_states[actor_id] = StreamState()
        return self._actor_states[actor_id]

    def _dispatch(self, topic: str, msg: object) -> None:
        """Route a deserialized message to the appropriate per-actor state."""
        if topic == TOPIC_DISTANCE_MATRIX and isinstance(msg, DistanceMatrix):
            actor_id = int(msg.env_ids[0]) if len(msg.env_ids) > 0 else 0
            state = self._resolve_state(actor_id)
            state.last_rx_time = time.time()
            if self._selected_actor_id is not None and actor_id != self._selected_actor_id:
                self._drop_total += 1
                return
            state.latest_matrix = msg
            self._observation_updated = True
            pose = msg.robot_pose
            state.pose_history.append((pose.x, pose.z, pose.yaw))

        elif topic == TOPIC_ACTION and isinstance(msg, Action):
            actor_id = int(msg.env_ids[0]) if len(msg.env_ids) > 0 else 0
            state = self._resolve_state(actor_id)
            state.last_rx_time = time.time()
            if self._selected_actor_id is not None and actor_id != self._selected_actor_id:
                self._drop_total += 1
                return
            state.latest_action = msg

        elif topic == TOPIC_TELEMETRY_EVENT and isinstance(msg, TelemetryEvent):
            actor_id = msg.env_id
            state = self._resolve_state(actor_id)
            state.last_rx_time = time.time()
            if self._selected_actor_id is not None and actor_id != self._selected_actor_id:
                self._drop_total += 1
                return
            state.telemetry_buffer.append(msg)
            self._route_telemetry(msg, state)

    def _route_telemetry(
        self,
        event: TelemetryEvent,
        state: StreamState,
    ) -> None:
        """Parse telemetry event type and push into typed ring buffers."""
        et = event.event_type
        p = event.payload

        if et == "actor.training.ppo.step" and len(p) >= 9:
            state.ppo_raw_reward_history.append(float(p[0]))
            state.ppo_shaped_reward_history.append(float(p[1]))
            state.ppo_intrinsic_reward_history.append(float(p[2]))
            state.ppo_loop_similarity_history.append(float(p[3]))
            state.ppo_loop_detected_history.append(float(p[4]))
            state.ppo_beta_history.append(float(p[5]))
            state.ppo_done_history.append(float(p[6]))
            # Keep aggregate histories for compact status and mode detection.
            state.reward_history.append(float(p[1]))
            state.collision_history.append(float(p[6]))
            state.forward_cmd_history.append(float(p[7]))
            state.yaw_cmd_history.append(float(p[8]))

        elif et == "actor.training.ppo.update" and len(p) >= 9:
            state.ppo_reward_ema_history.append(float(p[0]))
            state.ppo_policy_loss_history.append(float(p[1]))
            state.ppo_value_loss_history.append(float(p[2]))
            state.ppo_entropy_history.append(float(p[3]))
            state.ppo_kl_history.append(float(p[4]))
            state.ppo_clip_fraction_history.append(float(p[5]))
            state.ppo_total_loss_history.append(float(p[6]))
            state.ppo_rnd_loss_history.append(float(p[7]))
            if len(p) >= 11:
                state.ppo_lr_history.append(float(p[9]))
                state.ppo_rnd_lr_history.append(float(p[10]))

        elif et == "actor.training.ppo.perf" and len(p) >= 8:
            state.perf_sps_history.append(float(p[0]))
            state.perf_forward_ms_history.append(float(p[1]))
            state.perf_batch_step_ms_history.append(float(p[2]))
            state.perf_memory_ms_history.append(float(p[3]))
            state.perf_transition_ms_history.append(float(p[4]))
            state.perf_tick_ms_history.append(float(p[5]))
            state.perf_zero_wait_history.append(float(p[6]))
            state.perf_opt_ms_history.append(float(p[7]))
            if len(p) >= 14:
                n = int(p[13])
                if n > 0:
                    self._fleet_n_actors = n

        elif et in ("actor.training.ppo.scene", "actor.scene") and len(p) >= 1:
            # Scene name encoded as char ordinals in float32 payload
            scene_name = "".join(chr(max(0, min(0x10FFFF, int(c)))) for c in p if int(c) > 0)
            state.current_scene_name = scene_name

        elif et == "environment.sdfdag.perf" and len(p) >= 5:
            state.env_perf_sps_history.append(float(p[0]))
            state.env_perf_batch_ms_history.append(float(p[3]))
            state.env_perf_actor_ms_history.append(float(p[4]))

        elif et == "actor.inference.perf" and len(p) >= 4:
            state.perf_sps_history.append(float(p[0]))
            state.perf_forward_ms_history.append(float(p[1]))
            state.perf_batch_step_ms_history.append(float(p[2]))
            state.perf_tick_ms_history.append(float(p[3]))
            if len(p) >= 5:
                n = int(p[4])
                if n > 0:
                    self._fleet_n_actors = n

        elif et == "actor.inference.features":
            state.latest_features = np.asarray(p, dtype=np.float32)
            # Push individual features into time-series deques
            if len(p) >= 17:
                state.front_depth_history.append(float(p[0]))
                state.mean_depth_history.append(float(p[1]))
                state.near_fraction_history.append(float(p[16]))

        elif et == "actor.step_result" and len(p) >= 4:
            # Shallow-policy step result: [reward, episode_return, done, truncated]
            state.reward_history.append(float(p[0]))
            done_val = float(p[2])
            truncated_val = float(p[3])
            state.collision_history.append(done_val)
            state.ppo_raw_reward_history.append(float(p[0]))
            state.ppo_shaped_reward_history.append(float(p[0]))
            state.ppo_done_history.append(done_val + truncated_val)

        elif et == "actor.action_published" and len(p) >= 4:
            # Action published: [fwd, lateral, vertical, yaw]
            state.forward_cmd_history.append(float(p[0]))
            state.yaw_cmd_history.append(float(p[3]))
