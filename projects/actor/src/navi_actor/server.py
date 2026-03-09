"""ZMQ SUB/PUB server for the Brain service using DistanceMatrix v2."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import zmq

from navi_actor.spherical_features import extract_spherical_features
from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    Action,
    DistanceMatrix,
    StepRequest,
    StepResult,
    TelemetryEvent,
    deserialize,
    serialize,
)

if TYPE_CHECKING:
    from navi_actor.config import ActorConfig

_log = logging.getLogger(__name__)

# Telemetry decimation: publish every Nth step (keeps training loop fast)
_TELEMETRY_EVERY_N: int = 10

# Step-rate logging interval in seconds
_RATE_LOG_INTERVAL: float = 10.0


class _RuntimePolicyProtocol(Protocol):
    def act(self, observation: DistanceMatrix, step_id: int) -> Action:
        ...


class _StatefulPolicyProtocol(Protocol):
    """Protocol for policies with recurrent hidden state."""

    def act(
        self, obs: Any, step_id: int, hidden: Any,
    ) -> tuple[list[float], Any]:
        ...

__all__: list[str] = ["ActorServer"]


class ActorServer:
    """Brain ZMQ server.

    Modes:
    - async: subscribes to DistanceMatrix and publishes Action.
    - step: after infer, sends StepRequest and waits for StepResult.

    Supports both stateless (_RuntimePolicyProtocol) and stateful
    (_StatefulPolicyProtocol) policies. Stateful policies maintain
    a hidden state that is reset on episode boundaries.

    Telemetry is decimated to every Nth step so the training loop
    spends minimal time on dashboard observability.
    """

    def __init__(
        self,
        config: ActorConfig,
        policy: Any = None,
    ) -> None:
        self._config = config
        self._step_counter: int = 0
        self._policy: Any
        if policy is not None:
            self._policy = policy
        else:
            from navi_actor.cognitive_policy import CognitiveMambaPolicy

            self._policy = CognitiveMambaPolicy(
                embedding_dim=config.embedding_dim,
                azimuth_bins=config.azimuth_bins,
                elevation_bins=config.elevation_bins,
                max_forward=config.max_forward,
                max_vertical=config.max_vertical,
                max_lateral=config.max_lateral,
                max_yaw=config.max_yaw,
            )

        self._stateful = _is_stateful(self._policy)
        self._hidden_state: Any = None  # recurrent hidden state

        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sub_socket: zmq.Socket[bytes] | None = None
        self._pub_socket: zmq.Socket[bytes] | None = None
        self._step_socket: zmq.Socket[bytes] | None = None

        # Step-rate tracking
        self._rate_t0: float = 0.0
        self._rate_steps: int = 0

    def start(self) -> None:
        """Bind PUB socket, connect SUB socket, optionally connect REQ."""
        cfg = self._config

        sub_socket = self._context.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.RCVHWM, 500)
        sub_socket.setsockopt(zmq.LINGER, 0)
        sub_socket.connect(cfg.sub_address)
        sub_socket.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
        self._sub_socket = sub_socket

        pub_socket = self._context.socket(zmq.PUB)
        pub_socket.setsockopt(zmq.SNDHWM, 500)
        pub_socket.setsockopt(zmq.LINGER, 0)
        pub_socket.bind(cfg.pub_address)
        self._pub_socket = pub_socket

        if cfg.mode == "step":
            step_socket = self._context.socket(zmq.REQ)
            step_socket.setsockopt(zmq.LINGER, 0)
            step_socket.connect(cfg.step_endpoint)
            self._step_socket = step_socket

        self._rate_t0 = time.monotonic()
        self._rate_steps = 0

    def infer(self, update: DistanceMatrix) -> Action:
        """Compute Action from DistanceMatrix using the active policy.

        Handles both stateless and stateful (recurrent) policies.
        """
        if self._stateful:
            action_list, self._hidden_state = self._policy.act(
                update, step_id=self._step_counter, hidden=self._hidden_state,
            )
            return Action(
                env_ids=np.asarray(update.env_ids, dtype=np.int32),
                linear_velocity=np.array(
                    [[action_list[0], action_list[1], action_list[2]]],
                    dtype=np.float32,
                ),
                angular_velocity=np.array(
                    [[0.0, 0.0, action_list[3]]], dtype=np.float32
                ),
                policy_id="cognitive-mamba",
                step_id=self._step_counter,
                timestamp=time.time(),
            )

        action = self._policy.act(update, step_id=self._step_counter)
        return Action(
            env_ids=action.env_ids,
            linear_velocity=action.linear_velocity,
            angular_velocity=action.angular_velocity,
            policy_id=action.policy_id,
            step_id=action.step_id,
            timestamp=time.time(),
        )

    def step(self, action: Action) -> StepResult:
        """Send a StepRequest to Simulation Layer and return StepResult."""
        if self._step_socket is None:
            msg = "Step socket not initialised (mode != step)."
            raise RuntimeError(msg)

        request = StepRequest(
            action=action,
            step_id=self._step_counter,
            timestamp=time.time(),
        )
        self._step_counter += 1
        self._step_socket.send(serialize(request))
        reply_data = self._step_socket.recv()
        reply = deserialize(reply_data)
        if not isinstance(reply, StepResult):
            msg = f"Expected StepResult, got {type(reply).__name__}"
            raise TypeError(msg)
        return reply

    def run(self) -> None:
        """Run the main subscribe-infer-publish loop.

        The step loop is kept as tight as possible:
        1. recv DistanceMatrix (blocks until next observation)
        2. infer action (policy.act)
        3. publish Action on PUB (for dashboard)
        4. send StepRequest on REQ → wait for StepResult on REP
        5. every Nth step, publish decimated telemetry on PUB

        Telemetry is decimated (every Nth step) so the per-step cost
        of dashboard observability is near zero.
        """
        self.start()
        assert self._sub_socket is not None
        assert self._pub_socket is not None
        try:
            while True:
                _topic, data = self._sub_socket.recv_multipart()
                msg = deserialize(data)
                if not isinstance(msg, DistanceMatrix):
                    continue

                action = self.infer(msg)

                # Publish Action on PUB (lightweight, dashboard needs it)
                self._pub_socket.send_multipart(
                    [
                        TOPIC_ACTION.encode("utf-8"),
                        serialize(action),
                    ],
                    flags=zmq.NOBLOCK,
                )

                # ── Step mode: REQ/REP with Environment ──────────
                if self._config.mode == "step" and self._step_socket is not None:
                    result = self.step(action)
                    actor_id = int(action.env_ids[0]) if len(action.env_ids) > 0 else 0
                    # Reset hidden state on episode boundaries
                    if self._stateful and (result.done or result.truncated):
                        self._hidden_state = None

                    # ── Decimated telemetry (every Nth step) ─────────
                    if self._step_counter % _TELEMETRY_EVERY_N == 0:
                        self._publish_telemetry(
                            "actor.action_published",
                            action.step_id,
                            np.array(
                                [
                                    float(action.linear_velocity[0, 0]),
                                    float(action.linear_velocity[0, 1]),
                                    float(action.linear_velocity[0, 2]),
                                    float(action.angular_velocity[0, 2]),
                                ],
                                dtype=np.float32,
                            ),
                            env_id=actor_id,
                            episode_id=result.episode_id,
                        )
                        # Compute features only when publishing
                        features = extract_spherical_features(msg)
                        self._publish_telemetry(
                            "actor.inference.features",
                            action.step_id,
                            features,
                            env_id=actor_id,
                            episode_id=result.episode_id,
                        )
                    # Always publish step_result (reward chart needs it)
                    self._publish_telemetry(
                        "actor.step_result",
                        result.step_id,
                        np.array(
                            [
                                result.reward,
                                result.episode_return,
                                float(result.done),
                                float(result.truncated),
                            ],
                            dtype=np.float32,
                        ),
                        env_id=result.env_id,
                        episode_id=result.episode_id,
                    )
                else:
                    self._step_counter += 1

                # ── Step-rate logging ────────────────────────────────
                self._rate_steps += 1
                now = time.monotonic()
                elapsed = now - self._rate_t0
                if elapsed >= _RATE_LOG_INTERVAL:
                    rate = self._rate_steps / elapsed
                    _log.info(
                        "step_rate=%.1f steps/s  (steps=%d, window=%.1fs)",
                        rate, self._rate_steps, elapsed,
                    )
                    self._rate_t0 = now
                    self._rate_steps = 0

        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Shut down ZMQ sockets."""
        for sock in (self._sub_socket, self._pub_socket, self._step_socket):
            if sock is not None:
                sock.close()
        self._context.term()

    # ── Telemetry helpers ────────────────────────────────────────────

    def _publish_telemetry(
        self,
        event_type: str,
        step_id: int,
        payload: np.ndarray,
        *,
        env_id: int = 0,
        episode_id: int = 0,
    ) -> None:
        """Serialize and publish a single telemetry event (non-blocking)."""
        if self._pub_socket is None:
            return
        event = TelemetryEvent(
            event_type=event_type,
            episode_id=episode_id,
            env_id=env_id,
            step_id=step_id,
            payload=payload.astype(np.float32, copy=False),
            timestamp=time.time(),
        )
        with contextlib.suppress(zmq.Again):
            self._pub_socket.send_multipart(
                [TOPIC_TELEMETRY_EVENT.encode("utf-8"), serialize(event)],
                flags=zmq.NOBLOCK,
            )


def _is_stateful(policy: object) -> bool:
    """Check if a policy follows the stateful (recurrent) protocol.

    A stateful policy's ``act`` method accepts a ``hidden`` keyword argument
    and returns a ``(action_list, hidden)`` tuple.
    """
    import inspect

    act_method = getattr(policy, "act", None)
    if act_method is None:
        return False
    sig = inspect.signature(act_method)
    params = list(sig.parameters.keys())
    return "hidden" in params
