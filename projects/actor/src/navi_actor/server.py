"""ZMQ SUB/PUB server for the Brain service using DistanceMatrix v2."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import numpy as np
import zmq

from navi_actor.policy import ShallowPolicy
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


class _RuntimePolicyProtocol(Protocol):
    def act(self, observation: DistanceMatrix, step_id: int) -> Action:
        ...

__all__: list[str] = ["ActorServer"]


class ActorServer:
    """Brain ZMQ server.

    Modes:
    - async: subscribes to DistanceMatrix and publishes Action.
    - step: after infer, sends StepRequest and waits for StepResult.
    """

    def __init__(self, config: ActorConfig, policy: _RuntimePolicyProtocol | None = None) -> None:
        self._config = config
        self._step_counter: int = 0
        self._policy = policy if policy is not None else ShallowPolicy(policy_id="brain-v2-shallow", gain=0.5)

        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._sub_socket: zmq.Socket[bytes] | None = None
        self._pub_socket: zmq.Socket[bytes] | None = None
        self._step_socket: zmq.Socket[bytes] | None = None

    def start(self) -> None:
        """Bind PUB socket, connect SUB socket, optionally connect REQ."""
        cfg = self._config

        sub_socket = self._context.socket(zmq.SUB)
        sub_socket.connect(cfg.sub_address)
        sub_socket.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
        self._sub_socket = sub_socket

        pub_socket = self._context.socket(zmq.PUB)
        pub_socket.bind(cfg.pub_address)
        self._pub_socket = pub_socket

        if cfg.mode == "step":
            step_socket = self._context.socket(zmq.REQ)
            step_socket.connect(cfg.step_endpoint)
            self._step_socket = step_socket

    def infer(self, update: DistanceMatrix) -> Action:
        """Compute Action from DistanceMatrix using the shallow policy."""
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
        """Run the main subscribe-infer-publish loop."""
        self.start()
        assert self._sub_socket is not None
        assert self._pub_socket is not None
        try:
            while True:
                _topic, data = self._sub_socket.recv_multipart()
                msg = deserialize(data)
                if isinstance(msg, DistanceMatrix):
                    action = self.infer(msg)
                    payload = serialize(action)
                    self._pub_socket.send_multipart(
                        [
                            TOPIC_ACTION.encode("utf-8"),
                            payload,
                        ]
                    )
                    self._publish_telemetry(
                        event_type="actor.action_published",
                        step_id=action.step_id,
                        payload=np.array(
                            [
                                float(action.linear_velocity[0, 0]),
                                float(action.linear_velocity[0, 1]),
                                float(action.linear_velocity[0, 2]),
                                float(action.angular_velocity[0, 2]),
                            ],
                            dtype=np.float32,
                        ),
                    )
                    if self._config.mode == "step" and self._step_socket is not None:
                        result = self.step(action)
                        self._publish_telemetry(
                            event_type="actor.step_result",
                            step_id=result.step_id,
                            payload=np.array(
                                [
                                    result.reward,
                                    result.episode_return,
                                    float(result.done),
                                    float(result.truncated),
                                ],
                                dtype=np.float32,
                            ),
                        )
                    else:
                        self._step_counter += 1
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

    def _publish_telemetry(self, event_type: str, step_id: int, payload: np.ndarray) -> None:
        """Publish a TelemetryEvent on the actor PUB socket."""
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
