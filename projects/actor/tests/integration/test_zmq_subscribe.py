"""Integration test for Actor ZMQ subscription."""

from __future__ import annotations

import contextlib
import socket
import threading
import time

import numpy as np
import pytest
import zmq

from navi_actor.config import ActorConfig
from navi_actor.server import ActorServer
from navi_contracts import (
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _DummyPolicy:
    def act(self, observation: DistanceMatrix, step_id: int) -> Action:
        return Action(
            env_ids=np.asarray(observation.env_ids, dtype=np.int32),
            linear_velocity=np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
            angular_velocity=np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
            policy_id="test-policy",
            step_id=step_id,
            timestamp=time.time(),
        )


@pytest.mark.integration
class TestZmqSubscribe:
    """Integration tests for the Actor ZMQ SUB/PUB server."""

    def test_step_mode_end_to_end_identity(self) -> None:
        """Actor infer+step should preserve env/episode identity and publish telemetry correctly."""
        actor_port = _free_port()
        step_port = _free_port()

        actor_pub = f"tcp://127.0.0.1:{actor_port}"
        step_rep = f"tcp://127.0.0.1:{step_port}"

        cfg = ActorConfig(
            sub_address="tcp://127.0.0.1:65530",
            pub_address=actor_pub,
            mode="step",
            step_endpoint=step_rep,
            azimuth_bins=8,
            elevation_bins=4,
        )

        server = ActorServer(config=cfg, policy=_DummyPolicy())

        ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        out_sub = ctx.socket(zmq.SUB)
        out_sub.connect(actor_pub)
        out_sub.setsockopt(zmq.SUBSCRIBE, TOPIC_TELEMETRY_EVENT.encode("utf-8"))

        stop_rep = threading.Event()

        def _rep_worker() -> None:
            rep_ctx: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
            rep_socket = rep_ctx.socket(zmq.REP)
            rep_socket.bind(step_rep)
            poller = zmq.Poller()
            poller.register(rep_socket, zmq.POLLIN)
            try:
                while not stop_rep.is_set():
                    events = dict(poller.poll(100))
                    if rep_socket not in events:
                        continue
                    req_raw = rep_socket.recv()
                    req = deserialize(req_raw)
                    assert isinstance(req, StepRequest)
                    actor_id = int(req.action.env_ids[0]) if len(req.action.env_ids) > 0 else 0
                    result = StepResult(
                        step_id=req.step_id,
                        env_id=actor_id,
                        episode_id=42,
                        done=False,
                        truncated=False,
                        reward=1.25,
                        episode_return=3.5,
                        timestamp=time.time(),
                    )
                    rep_socket.send(serialize(result))
            finally:
                rep_socket.close(0)
                rep_ctx.term()

        rep_thread = threading.Thread(target=_rep_worker, daemon=True)
        rep_thread.start()

        try:
            server.start()
            time.sleep(0.2)

            obs = DistanceMatrix(
                episode_id=7,
                env_ids=np.array([3], dtype=np.int32),
                matrix_shape=(8, 4),
                depth=np.ones((1, 8, 4), dtype=np.float32) * 0.5,
                delta_depth=np.zeros((1, 8, 4), dtype=np.float32),
                semantic=np.zeros((1, 8, 4), dtype=np.int32),
                valid_mask=np.ones((1, 8, 4), dtype=np.bool_),
                overhead=np.zeros((8, 8, 3), dtype=np.float32),
                robot_pose=RobotPose(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                    timestamp=time.time(),
                ),
                step_id=1,
                timestamp=time.time(),
            )

            action = server.infer(obs)
            result = server.step(action)

            assert int(action.env_ids[0]) == 3
            assert result.env_id == 3
            assert result.episode_id == 42

            got_step_result_telemetry = False
            deadline = time.time() + 3.0
            poller = zmq.Poller()
            poller.register(out_sub, zmq.POLLIN)
            while time.time() < deadline and not got_step_result_telemetry:
                server._publish_telemetry(
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
                events = dict(poller.poll(100))
                if out_sub not in events:
                    continue
                topic_b, payload = out_sub.recv_multipart()
                topic = topic_b.decode("utf-8")
                msg = deserialize(payload)

                if (
                    topic == TOPIC_TELEMETRY_EVENT
                    and isinstance(msg, TelemetryEvent)
                    and msg.event_type == "actor.step_result"
                ):
                    got_step_result_telemetry = True
                    assert msg.env_id == 3
                    assert msg.episode_id == 42

            assert got_step_result_telemetry, "timed out waiting for actor.step_result telemetry"
        finally:
            with contextlib.suppress(Exception):
                server.stop()
            stop_rep.set()
            rep_thread.join(timeout=1.0)
            out_sub.close(0)
            ctx.term()
