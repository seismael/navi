"""Rewinder — reads stored data and replays via ZMQ PUB."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import zmq

if TYPE_CHECKING:
    from navi_auditor.config import AuditorConfig
    from navi_auditor.storage.base import AbstractStorageBackend

__all__: list[str] = ["Rewinder"]

_PUB_SUB_READY_DELAY_SECONDS = 0.25


class Rewinder:
    """Reads recorded sessions and republishes them via ZMQ PUB.

    Used for offline replay and analysis.
    """

    def __init__(self, config: AuditorConfig, backend: AbstractStorageBackend) -> None:
        self._config = config
        self._backend = backend
        self._context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
        self._pub_socket: zmq.Socket[bytes] | None = None

    def start(self) -> None:
        """Open storage for reading and bind the PUB socket."""
        self._backend.open(self._config.output_path, mode="r")
        self._pub_socket = self._context.socket(zmq.PUB)
        self._pub_socket.bind(self._config.pub_address)
        # Give passive subscribers a brief window to complete the ZMQ slow-joiner handshake.
        time.sleep(_PUB_SUB_READY_DELAY_SECONDS)

    def replay(self, speed: float = 1.0) -> int:
        """Replay all recorded messages at the given speed multiplier.

        Args:
            speed: Playback speed multiplier (1.0 = real-time).

        Returns:
            Number of messages replayed.
        """
        if self._pub_socket is None:
            msg = "Rewinder not started. Call start() first."
            raise RuntimeError(msg)

        messages = self._backend.read_all()
        if not messages:
            return 0

        count = 0
        prev_ts = messages[0][2]

        for topic, data, ts in messages:
            # Wait proportional to the original time gap
            if count > 0:
                gap = (ts - prev_ts) / speed
                if gap > 0:
                    time.sleep(gap)

            self._pub_socket.send_multipart([topic.encode("utf-8"), data])
            prev_ts = ts
            count += 1

        return count

    def stop(self) -> None:
        """Close storage and ZMQ socket."""
        self._backend.close()
        if self._pub_socket is not None:
            self._pub_socket.close()
        self._context.term()
