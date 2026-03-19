"""Tests for the Rewinder."""

from __future__ import annotations

from typing import Any

from navi_auditor import rewinder as rewinder_module
from navi_auditor.config import AuditorConfig
from navi_auditor.rewinder import Rewinder
from navi_auditor.storage.zarr_backend import ZarrBackend


class _BackendSpy:
    def __init__(self) -> None:
        self.open_calls: list[tuple[str, str]] = []
        self.closed = False

    def open(self, path: str, mode: str = "r") -> None:
        self.open_calls.append((path, mode))

    def read_all(self) -> list[tuple[str, bytes, float]]:
        return []

    def close(self) -> None:
        self.closed = True


class _SocketSpy:
    def __init__(self) -> None:
        self.bound: list[str] = []
        self.closed = False

    def bind(self, address: str) -> None:
        self.bound.append(address)

    def close(self) -> None:
        self.closed = True


class _ContextSpy:
    def __init__(self) -> None:
        self.sockets: list[_SocketSpy] = []
        self.terminated = False

    def socket(self, _socket_type: int) -> _SocketSpy:
        socket = _SocketSpy()
        self.sockets.append(socket)
        return socket

    def term(self) -> None:
        self.terminated = True


class TestRewinder:
    """Tests for Rewinder."""

    def test_creation(self) -> None:
        config = AuditorConfig()
        backend = ZarrBackend()
        rewinder = Rewinder(config=config, backend=backend)
        assert rewinder is not None

    def test_start_waits_briefly_after_binding_pub_socket(self, monkeypatch: Any) -> None:
        context = _ContextSpy()
        sleep_calls: list[float] = []

        monkeypatch.setattr(rewinder_module.zmq, "Context", lambda: context)
        monkeypatch.setattr(
            rewinder_module.time, "sleep", lambda seconds: sleep_calls.append(seconds)
        )

        config = AuditorConfig(output_path="session.zarr", pub_address="tcp://*:5558")
        backend = _BackendSpy()
        rewinder = Rewinder(config=config, backend=backend)

        rewinder.start()

        assert backend.open_calls == [("session.zarr", "r")]
        assert len(context.sockets) == 1
        assert context.sockets[0].bound == ["tcp://*:5558"]
        assert sleep_calls == [rewinder_module._PUB_SUB_READY_DELAY_SECONDS]

        rewinder.stop()
        assert backend.closed is True
        assert context.sockets[0].closed is True
        assert context.terminated is True
