"""Tests for the Recorder node."""

from __future__ import annotations

from typing import Any

from navi_auditor.config import AuditorConfig
from navi_auditor import recorder as recorder_module
from navi_auditor.recorder import Recorder
from navi_auditor.storage.zarr_backend import ZarrBackend


class _BackendSpy:
    def __init__(self) -> None:
        self.open_calls: list[tuple[str, str]] = []

    def open(self, path: str, mode: str = "w") -> None:
        self.open_calls.append((path, mode))

    def write(self, topic: str, data: bytes, timestamp: float) -> None:
        raise NotImplementedError()

    def read_all(self) -> list[tuple[str, bytes, float]]:
        raise NotImplementedError()

    def close(self) -> None:
        return

    def __len__(self) -> int:
        return 0


class _SocketSpy:
    def __init__(self) -> None:
        self.connected: list[str] = []
        self.subscriptions: list[bytes] = []
        self.closed = False

    def connect(self, addr: str) -> None:
        self.connected.append(addr)

    def setsockopt(self, option: int, value: bytes) -> None:
        self.subscriptions.append(value)

    def close(self) -> None:
        self.closed = True


class _ContextSpy:
    def __init__(self) -> None:
        self.sockets: list[_SocketSpy] = []
        self.terminated = False

    def socket(self, _socket_type: int) -> _SocketSpy:
        sock = _SocketSpy()
        self.sockets.append(sock)
        return sock

    def term(self) -> None:
        self.terminated = True


class _ConfigStub:
    def __init__(self, output_path: str, sub_addresses: tuple[str, ...]) -> None:
        self.output_path = output_path
        self.sub_addresses = sub_addresses


class TestRecorder:
    """Tests for Recorder."""

    def test_creation(self) -> None:
        config = AuditorConfig()
        backend = ZarrBackend()
        recorder = Recorder(config=config, backend=backend)
        assert recorder is not None

    def test_start_deduplicates_identical_sub_addresses(self, monkeypatch: Any) -> None:
        context = _ContextSpy()
        monkeypatch.setattr(recorder_module.zmq, "Context", lambda: context)

        backend = _BackendSpy()
        config = _ConfigStub(output_path="session.zarr", sub_addresses=("tcp://localhost:5557", "tcp://localhost:5557"))
        recorder = Recorder(config=config, backend=backend)

        recorder.start()

        assert backend.open_calls == [(config.output_path, "w")]
        assert len(context.sockets) == 1
        assert context.sockets[0].connected == ["tcp://localhost:5557"]

        recorder.stop()
        assert context.sockets[0].closed is True
        assert context.terminated is True
