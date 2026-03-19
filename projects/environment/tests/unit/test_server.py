"""Environment server telemetry tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from navi_environment.config import EnvironmentConfig
from navi_environment.server import EnvironmentServer


class _FakeBackend:
    def __init__(self, snapshot: object | None) -> None:
        self._snapshot = snapshot

    def perf_snapshot(self) -> object | None:
        return self._snapshot

    def close(self) -> None:
        return None


def test_publish_backend_perf_emits_sdfdag_event() -> None:
    backend = _FakeBackend(
        SimpleNamespace(
            sps=61.5,
            last_batch_step_ms=14.0,
            ema_batch_step_ms=13.7,
            avg_batch_step_ms=14.2,
            avg_actor_step_ms=3.55,
            total_batches=100,
            total_actor_steps=400,
        )
    )
    server = EnvironmentServer(EnvironmentConfig(backend="sdfdag"), backend)
    published: dict[str, object] = {}

    def capture(*, event_type: str, step_id: int, payload: np.ndarray, actor_id: int = 0) -> None:
        published["event_type"] = event_type
        published["step_id"] = step_id
        published["payload"] = payload
        published["actor_id"] = actor_id

    server._publish_telemetry = capture  # type: ignore[method-assign]

    server._publish_backend_perf(step_id=200)

    assert published["event_type"] == "environment.sdfdag.perf"
    assert published["step_id"] == 200
    assert published["actor_id"] == 0
    payload = published["payload"]
    assert isinstance(payload, np.ndarray)
    assert payload.dtype == np.float32
    np.testing.assert_allclose(
        payload,
        np.array([61.5, 14.0, 13.7, 14.2, 3.55, 100.0, 400.0], dtype=np.float32),
    )


def test_publish_backend_perf_skips_non_sdfdag_backend() -> None:
    backend = _FakeBackend(
        SimpleNamespace(
            sps=1.0,
            last_batch_step_ms=1.0,
            ema_batch_step_ms=1.0,
            avg_batch_step_ms=1.0,
            avg_actor_step_ms=1.0,
            total_batches=1,
            total_actor_steps=1,
        )
    )
    server = EnvironmentServer(EnvironmentConfig(backend="legacy"), backend)
    published = False

    def capture(*, event_type: str, step_id: int, payload: np.ndarray, actor_id: int = 0) -> None:
        nonlocal published
        published = True

    server._publish_telemetry = capture  # type: ignore[method-assign]

    server._publish_backend_perf(step_id=100)

    assert published is False


def test_publish_backend_perf_skips_missing_snapshot() -> None:
    server = EnvironmentServer(EnvironmentConfig(backend="sdfdag"), _FakeBackend(None))
    published = False

    def capture(*, event_type: str, step_id: int, payload: np.ndarray, actor_id: int = 0) -> None:
        nonlocal published
        published = True

    server._publish_telemetry = capture  # type: ignore[method-assign]

    server._publish_backend_perf(step_id=100)

    assert published is False
