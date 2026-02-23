"""Tests for LookAheadBuffer."""

from __future__ import annotations

import numpy as np

from navi_environment.lookahead import LookAheadBuffer


def _dummy_generate(cx: int, cy: int, cz: int) -> np.ndarray:
    """Return a tiny test chunk marked with chunk coords."""
    chunk = np.zeros((2, 2, 2, 2), dtype=np.float32)
    chunk[0, 0, 0, 0] = float(cx)
    chunk[0, 0, 0, 1] = float(cy + cz)
    return chunk


class TestLookAheadBuffer:
    """Unit tests for the prefetch cache."""

    def test_prefetch_and_get(self) -> None:
        buf = LookAheadBuffer(capacity=16)
        buf.prefetch([(1, 2, 3)], _dummy_generate)
        assert buf.contains((1, 2, 3))
        cached = buf.get((1, 2, 3))
        assert cached is not None
        assert cached[0, 0, 0, 0] == 1.0

    def test_get_missing_returns_none(self) -> None:
        buf = LookAheadBuffer(capacity=16)
        assert buf.get((99, 99, 99)) is None

    def test_promote_uses_cache(self) -> None:
        buf = LookAheadBuffer(capacity=16)
        buf.prefetch([(0, 0, 0)], _dummy_generate)
        result = buf.promote({(0, 0, 0)}, _dummy_generate)
        assert (0, 0, 0) in result
        # After promote, the entry is consumed
        assert not buf.contains((0, 0, 0))

    def test_promote_generates_missing(self) -> None:
        buf = LookAheadBuffer(capacity=16)
        result = buf.promote({(5, 5, 5)}, _dummy_generate)
        assert (5, 5, 5) in result
        assert result[(5, 5, 5)][0, 0, 0, 0] == 5.0

    def test_capacity_eviction(self) -> None:
        buf = LookAheadBuffer(capacity=3)
        coords = [(i, 0, 0) for i in range(5)]
        buf.prefetch(coords, _dummy_generate)
        assert buf.size == 3
        # Oldest entries (0,0,0) and (1,0,0) should have been evicted
        assert not buf.contains((0, 0, 0))
        assert not buf.contains((1, 0, 0))
        assert buf.contains((4, 0, 0))

    def test_clear(self) -> None:
        buf = LookAheadBuffer(capacity=16)
        buf.prefetch([(0, 0, 0), (1, 1, 1)], _dummy_generate)
        assert buf.size == 2
        buf.clear()
        assert buf.size == 0
