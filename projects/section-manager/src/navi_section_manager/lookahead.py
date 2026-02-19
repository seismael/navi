"""LookAheadBuffer — pre-loaded chunk ring for predictive loading."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__: list[str] = ["LookAheadBuffer"]

_GenerateChunkFn = Callable[[int, int, int], NDArray[np.float32]]


class LookAheadBuffer:
    """Fixed-capacity cache of pre-generated chunk data.

    Chunks that the :class:`FrustumLoader` predicts will be needed are
    prefetched here.  When the :class:`SlidingWindow` actually enters
    those chunks, it can ``promote`` them instead of generating on-demand.
    """

    __slots__ = ("_cache", "_capacity")

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = max(1, capacity)
        self._cache: OrderedDict[tuple[int, int, int], NDArray[np.float32]] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: tuple[int, int, int]) -> NDArray[np.float32] | None:
        """Retrieve a prefetched chunk, or *None* if not cached."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def promote(
        self,
        entering: set[tuple[int, int, int]],
        generate: _GenerateChunkFn,
    ) -> dict[tuple[int, int, int], NDArray[np.float32]]:
        """Return chunk data for *entering* coords from cache or generator.

        Cached entries are consumed (removed from buffer).  Missing chunks
        are generated on the spot via *generate*.
        """
        result: dict[tuple[int, int, int], NDArray[np.float32]] = {}
        for coord in entering:
            cached = self._cache.pop(coord, None)
            if cached is not None:
                result[coord] = cached
            else:
                result[coord] = generate(*coord)
        return result

    def prefetch(
        self,
        frustum_coords: list[tuple[int, int, int]],
        generate: _GenerateChunkFn,
    ) -> None:
        """Pre-generate chunks for *frustum_coords* that are not yet cached."""
        for coord in frustum_coords:
            if coord in self._cache:
                continue
            self._cache[coord] = generate(*coord)
            # Evict oldest if over capacity
            while len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def contains(self, key: tuple[int, int, int]) -> bool:
        """Check whether *key* is in the buffer."""
        return key in self._cache

    def clear(self) -> None:
        """Drop all cached chunks."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Current number of cached chunks."""
        return len(self._cache)
