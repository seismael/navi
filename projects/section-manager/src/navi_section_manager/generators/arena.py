"""ArenaGenerator — open arena with pillars, ramps, and obstacles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navi_section_manager.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["ArenaGenerator"]

# Semantic IDs
_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0
_PILLAR: float = 4.0
_RAMP: float = 5.0
_OBSTACLE: float = 6.0


class ArenaGenerator(AbstractWorldGenerator):
    """Open arena with scattered pillars, ramps, and block obstacles.

    Produces a rich, navigable environment with objects at varied
    distances so the depth map shows clear structure and contrast.
    The arena is a large flat floor with a perimeter wall and
    procedurally-placed interior geometry.
    """

    __slots__ = ("_chunk_size", "_seed")

    def __init__(self, seed: int = 42, chunk_size: int = 16) -> None:
        self._seed = seed
        self._chunk_size = chunk_size

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        cz: int,
    ) -> NDArray[np.float32]:
        """Generate an arena chunk at ``(cx, cy, cz)``."""
        c = self._chunk_size
        chunk = np.zeros((c, c, c, 2), dtype=np.float32)

        if cy < 0:
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        if cy > 1:
            # High sky: empty
            return chunk

        # Lay floor at y=0 for cy==0
        if cy == 0:
            chunk[:, 0, :, 0] = 1.0
            chunk[:, 0, :, 1] = _FLOOR

        # Perimeter walls — create bounding walls at world boundary chunks
        # Wall ring at |x| or |z| >= 6 chunks from origin
        boundary = 6
        if (abs(cx) == boundary or abs(cz) == boundary) and cy in (0, 1):
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        if abs(cx) > boundary or abs(cz) > boundary:
            return chunk

        # Interior geometry — deterministic per chunk
        rng = np.random.default_rng(hash((self._seed, cx, cz)) & 0xFFFFFFFF)

        # Skip origin chunk (spawn area) — keep it clear
        if cx == 0 and cz == 0:
            return chunk

        # Place 0-3 features per chunk based on hash
        n_features = rng.integers(0, 4)
        for _ in range(n_features):
            kind = rng.integers(0, 3)
            fx = rng.integers(2, c - 2)
            fz = rng.integers(2, c - 2)

            if kind == 0:
                # Pillar — 2x2 base, up to 12 voxels tall
                height = int(rng.integers(4, min(13, c)))
                for dx in range(2):
                    for dz in range(2):
                        px, pz = fx + dx, fz + dz
                        if 0 <= px < c and 0 <= pz < c:
                            if cy == 0:
                                for ly in range(1, height):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _PILLAR
                            elif cy == 1 and height > c:
                                for ly in range(0, height - c):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _PILLAR

            elif kind == 1:
                # Block obstacle — 3x3 base, 2-4 tall
                height = int(rng.integers(2, 5))
                if cy == 0:
                    for dx in range(3):
                        for dz in range(3):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(1, min(height + 1, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _OBSTACLE

            else:
                # Ramp — wedge shape along x
                if cy == 0:
                    ramp_len = rng.integers(3, 7)
                    for dx in range(ramp_len):
                        ramp_height = max(1, int((dx + 1) * 2 // ramp_len + 1))
                        for dz in range(2):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(1, min(ramp_height + 1, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _RAMP

        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Spawn in the centre of the origin chunk, above the floor."""
        half = self._chunk_size / 2.0
        return (half, 1.5, half)
