"""CityGenerator — grid-based city with streets and buildings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navi_section_manager.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["CityGenerator"]

# Semantic IDs
_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0
_ROAD: float = 7.0
_BUILDING: float = 8.0
_WINDOW: float = 9.0


class CityGenerator(AbstractWorldGenerator):
    """Generates a city grid with streets and buildings of varying height.

    The city is laid out on a regular grid: each chunk is either a
    *street* chunk (flat road surface) or a *building* chunk (solid
    block with window holes).  Streets run every 3rd chunk in both
    X and Z, creating a walkable grid with good line-of-sight down
    corridors and interesting depth layering.
    """

    __slots__ = ("_chunk_size", "_seed", "_street_every")

    def __init__(
        self,
        seed: int = 42,
        chunk_size: int = 16,
        street_every: int = 3,
    ) -> None:
        self._seed = seed
        self._chunk_size = chunk_size
        self._street_every = max(2, street_every)

    def _is_street(self, cx: int, cz: int) -> bool:
        """True if the chunk is part of a street corridor."""
        return cx % self._street_every == 0 or cz % self._street_every == 0

    def _building_height(self, cx: int, cz: int) -> int:
        """Deterministic building height in voxels (spans multiple cy layers)."""
        rng = np.random.default_rng(hash((self._seed, cx, cz)) & 0xFFFFFFFF)
        # Buildings range from 8 to 40 voxels tall
        return int(rng.integers(8, 41))

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        cz: int,
    ) -> NDArray[np.float32]:
        """Generate a city chunk at ``(cx, cy, cz)``."""
        c = self._chunk_size
        chunk = np.zeros((c, c, c, 2), dtype=np.float32)

        if cy < 0:
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        is_street = self._is_street(cx, cz)

        if is_street:
            # Street chunk: flat road surface at y=0
            if cy == 0:
                chunk[:, 0, :, 0] = 1.0
                chunk[:, 0, :, 1] = _ROAD
            # Above ground: empty air
            return chunk

        # Building chunk
        height = self._building_height(cx, cz)
        y_base = cy * c

        if y_base >= height:
            # Entirely above building — empty
            return chunk

        # Fill building body
        rng = np.random.default_rng(hash((self._seed, cx, cz, cy)) & 0xFFFFFFFF)
        for ly in range(c):
            world_y = y_base + ly
            if world_y == 0:
                # Ground floor
                chunk[:, ly, :, 0] = 1.0
                chunk[:, ly, :, 1] = _FLOOR
            elif world_y < height:
                # Building body — solid walls with window holes
                chunk[:, ly, :, 0] = 1.0
                chunk[:, ly, :, 1] = _BUILDING

                # Carve window holes on exterior faces every 3 voxels
                if world_y % 3 == 1:
                    for i in range(2, c - 2, 3):
                        # Windows on x=0 face
                        chunk[0, ly, i, 0] = 0.0
                        chunk[0, ly, i, 1] = _AIR
                        # Windows on x=c-1 face
                        chunk[c - 1, ly, i, 0] = 0.0
                        chunk[c - 1, ly, i, 1] = _AIR
                        # Windows on z=0 face
                        chunk[i, ly, 0, 0] = 0.0
                        chunk[i, ly, 0, 1] = _AIR
                        # Windows on z=c-1 face
                        chunk[i, ly, c - 1, 0] = 0.0
                        chunk[i, ly, c - 1, 1] = _AIR

                # Carve interior hallway through each floor
                if rng.random() < 0.6:
                    hw = c // 2
                    for i in range(c):
                        chunk[hw, ly, i, 0] = 0.0
                        chunk[hw, ly, i, 1] = _AIR
                        chunk[hw + 1, ly, i, 0] = 0.0
                        chunk[hw + 1, ly, i, 1] = _AIR

            elif world_y == height:
                # Roof
                chunk[:, ly, :, 0] = 1.0
                chunk[:, ly, :, 1] = _FLOOR

        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Spawn on the first street intersection near the origin."""
        c = self._chunk_size
        # Street at cx=0, cz=0 is always a street
        return (c / 2.0, 1.5, c / 2.0)
