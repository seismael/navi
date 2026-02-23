"""RoomsGenerator — structured rooms with clear walls, barriers, and blockers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navi_environment.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["RoomsGenerator"]

# Semantic IDs
_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0
_CEILING: float = 3.0
_OBSTACLE: float = 6.0


class RoomsGenerator(AbstractWorldGenerator):
    """Generates interconnected rectangular rooms with corridors and blockers.

    Every chunk at ground level (``cy == 0``) contains a recognisable
    interior: thick walls, a floor, a ceiling, and 1-4 doorways that
    connect to neighbouring chunks.  The origin chunk always produces
    a room so the spawn point is inside a clearly walled space.

    Chunk layout is deterministic per ``(seed, cx, cz)``.
    """

    __slots__ = (
        "_chunk_size",
        "_corridor_height",
        "_seed",
        "_wall_thickness",
        "_doorway_probability",
    )

    def __init__(
        self,
        seed: int = 42,
        chunk_size: int = 16,
        corridor_height: int = 5,
        wall_thickness: int = 2,
        doorway_probability: float = 0.55,
    ) -> None:
        self._seed = seed
        self._chunk_size = chunk_size
        self._corridor_height = min(corridor_height, chunk_size - 2)
        self._wall_thickness = max(1, min(wall_thickness, 3))
        self._doorway_probability = max(0.1, min(0.9, doorway_probability))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        cz: int,
    ) -> NDArray[np.float32]:
        """Generate a rooms chunk at ``(cx, cy, cz)``."""
        c = self._chunk_size
        chunk = np.zeros((c, c, c, 2), dtype=np.float32)

        if cy < 0:
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        if cy > 0:
            return chunk

        # cy == 0 — ground level
        # Step 1: floor everywhere
        chunk[:, 0, :, 0] = 1.0
        chunk[:, 0, :, 1] = _FLOOR

        # Step 2: ceiling
        ceil_y = min(1 + self._corridor_height, c - 1)
        chunk[:, ceil_y, :, 0] = 1.0
        chunk[:, ceil_y, :, 1] = _CEILING

        # Step 3: solid above ceiling
        for ly in range(ceil_y + 1, c):
            chunk[:, ly, :, 0] = 1.0
            chunk[:, ly, :, 1] = _WALL

        # Step 4: perimeter walls (all four edges of the XZ plane)
        wt = self._wall_thickness
        self._fill_wall(chunk, 0, wt, 0, c, ceil_y)         # -X wall
        self._fill_wall(chunk, c - wt, c, 0, c, ceil_y)     # +X wall
        self._fill_wall(chunk, 0, c, 0, wt, ceil_y)         # -Z wall
        self._fill_wall(chunk, 0, c, c - wt, c, ceil_y)     # +Z wall

        # Step 5: interior features — divider walls and pillars
        rng = np.random.default_rng(
            hash((self._seed, cx, cz)) & 0xFFFFFFFF,
        )
        self._place_interior(chunk, rng, ceil_y, cx, cz)

        # Step 6: doorways — carve openings in the perimeter walls
        # Each face has a deterministic chance of a doorway.
        self._carve_doorways(chunk, cx, cz, ceil_y)

        # Step 7: keep spawn zone clear in origin chunk.
        if cx == 0 and cz == 0:
            center = c // 2
            for lx in range(max(0, center - 1), min(c, center + 2)):
                for lz in range(max(0, center - 1), min(c, center + 2)):
                    for ly in range(1, ceil_y):
                        chunk[lx, ly, lz, 0] = 0.0
                        chunk[lx, ly, lz, 1] = _AIR

        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Spawn in the centre of the origin chunk, above the floor."""
        half = self._chunk_size / 2.0
        return (half, 1.5, half)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_wall(
        self,
        chunk: NDArray[np.float32],
        x0: int,
        x1: int,
        z0: int,
        z1: int,
        ceil_y: int,
    ) -> None:
        """Fill a rectangular wall column from y=1 to *ceil_y - 1*."""
        chunk[x0:x1, 1:ceil_y, z0:z1, 0] = 1.0
        chunk[x0:x1, 1:ceil_y, z0:z1, 1] = _WALL

    def _place_interior(
        self,
        chunk: NDArray[np.float32],
        rng: np.random.Generator,
        ceil_y: int,
        cx: int,
        cz: int,
    ) -> None:
        """Add divider walls, support pillars, and blocker obstacles."""
        c = self._chunk_size
        wt = self._wall_thickness
        inner_lo = wt + 1
        inner_hi = c - wt - 1

        if inner_hi - inner_lo < 4:
            return  # chunk too small for interior features

        n_dividers = int(rng.integers(1, 3))
        for _ in range(n_dividers):
            axis = int(rng.integers(0, 2))  # 0 = X-aligned, 1 = Z-aligned
            pos = int(rng.integers(inner_lo + 2, inner_hi - 2))
            length = int(rng.integers(3, max(4, (inner_hi - inner_lo) // 2 + 1)))
            start = int(rng.integers(inner_lo, max(inner_lo + 1, inner_hi - length)))

            # Leave a gap (doorway) in the divider
            gap_start = start + length // 3
            gap_end = gap_start + max(2, length // 4)

            for i in range(start, min(start + length, inner_hi)):
                if gap_start <= i < gap_end:
                    continue  # doorway gap
                for ly in range(1, ceil_y):
                    if axis == 0:
                        if 0 <= pos < c and 0 <= i < c:
                            chunk[pos, ly, i, 0] = 1.0
                            chunk[pos, ly, i, 1] = _WALL
                    else:
                        if 0 <= i < c and 0 <= pos < c:
                            chunk[i, ly, pos, 0] = 1.0
                            chunk[i, ly, pos, 1] = _WALL

        # Pillars (2x2 columns)
        n_pillars = int(rng.integers(0, 3))
        for _ in range(n_pillars):
            px = int(rng.integers(inner_lo + 1, inner_hi - 1))
            pz = int(rng.integers(inner_lo + 1, inner_hi - 1))
            for dx in range(2):
                for dz in range(2):
                    xx, zz = px + dx, pz + dz
                    if 0 <= xx < c and 0 <= zz < c:
                        for ly in range(1, ceil_y):
                            chunk[xx, ly, zz, 0] = 1.0
                            chunk[xx, ly, zz, 1] = _WALL

        # Blockers (navigation obstacles) — compact, clearly visible masses.
        n_blockers = int(rng.integers(1, 4))
        for _ in range(n_blockers):
            bw = int(rng.integers(2, 4))
            bd = int(rng.integers(2, 4))
            bh = int(rng.integers(2, min(6, ceil_y)))
            bx = int(rng.integers(inner_lo, max(inner_lo + 1, inner_hi - bw)))
            bz = int(rng.integers(inner_lo, max(inner_lo + 1, inner_hi - bd)))

            # Keep spawn area in origin chunk clear.
            if cx == 0 and cz == 0:
                center = self._chunk_size // 2
                if bx <= center < bx + bw and bz <= center < bz + bd:
                    continue

            x1 = min(bx + bw, c)
            z1 = min(bz + bd, c)
            y1 = min(1 + bh, ceil_y)
            chunk[bx:x1, 1:y1, bz:z1, 0] = 1.0
            chunk[bx:x1, 1:y1, bz:z1, 1] = _OBSTACLE

    def _carve_doorways(
        self,
        chunk: NDArray[np.float32],
        cx: int,
        cz: int,
        ceil_y: int,
    ) -> None:
        """Carve doorways in the perimeter walls.

        Each of the 4 faces (-X, +X, -Z, +Z) gets a doorway with a
        deterministic probability based on the pair of adjacent chunk
        coordinates so both sides agree on whether there is a door.
        """
        c = self._chunk_size
        wt = self._wall_thickness
        door_h = min(4, ceil_y - 1)  # door is 4 voxels tall (or less)

        faces: list[tuple[int, int, int, int, int, int, int]] = [
            (0, wt, 0, c, cx - 1, cz, 0),      # -X face
            (c - wt, c, 0, c, cx + 1, cz, 1),   # +X face
            (0, c, 0, wt, cx, cz - 1, 2),        # -Z face
            (0, c, c - wt, c, cx, cz + 1, 3),    # +Z face
        ]

        for x0, x1, z0, z1, ncx, ncz, face_id in faces:
            # Deterministic: both this chunk and neighbour must agree.
            # Use the sorted pair so both sides compute the same hash.
            pair_key = tuple(sorted(((cx, cz, face_id), (ncx, ncz, face_id ^ 1))))
            pair_hash = hash((self._seed, pair_key)) & 0xFFFFFFFF
            pair_rng = np.random.default_rng(pair_hash)

            # Moderate doorway density to preserve strong barrier perception.
            if pair_rng.random() > self._doorway_probability:
                continue

            # Compute doorway position along the face
            if face_id in (0, 1):
                # X-face: doorway spans along Z
                door_width = max(2, min(3, c - 2 * wt - 2))
                door_center = c // 2
                dz0 = max(wt, door_center - door_width // 2)
                dz1 = min(c - wt, dz0 + door_width)
                for lx in range(x0, x1):
                    for lz in range(dz0, dz1):
                        for ly in range(1, 1 + door_h):
                            if ly < c:
                                chunk[lx, ly, lz, 0] = 0.0
                                chunk[lx, ly, lz, 1] = _AIR
            else:
                # Z-face: doorway spans along X
                door_width = max(2, min(3, c - 2 * wt - 2))
                door_center = c // 2
                dx0 = max(wt, door_center - door_width // 2)
                dx1 = min(c - wt, dx0 + door_width)
                for lx in range(dx0, dx1):
                    for lz in range(z0, z1):
                        for ly in range(1, 1 + door_h):
                            if ly < c:
                                chunk[lx, ly, lz, 0] = 0.0
                                chunk[lx, ly, lz, 1] = _AIR
