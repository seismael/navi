"""ArenaGenerator — dense arena with pillars, ramps, tunnels, targets, and obstacles."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navi_environment.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["ArenaGenerator"]

# Semantic IDs
_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0
_CEILING: float = 3.0
_PILLAR: float = 4.0
_RAMP: float = 5.0
_OBSTACLE: float = 6.0
_TARGET: float = 10.0


class ArenaGenerator(AbstractWorldGenerator):
    """Dense arena with pillars, ramps, tunnels, platforms, and discoverable targets.

    Produces a rich, varied environment designed for drone exploration RL.
    Each chunk is procedurally generated with high structural density
    and a variety of geometry types:

    - **Pillars** (2x2 or 3x3 columns of varied height)
    - **Block obstacles** (3x3 or 4x4 bases, 2-6 tall)
    - **Ramps / wedges** (inclined surfaces for height transitions)
    - **Tunnels** (enclosed corridors the drone must fly through)
    - **Platforms** (elevated floors at 3-6 voxels high)
    - **L-walls / T-walls** (angular barrier shapes)
    - **Targets** (bright discoverable objects — high reward when found)

    The arena has a raised-floor variant creating terraced areas and
    perimeter walls bounding the navigable area.
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
            return chunk

        # Lay floor at y=0 for cy==0
        if cy == 0:
            chunk[:, 0, :, 0] = 1.0
            chunk[:, 0, :, 1] = _FLOOR

        # Perimeter walls
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

        # Place 2-6 features per chunk (denser than before)
        n_features = int(rng.integers(2, 7))
        for _ in range(n_features):
            kind = int(rng.integers(0, 8))
            fx = int(rng.integers(1, c - 3))
            fz = int(rng.integers(1, c - 3))

            if kind == 0:
                # Tall pillar — 2x2 or 3x3 base, up to 14 voxels tall
                base = int(rng.integers(2, 4))
                height = int(rng.integers(5, min(15, c)))
                if cy == 0:
                    for dx in range(base):
                        for dz in range(base):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(1, min(height, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _PILLAR
                elif cy == 1 and height > c:
                    for dx in range(base):
                        for dz in range(base):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(0, min(height - c, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _PILLAR

            elif kind == 1:
                # Block obstacle — 3x3 or 4x4 base, 2-6 tall
                base = int(rng.integers(3, 5))
                height = int(rng.integers(2, 7))
                if cy == 0:
                    for dx in range(base):
                        for dz in range(base):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(1, min(height + 1, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _OBSTACLE

            elif kind == 2:
                # Ramp — wedge shape along x or z axis
                if cy == 0:
                    ramp_len = int(rng.integers(4, 8))
                    ramp_width = int(rng.integers(2, 4))
                    axis = int(rng.integers(0, 2))
                    for di in range(ramp_len):
                        ramp_height = max(1, (di + 1) * 6 // ramp_len)
                        for dj in range(ramp_width):
                            if axis == 0:
                                px, pz = fx + di, fz + dj
                            else:
                                px, pz = fx + dj, fz + di
                            if 0 <= px < c and 0 <= pz < c:
                                for ly in range(1, min(ramp_height + 1, c)):
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _RAMP

            elif kind == 3:
                # Tunnel — enclosed corridor the drone must fly through
                if cy == 0:
                    tunnel_len = int(rng.integers(5, min(12, c - 2)))
                    tunnel_h = int(rng.integers(3, 6))
                    tunnel_w = int(rng.integers(2, 4))
                    axis = int(rng.integers(0, 2))
                    for di in range(tunnel_len):
                        for dj in range(tunnel_w + 2):
                            if axis == 0:
                                px, pz = fx + di, fz + dj
                            else:
                                px, pz = fx + dj, fz + di
                            if not (0 <= px < c and 0 <= pz < c):
                                continue
                            is_wall = dj == 0 or dj == tunnel_w + 1
                            for ly in range(1, min(tunnel_h + 2, c)):
                                is_ceiling = ly == tunnel_h + 1
                                if is_wall or is_ceiling:
                                    chunk[px, ly, pz, 0] = 1.0
                                    chunk[px, ly, pz, 1] = _WALL if not is_ceiling else _CEILING

            elif kind == 4:
                # Platform — elevated floor at 3-6 voxels high
                if cy == 0:
                    plat_w = int(rng.integers(3, 7))
                    plat_d = int(rng.integers(3, 7))
                    plat_h = int(rng.integers(3, 7))
                    for dx in range(plat_w):
                        for dz in range(plat_d):
                            px, pz = fx + dx, fz + dz
                            if 0 <= px < c and 0 <= pz < c:
                                # Support columns
                                for ly in range(1, min(plat_h, c)):
                                    if dx == 0 or dx == plat_w - 1 or dz == 0 or dz == plat_d - 1:
                                        chunk[px, ly, pz, 0] = 1.0
                                        chunk[px, ly, pz, 1] = _PILLAR
                                # Platform surface
                                if plat_h < c:
                                    chunk[px, plat_h, pz, 0] = 1.0
                                    chunk[px, plat_h, pz, 1] = _FLOOR

            elif kind == 5:
                # L-wall — angular barrier
                if cy == 0:
                    arm_a = int(rng.integers(3, 7))
                    arm_b = int(rng.integers(3, 7))
                    wall_h = int(rng.integers(3, 8))
                    for di in range(arm_a):
                        px, pz = fx + di, fz
                        if 0 <= px < c and 0 <= pz < c:
                            for ly in range(1, min(wall_h, c)):
                                chunk[px, ly, pz, 0] = 1.0
                                chunk[px, ly, pz, 1] = _WALL
                    for dj in range(arm_b):
                        px, pz = fx, fz + dj
                        if 0 <= px < c and 0 <= pz < c:
                            for ly in range(1, min(wall_h, c)):
                                chunk[px, ly, pz, 0] = 1.0
                                chunk[px, ly, pz, 1] = _WALL

            elif kind == 6:
                # T-wall — T-shaped barrier
                if cy == 0:
                    stem_len = int(rng.integers(3, 6))
                    cross_len = int(rng.integers(4, 8))
                    wall_h = int(rng.integers(3, 7))
                    # Stem
                    for di in range(stem_len):
                        px, pz = fx + cross_len // 2, fz + di
                        if 0 <= px < c and 0 <= pz < c:
                            for ly in range(1, min(wall_h, c)):
                                chunk[px, ly, pz, 0] = 1.0
                                chunk[px, ly, pz, 1] = _WALL
                    # Cross
                    for di in range(cross_len):
                        px, pz = fx + di, fz + stem_len - 1
                        if 0 <= px < c and 0 <= pz < c:
                            for ly in range(1, min(wall_h, c)):
                                chunk[px, ly, pz, 0] = 1.0
                                chunk[px, ly, pz, 1] = _WALL

            else:
                # Target beacon — small bright discoverable object
                if cy == 0:
                    tx, tz = fx + 1, fz + 1
                    if 0 <= tx < c and 0 <= tz < c:
                        # Place target at various heights for 3D exploration
                        target_h = int(rng.integers(1, 5))
                        chunk[tx, target_h, tz, 0] = 1.0
                        chunk[tx, target_h, tz, 1] = _TARGET
                        # Small pedestal under target if elevated
                        if target_h > 1:
                            for ly in range(1, target_h):
                                chunk[tx, ly, tz, 0] = 1.0
                                chunk[tx, ly, tz, 1] = _PILLAR

        # Terraced floor variant — some chunks get raised floor sections
        if cy == 0 and rng.random() < 0.3 and not (cx == 0 and cz == 0):
            terrace_h = int(rng.integers(1, 3))
            tx0 = int(rng.integers(0, c // 2))
            tz0 = int(rng.integers(0, c // 2))
            tx1 = int(rng.integers(c // 2, c))
            tz1 = int(rng.integers(c // 2, c))
            for lx in range(tx0, tx1):
                for lz in range(tz0, tz1):
                    if chunk[lx, 1, lz, 0] == 0.0:
                        for ly in range(1, terrace_h + 1):
                            if ly < c:
                                chunk[lx, ly, lz, 0] = 1.0
                                chunk[lx, ly, lz, 1] = _FLOOR

        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Spawn in the centre of the origin chunk, above the floor."""
        half = self._chunk_size / 2.0
        return (half, 1.5, half)
