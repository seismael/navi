"""SlidingWindow — bit-shift + gap fill on robot movement."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from navi_environment.matrix import SparseVoxelGrid

__all__: list[str] = ["SlidingWindow", "WedgeResult"]

_GenerateChunkFn = Callable[[int, int, int], NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class WedgeResult:
    """Result of a window shift — the new wedge of voxels and cull count."""

    new_voxels: NDArray[np.float32]  # (N, 5) — [x, y, z, density, semantic_id]
    culled_count: int  # chunks dropped from the trailing edge


class SlidingWindow:
    """A 3D window centered on the robot that slides with movement.

    When the robot moves, the window computes which chunks enter (the leading
    wedge) and which chunks leave (the trailing dump).  Entering chunks are
    requested from a generator callback; exiting chunks are removed from the
    grid.  The result is the packed new voxels ready to be transformed into
    the current observation contract.
    """

    __slots__ = ("_center_chunk", "_grid", "_initialized", "_radius")

    def __init__(self, grid: SparseVoxelGrid, radius: int = 3) -> None:
        self._grid = grid
        self._radius = radius  # radius in chunks
        self._center_chunk: tuple[int, int, int] = (0, 0, 0)
        self._initialized: bool = False

    @property
    def center_chunk(self) -> tuple[int, int, int]:
        """Return the current centre chunk coordinate."""
        return self._center_chunk

    @property
    def radius(self) -> int:
        """Return the window radius in chunks."""
        return self._radius

    def world_to_chunk(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """Convert a world-space position to chunk coordinates."""
        cs = self._grid.chunk_size
        return (
            int(math.floor(x / cs)),
            int(math.floor(y / cs)),
            int(math.floor(z / cs)),
        )

    def _window_set(self, center: tuple[int, int, int]) -> set[tuple[int, int, int]]:
        """Return the set of chunk coords within radius of *center*."""
        cx, cy, cz = center
        r = self._radius
        chunks: set[tuple[int, int, int]] = set()
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    chunks.add((cx + dx, cy + dy, cz + dz))
        return chunks

    def shift(
        self,
        new_world_x: float,
        new_world_y: float,
        new_world_z: float,
        generate_chunk: _GenerateChunkFn,
    ) -> WedgeResult:
        """Slide the window to a new robot position.

        Args:
            new_world_x: Robot X in world space.
            new_world_y: Robot Y in world space.
            new_world_z: Robot Z in world space.
            generate_chunk: Callback ``(cx, cy, cz) -> NDArray (C,C,C,2)`` that
                produces voxel data for a chunk that is entering the view.

        Returns:
            ``WedgeResult`` with the packed new-wedge voxels and cull count.
        """
        new_center = self.world_to_chunk(new_world_x, new_world_y, new_world_z)

        if not self._initialized:
            # First call: populate the entire window from scratch
            old_set: set[tuple[int, int, int]] = set()
            self._initialized = True
        else:
            old_set = self._window_set(self._center_chunk)

        new_set = self._window_set(new_center)

        entering = new_set - old_set
        exiting = old_set - new_set

        # Cull: remove exiting chunks
        for coord in exiting:
            self._grid.remove_chunk(*coord)
        culled = len(exiting)

        # Fetch: generate entering chunks and add to grid
        new_rows: list[NDArray[np.float32]] = []
        cs = self._grid.chunk_size
        for coord in entering:
            chunk_data = generate_chunk(*coord)
            self._grid.set_chunk(*coord, chunk_data)
            # Extract non-air voxels from this chunk
            density = chunk_data[..., 0]
            mask = density > 0.0
            if np.any(mask):
                indices = np.argwhere(mask)
                wx = (coord[0] * cs + indices[:, 0]).astype(np.float32)
                wy = (coord[1] * cs + indices[:, 1]).astype(np.float32)
                wz = (coord[2] * cs + indices[:, 2]).astype(np.float32)
                vals = chunk_data[mask]  # (M, 2)
                new_rows.append(np.column_stack([wx, wy, wz, vals]))

        self._center_chunk = new_center

        if new_rows:
            new_voxels = np.concatenate(new_rows, axis=0).astype(np.float32)
        else:
            new_voxels = np.empty((0, 5), dtype=np.float32)

        return WedgeResult(new_voxels=new_voxels, culled_count=culled)
