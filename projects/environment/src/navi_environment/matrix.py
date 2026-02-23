"""SparseVoxelGrid — chunked 3D spatial hash map for voxel storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["SparseVoxelGrid"]


class SparseVoxelGrid:
    """Sparse 3D voxel grid stored as a spatial hash of chunks.

    Each chunk is a dense ``(C, C, C, 2)`` array holding
    ``(density, semantic_id)`` per voxel.  Chunk coordinates are integer
    tuples ``(cx, cy, cz)`` addressing the spatial hash.

    **No color, no reflectivity** — purely geometric + semantic.
    """

    __slots__ = ("_chunk_size", "_chunks")

    def __init__(self, chunk_size: int = 16) -> None:
        self._chunk_size = chunk_size
        self._chunks: dict[tuple[int, int, int], NDArray[np.float32]] = {}

    @property
    def chunk_size(self) -> int:
        """Return the side length of each cubic chunk."""
        return self._chunk_size

    @property
    def occupied_chunks(self) -> set[tuple[int, int, int]]:
        """Return the set of chunk coordinates currently in memory."""
        return set(self._chunks.keys())

    # ------------------------------------------------------------------
    # Chunk-level access
    # ------------------------------------------------------------------

    def get_chunk(self, cx: int, cy: int, cz: int) -> NDArray[np.float32] | None:
        """Return the chunk at ``(cx, cy, cz)`` or ``None``."""
        return self._chunks.get((cx, cy, cz))

    def set_chunk(self, cx: int, cy: int, cz: int, data: NDArray[np.float32]) -> None:
        """Store a ``(C, C, C, 2)`` chunk at ``(cx, cy, cz)``."""
        expected = (self._chunk_size, self._chunk_size, self._chunk_size, 2)
        if data.shape != expected:
            msg = f"Chunk shape must be {expected}, got {data.shape}"
            raise ValueError(msg)
        self._chunks[(cx, cy, cz)] = data

    def remove_chunk(self, cx: int, cy: int, cz: int) -> None:
        """Remove a chunk from the grid (the 'Dump')."""
        self._chunks.pop((cx, cy, cz), None)

    # ------------------------------------------------------------------
    # Region queries
    # ------------------------------------------------------------------

    def query_region(
        self,
        min_corner: tuple[int, int, int],
        max_corner: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Return packed ``(N, 5)`` voxels ``[x, y, z, density, semantic_id]`` in a box.

        Coordinates *min_corner* and *max_corner* are in **chunk** space.
        Only non-zero-density voxels are returned.
        """
        rows: list[NDArray[np.float32]] = []
        cs = self._chunk_size
        for cx in range(min_corner[0], max_corner[0] + 1):
            for cy in range(min_corner[1], max_corner[1] + 1):
                for cz in range(min_corner[2], max_corner[2] + 1):
                    chunk = self._chunks.get((cx, cy, cz))
                    if chunk is None:
                        continue
                    # Build world-space coordinates for every non-air voxel
                    density = chunk[..., 0]
                    mask = density > 0.0
                    if not np.any(mask):
                        continue
                    indices = np.argwhere(mask)  # (M, 3) — local ix, iy, iz
                    world_x = (cx * cs + indices[:, 0]).astype(np.float32)
                    world_y = (cy * cs + indices[:, 1]).astype(np.float32)
                    world_z = (cz * cs + indices[:, 2]).astype(np.float32)
                    vals = chunk[mask]  # (M, 2)
                    packed = np.column_stack([world_x, world_y, world_z, vals])
                    rows.append(packed)
        if not rows:
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate(rows, axis=0).astype(np.float32)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks currently stored."""
        return len(self._chunks)
