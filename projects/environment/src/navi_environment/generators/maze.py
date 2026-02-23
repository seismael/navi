"""MazeGenerator — procedural 3D voxel maze via Recursive Backtracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from navi_environment.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["MazeGenerator"]

# Semantic IDs
_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0
_CEILING: float = 3.0


class MazeGenerator(AbstractWorldGenerator):
    """Generates infinite 3D maze chunks using Recursive Backtracking.

    Each chunk is a ``(C, C, C, 2)`` array with ``[density, semantic_id]``.
    The maze is a 2D layout extruded vertically: floor at y=0, walls from
    y=1 to y=C-2, ceiling at y=C-1, with carved corridors between walls.

    The maze is deterministic per (seed, cx, cz) — the Y axis is the
    vertical axis, so chunk_y only contributes floor/ceiling layers.
    """

    __slots__ = ("_chunk_size", "_complexity", "_corridor_height", "_seed")

    def __init__(
        self,
        seed: int = 42,
        chunk_size: int = 16,
        complexity: float = 0.5,
        corridor_height: int = 4,
    ) -> None:
        self._seed = seed
        self._chunk_size = chunk_size
        self._complexity = max(0.0, min(1.0, complexity))
        self._corridor_height = min(corridor_height, chunk_size - 2)

    def generate_chunk(
        self,
        cx: int,
        cy: int,
        cz: int,
    ) -> NDArray[np.float32]:
        """Generate a maze chunk at chunk coordinates ``(cx, cy, cz)``.

        The XZ plane defines the maze layout; Y is vertical.

        Returns:
            ``(C, C, C, 2)`` array with ``[density, semantic_id]``.
        """
        c = self._chunk_size
        chunk = np.zeros((c, c, c, 2), dtype=np.float32)

        # Determine which vertical layers this chunk covers
        # Ground level chunks (cy == 0) get the maze pattern
        # Above/below get solid fill or air
        if cy < 0:
            # Underground: solid rock
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        if cy > 0:
            # Sky: empty air
            return chunk

        # cy == 0: maze floor level
        maze_grid = self._generate_2d_maze(cx, cz)
        grid_dim = maze_grid.shape[0]

        for lx in range(c):
            for lz in range(c):
                # Map local voxel (lx, lz) to maze grid cell
                gi = lx * grid_dim // c
                gj = lz * grid_dim // c
                gi = min(gi, grid_dim - 1)
                gj = min(gj, grid_dim - 1)

                is_wall = maze_grid[gi, gj]

                # Floor layer (y=0)
                chunk[lx, 0, lz, 0] = 1.0
                chunk[lx, 0, lz, 1] = _FLOOR

                if is_wall:
                    # Wall column: fill from y=1 to y=c-1
                    for ly in range(1, c):
                        chunk[lx, ly, lz, 0] = 1.0
                        chunk[lx, ly, lz, 1] = _WALL
                else:
                    # Open corridor: air from y=1 to corridor_height
                    # Ceiling at corridor_height + 1
                    ceil_y = min(1 + self._corridor_height, c - 1)
                    chunk[lx, ceil_y, lz, 0] = 1.0
                    chunk[lx, ceil_y, lz, 1] = _CEILING
                    # Fill above ceiling as solid
                    for ly in range(ceil_y + 1, c):
                        chunk[lx, ly, lz, 0] = 1.0
                        chunk[lx, ly, lz, 1] = _WALL

        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Return a spawn point inside the origin chunk's first open cell."""
        maze_grid = self._generate_2d_maze(0, 0)
        grid_dim = maze_grid.shape[0]
        c = self._chunk_size

        # Find first corridor cell (odd indices) and map back to voxel coords
        for gi in range(1, grid_dim, 2):
            for gj in range(1, grid_dim, 2):
                if not maze_grid[gi, gj]:
                    wx = (gi + 0.5) / grid_dim * c
                    wz = (gj + 0.5) / grid_dim * c
                    return (wx, 1.5, wz)

        # Fallback: centre of chunk at ground + 1
        return (c / 2.0, 1.5, c / 2.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_2d_maze(self, cx: int, cz: int) -> NDArray[np.bool_]:
        """Carve a 2D maze grid for the given horizontal chunk coords.

        Uses ``maze_cells = chunk_size // 2`` so that ``grid_dim = chunk_size + 1``.
        This ensures the voxel-to-grid mapping hits both wall (even) and
        corridor (odd) indices for proper visual representation.

        Returns:
            ``(grid_dim, grid_dim)`` boolean array — True = wall.
        """
        chunk_seed = hash((self._seed, cx, cz)) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)
        maze_cells = max(self._chunk_size // 2, 2)
        grid_dim = 2 * maze_cells + 1
        grid = np.ones((grid_dim, grid_dim), dtype=np.bool_)

        start_r, start_c = 1, 1
        grid[start_r, start_c] = False

        stack: list[tuple[int, int]] = [(start_r, start_c)]
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]

        max_cells = maze_cells * maze_cells
        target = max(1, int(max_cells * (0.3 + 0.7 * self._complexity)))
        carved = 1

        while stack and carved < target:
            cr, cc = stack[-1]
            order = rng.permutation(len(directions))
            found = False
            for idx in order:
                dr, dc = directions[idx]
                nr, nc = cr + dr, cc + dc
                if 0 < nr < grid_dim and 0 < nc < grid_dim and grid[nr, nc]:
                    grid[cr + dr // 2, cc + dc // 2] = False
                    grid[nr, nc] = False
                    stack.append((nr, nc))
                    carved += 1
                    found = True
                    break
            if not found:
                stack.pop()

        # Border openings for connectivity
        mid = maze_cells
        grid[0, mid] = False
        grid[grid_dim - 1, mid] = False
        grid[mid, grid_dim - 1] = False
        grid[mid, 0] = False

        return grid
