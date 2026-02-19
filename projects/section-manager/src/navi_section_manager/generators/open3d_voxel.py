"""Open3DVoxelGenerator — Open3D-native cave-like voxel world generator."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

from navi_section_manager.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["Open3DVoxelGenerator"]

_AIR: float = 0.0
_WALL: float = 1.0
_FLOOR: float = 2.0


def _load_open3d() -> object:
    try:
        return import_module("open3d")
    except ModuleNotFoundError as exc:
        msg = "Open3D is required for Open3DVoxelGenerator. Install with: uv sync --extra open3d"
        raise RuntimeError(msg) from exc


class Open3DVoxelGenerator(AbstractWorldGenerator):
    """Generate chunked voxel data from Open3D point-cloud voxelization.

    The generator builds a deterministic local point cloud per chunk,
    voxelizes it via Open3D, and then carves navigable corridors for
    drone-like movement. Output shape is ``(C, C, C, 2)`` with
    ``[density, semantic_id]`` channels.
    """

    __slots__ = ("_chunk_size", "_corridor_steps", "_points_per_chunk", "_seed")

    def __init__(
        self,
        seed: int = 42,
        chunk_size: int = 16,
        points_per_chunk: int = 600,
        corridor_steps: int = 128,
    ) -> None:
        self._seed = seed
        self._chunk_size = chunk_size
        self._points_per_chunk = max(64, points_per_chunk)
        self._corridor_steps = max(16, corridor_steps)

    def generate_chunk(self, cx: int, cy: int, cz: int) -> NDArray[np.float32]:
        """Generate one chunk of Open3D-derived voxel occupancy."""
        c = self._chunk_size
        chunk = np.zeros((c, c, c, 2), dtype=np.float32)

        if cy < 0:
            chunk[:, :, :, 0] = 1.0
            chunk[:, :, :, 1] = _WALL
            return chunk

        if cy > 1:
            return chunk

        rng = self._rng_for_chunk(cx, cy, cz)
        o3d = _load_open3d()

        points = rng.uniform(low=0.0, high=float(c), size=(self._points_per_chunk, 3))
        # Bias obstacles toward walls/ceiling to keep a generally navigable interior.
        points[:, 1] = np.clip(
            points[:, 1] + rng.normal(loc=1.5, scale=2.0, size=(self._points_per_chunk,)),
            0.0,
            c - 1,
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            pcd,
            voxel_size=1.0,
            min_bound=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            max_bound=np.array([float(c), float(c), float(c)], dtype=np.float64),
        )

        for voxel in voxel_grid.get_voxels():
            ix, iy, iz = voxel.grid_index
            if 0 <= ix < c and 0 <= iy < c and 0 <= iz < c:
                chunk[ix, iy, iz, 0] = 1.0
                chunk[ix, iy, iz, 1] = _WALL

        self._add_floor(chunk)
        self._carve_corridor(chunk, rng)
        return chunk

    def spawn_position(self) -> tuple[float, float, float]:
        """Return spawn in open central air corridor."""
        c = float(self._chunk_size)
        return (c / 2.0, 1.8, c / 2.0)

    def _rng_for_chunk(self, cx: int, cy: int, cz: int) -> np.random.Generator:
        chunk_seed = hash((self._seed, cx, cy, cz)) & 0xFFFFFFFF
        return np.random.default_rng(chunk_seed)

    def _add_floor(self, chunk: NDArray[np.float32]) -> None:
        chunk[:, 0, :, 0] = 1.0
        chunk[:, 0, :, 1] = _FLOOR

    def _carve_corridor(self, chunk: NDArray[np.float32], rng: np.random.Generator) -> None:
        c = self._chunk_size
        pos = np.array([c // 2, 2, c // 2], dtype=np.int64)
        directions = np.array(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 0, 1],
                [0, 0, -1],
                [0, 1, 0],
                [0, -1, 0],
            ],
            dtype=np.int64,
        )

        for _ in range(self._corridor_steps):
            step = directions[int(rng.integers(0, len(directions)))]
            pos = np.clip(pos + step, [1, 1, 1], [c - 2, c - 2, c - 2])
            x0, y0, z0 = int(pos[0]), int(pos[1]), int(pos[2])
            for ix in range(max(0, x0 - 1), min(c, x0 + 2)):
                for iy in range(max(1, y0 - 1), min(c, y0 + 2)):
                    for iz in range(max(0, z0 - 1), min(c, z0 + 2)):
                        chunk[ix, iy, iz, 0] = _AIR
                        chunk[ix, iy, iz, 1] = _AIR
