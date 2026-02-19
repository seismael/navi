"""FileGenerator — load voxel chunks from Zarr world stores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from navi_section_manager.generators.base import AbstractWorldGenerator

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__: list[str] = ["FileGenerator"]


class FileGenerator(AbstractWorldGenerator):
    """Loads pre-built voxel worlds from Zarr stores.

    Supports two Zarr layouts:
      - Dense legacy layout with ``voxels`` dataset of shape ``(Wx, Wy, Wz, 2)``.
      - Sparse chunk layout with ``chunk_index`` + ``chunks/<cx>_<cy>_<cz>`` arrays.
    """

    __slots__ = ("_chunk_size", "_dataset", "_path", "_sparse_chunks", "_spawn")

    def __init__(self, path: str | Path, chunk_size: int = 16) -> None:
        self._path = Path(path)
        self._chunk_size = chunk_size
        self._dataset: object | None = None
        self._sparse_chunks: dict[tuple[int, int, int], NDArray[np.float32]] | None = None
        self._spawn: tuple[float, float, float] | None = None

    def generate_chunk(self, cx: int, cy: int, cz: int) -> NDArray[np.float32]:
        """Load a chunk from the file-backed dataset."""
        c = self._chunk_size

        self._ensure_loaded()
        if self._sparse_chunks is not None:
            sparse_chunk = self._sparse_chunks.get((cx, cy, cz))
            if sparse_chunk is None:
                return np.zeros((c, c, c, 2), dtype=np.float32)
            return sparse_chunk

        assert self._dataset is not None
        ds = self._dataset

        x0, y0, z0 = cx * c, cy * c, cz * c
        x1, y1, z1 = x0 + c, y0 + c, z0 + c

        shape = ds.shape  # type: ignore[union-attr]
        if x0 < 0 or y0 < 0 or z0 < 0 or x1 > shape[0] or y1 > shape[1] or z1 > shape[2]:
            return np.zeros((c, c, c, 2), dtype=np.float32)

        return np.asarray(ds[x0:x1, y0:y1, z0:z1], dtype=np.float32)  # type: ignore[index]

    def spawn_position(self) -> tuple[float, float, float]:
        """Return spawn position from store metadata or default center."""
        self._ensure_loaded()
        if self._spawn is not None:
            return self._spawn
        c = self._chunk_size
        return (c / 2.0, c / 2.0, c / 2.0)

    def _ensure_loaded(self) -> object:
        """Lazy-load the dataset on first access."""
        if self._dataset is not None:
            return self._dataset
        if self._sparse_chunks is not None:
            return self._sparse_chunks

        suffix = self._path.suffix.lower()
        if not (self._path.is_dir() or suffix in {".zarr", ".zip"}):
            msg = f"Unsupported file format: {suffix}. Use a .zarr world store."
            raise ValueError(msg)

        try:
            import zarr  # type: ignore[import-untyped]
        except ImportError as exc:
            msg = "zarr package is required for file-backed worlds: pip install zarr"
            raise ImportError(msg) from exc

        store = zarr.open(str(self._path), mode="r")

        if "chunk_index" in store and "chunks" in store:  # type: ignore[operator]
            chunk_index = np.asarray(store["chunk_index"][:], dtype=np.int32)  # type: ignore[index]
            chunks_group = store["chunks"]  # type: ignore[index]
            sparse_chunks: dict[tuple[int, int, int], NDArray[np.float32]] = {}
            for row in chunk_index:
                cx, cy, cz = int(row[0]), int(row[1]), int(row[2])
                key = f"{cx}_{cy}_{cz}"
                sparse_chunks[(cx, cy, cz)] = np.asarray(chunks_group[key][:], dtype=np.float32)  # type: ignore[index]
            self._sparse_chunks = sparse_chunks

            spawn_attr = store.attrs.get("spawn_position")
            if isinstance(spawn_attr, list | tuple) and len(spawn_attr) == 3:
                self._spawn = (
                    float(spawn_attr[0]),
                    float(spawn_attr[1]),
                    float(spawn_attr[2]),
                )

        elif "voxels" in store:  # type: ignore[operator]
            self._dataset = store["voxels"]  # type: ignore[index]

        else:
            msg = "Unsupported Zarr world format. Expected 'voxels' or 'chunk_index'+'chunks'."
            raise ValueError(msg)

        if self._dataset is not None:
            return self._dataset
        assert self._sparse_chunks is not None
        return self._sparse_chunks
