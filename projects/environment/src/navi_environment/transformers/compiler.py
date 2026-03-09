"""Offline model -> sparse voxel world compiler."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__: list[str] = [
    "WorldCompileConfig",
    "WorldCompileResult",
    "WorldModelCompiler",
]


@dataclass(frozen=True)
class WorldCompileConfig:
    """Compile settings for source-to-sparse-voxel conversion."""

    chunk_size: int = 16
    voxel_size: float = 1.0
    semantic_id: int = 6
    source_format: str = "auto"  # auto, ply, obj, stl


@dataclass(frozen=True)
class WorldCompileResult:
    """Compiler output summary."""

    source_path: str
    output_path: str
    occupied_voxels: int
    chunk_count: int
    spawn_position: tuple[float, float, float]


class WorldModelCompiler:
    """Compiles PLY/OBJ/STL assets into sparse chunked Zarr world stores."""

    def compile(
        self,
        source_path: str | Path,
        output_path: str | Path,
        config: WorldCompileConfig,
    ) -> WorldCompileResult:
        """Compile a source model into canonical sparse chunk world store."""
        src = Path(source_path)
        out = Path(output_path)

        source_format = self._resolve_source_format(src, config.source_format)
        if source_format == "ply":
            points = self._load_ascii_ply_vertices(src)
        elif source_format == "obj":
            points = self._load_obj_vertices(src)
        elif source_format == "stl":
            points = self._load_ascii_stl_vertices(src)
        else:
            msg = f"Unsupported source format: {source_format}"
            raise ValueError(msg)

        voxel_coords, world_offset = self._voxelize(points, config.voxel_size)
        sparse_chunks = self._build_sparse_chunks(
            voxel_coords=voxel_coords,
            chunk_size=config.chunk_size,
            semantic_id=config.semantic_id,
        )
        spawn = self._compute_spawn(voxel_coords)

        self._write_sparse_zarr(
            output_path=out,
            sparse_chunks=sparse_chunks,
            config=config,
            spawn_position=spawn,
            world_offset=world_offset,
        )

        return WorldCompileResult(
            source_path=str(src),
            output_path=str(out),
            occupied_voxels=int(voxel_coords.shape[0]),
            chunk_count=len(sparse_chunks),
            spawn_position=spawn,
        )

    def _resolve_source_format(self, source_path: Path, requested: str) -> str:
        """Resolve source format from user option and filename extension."""
        requested_norm = requested.strip().lower()
        if requested_norm in {"ply", "obj", "stl"}:
            return requested_norm
        if requested_norm != "auto":
            msg = f"Unknown source format option: {requested}"
            raise ValueError(msg)

        ext = source_path.suffix.lower()
        if ext == ".ply":
            return "ply"
        if ext == ".obj":
            return "obj"
        if ext == ".stl":
            return "stl"
        msg = f"Cannot infer format from extension '{ext}'. Set --source-format explicitly."
        raise ValueError(msg)

    def _load_ascii_ply_vertices(self, path: Path) -> np.ndarray:
        """Load xyz vertices from an ASCII PLY file."""
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "ply":
            msg = f"Not a PLY file: {path}"
            raise ValueError(msg)

        if len(lines) < 3 or lines[1].strip() != "format ascii 1.0":
            msg = "Only ASCII PLY format is supported in v1"
            raise ValueError(msg)

        vertex_count = 0
        properties: list[str] = []
        in_vertex_block = False
        header_end = -1

        for idx, raw in enumerate(lines[2:], start=2):
            line = raw.strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex_block = len(parts) == 3 and parts[1] == "vertex"
                if in_vertex_block:
                    vertex_count = int(parts[2])
                    properties = []
            elif line.startswith("property ") and in_vertex_block:
                parts = line.split()
                if len(parts) >= 3:
                    properties.append(parts[-1])
            elif line == "end_header":
                header_end = idx
                break

        if header_end < 0 or vertex_count <= 0:
            msg = "Invalid PLY header: missing vertex section"
            raise ValueError(msg)

        try:
            x_idx = properties.index("x")
            y_idx = properties.index("y")
            z_idx = properties.index("z")
        except ValueError as exc:
            msg = "PLY vertex properties must include x, y, z"
            raise ValueError(msg) from exc

        start = header_end + 1
        end = start + vertex_count
        if end > len(lines):
            msg = "PLY file ended before all vertex rows were read"
            raise ValueError(msg)

        pts = np.zeros((vertex_count, 3), dtype=np.float32)
        for i, row in enumerate(lines[start:end]):
            parts = row.strip().split()
            if len(parts) < len(properties):
                msg = f"Invalid vertex row at index {i}: '{row}'"
                raise ValueError(msg)
            pts[i, 0] = float(parts[x_idx])
            pts[i, 1] = float(parts[y_idx])
            pts[i, 2] = float(parts[z_idx])

        return pts

    def _load_obj_vertices(self, path: Path) -> np.ndarray:
        """Load xyz vertices from an OBJ file."""
        vertices: list[tuple[float, float, float]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

        if not vertices:
            msg = f"OBJ file has no vertex records: {path}"
            raise ValueError(msg)
        return np.asarray(vertices, dtype=np.float32)

    def _load_ascii_stl_vertices(self, path: Path) -> np.ndarray:
        """Load xyz vertices from an ASCII STL file."""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        vertices: list[tuple[float, float, float]] = []
        for raw in lines:
            line = raw.strip()
            if not line.lower().startswith("vertex "):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

        if not vertices:
            msg = f"ASCII STL file has no vertex records: {path}"
            raise ValueError(msg)
        return np.asarray(vertices, dtype=np.float32)

    def _voxelize(self, points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Voxelize points and normalize coordinates to non-negative space."""
        if voxel_size <= 0.0:
            msg = "voxel_size must be > 0"
            raise ValueError(msg)

        coords = np.floor(points / voxel_size).astype(np.int32)
        unique_coords = np.unique(coords, axis=0)
        min_coords = np.min(unique_coords, axis=0)
        normalized = unique_coords - min_coords
        return normalized, min_coords

    def _build_sparse_chunks(
        self,
        voxel_coords: np.ndarray,
        chunk_size: int,
        semantic_id: int,
    ) -> dict[tuple[int, int, int], np.ndarray]:
        """Pack voxel coordinates into sparse chunk tensors."""
        if chunk_size <= 0:
            msg = "chunk_size must be > 0"
            raise ValueError(msg)

        chunks: dict[tuple[int, int, int], np.ndarray] = {}
        for row in voxel_coords:
            vx, vy, vz = int(row[0]), int(row[1]), int(row[2])
            cx, lx = divmod(vx, chunk_size)
            cy, ly = divmod(vy, chunk_size)
            cz, lz = divmod(vz, chunk_size)
            key = (cx, cy, cz)
            if key not in chunks:
                chunks[key] = np.zeros((chunk_size, chunk_size, chunk_size, 2), dtype=np.float32)

            chunk = chunks[key]
            chunk[lx, ly, lz, 0] = 1.0
            chunk[lx, ly, lz, 1] = float(semantic_id)

        return chunks

    def _compute_spawn(self, voxel_coords: np.ndarray) -> tuple[float, float, float]:
        """Compute spawn position as center of occupied voxel bounding box."""
        mins = np.min(voxel_coords, axis=0).astype(np.float32)
        maxs = np.max(voxel_coords, axis=0).astype(np.float32)
        center = (mins + maxs + 1.0) / 2.0
        return float(center[0]), float(center[1]), float(center[2])

    def _write_sparse_zarr(
        self,
        output_path: Path,
        sparse_chunks: dict[tuple[int, int, int], np.ndarray],
        config: WorldCompileConfig,
        spawn_position: tuple[float, float, float],
        world_offset: np.ndarray,
    ) -> None:
        """Write sparse chunks to canonical world zarr layout."""
        import zarr  # type: ignore[import-untyped]

        store = zarr.open_group(str(output_path), mode="w")
        store.attrs["format_version"] = 1
        store.attrs["representation"] = "sparse_chunks"
        store.attrs["chunk_size"] = int(config.chunk_size)
        store.attrs["voxel_size"] = float(config.voxel_size)
        store.attrs["spawn_position"] = [
            float(spawn_position[0]),
            float(spawn_position[1]),
            float(spawn_position[2]),
        ]
        store.attrs["source_offset"] = [
            int(world_offset[0]),
            int(world_offset[1]),
            int(world_offset[2]),
        ]

        keys_sorted = sorted(sparse_chunks.keys())
        index = np.asarray(keys_sorted, dtype=np.int32)
        chunk_index = store.create_array(
            "chunk_index",
            shape=index.shape,
            dtype=np.int32,
            overwrite=True,
        )
        chunk_index[:] = index

        chunks_group = store.create_group("chunks", overwrite=True)
        for cx, cy, cz in keys_sorted:
            key = f"{cx}_{cy}_{cz}"
            data = sparse_chunks[(cx, cy, cz)]
            arr = chunks_group.create_array(
                key,
                shape=data.shape,
                dtype=np.float32,
                overwrite=True,
            )
            arr[:] = data
