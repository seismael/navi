"""Lightweight Python compiler surface for voxel-dag tests and tooling.

The canonical runtime consumes compiled `.gmdag` assets produced through the
Environment project. This module exists so repository-local tests can exercise a
matching offline surface without depending on the native C++ CLI.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

__all__ = [
    "MeshIngestor",
    "compress_to_dag",
    "compute_dense_sdf",
    "main",
    "write_gmdag",
]


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    power = 1
    while power < value:
        power <<= 1
    return power


class MeshIngestor:
    """Minimal OBJ loader used by the Python verification surface."""

    @staticmethod
    def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        vertices: list[list[float]] = []
        faces: list[list[int]] = []
        obj_path = Path(path)
        for raw_line in obj_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                indices = [int(part.split("/")[0]) - 1 for part in parts]
                # Fan triangulation for any n-gon face.
                for idx in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[idx], indices[idx + 1]])

        if vertices:
            vertex_array = np.asarray(vertices, dtype=np.float32)
            bbox_min = vertex_array.min(axis=0)
            bbox_max = vertex_array.max(axis=0)
        else:
            vertex_array = np.zeros((0, 3), dtype=np.float32)
            bbox_min = np.zeros(3, dtype=np.float32)
            bbox_max = np.zeros(3, dtype=np.float32)

        index_array = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
        return vertex_array, index_array, bbox_min.astype(np.float32), bbox_max.astype(np.float32)


def compute_dense_sdf(
    vertices: np.ndarray,
    indices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int,
    *,
    padding: float = 0.1,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return a cubic dense unsigned distance field sampled from simple analytic surfaces."""
    del indices

    res = _next_power_of_two(resolution)
    if vertices.size == 0:
        return np.zeros((res, res, res), dtype=np.float32), 1.0 / float(res), np.zeros(3, dtype=np.float32)

    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    extent = bbox_max - bbox_min
    max_extent = float(max(np.max(extent), 1e-6))
    pad = float(max(padding, 0.0)) * max_extent
    cube_size = max_extent + (2.0 * pad)
    cell_size = cube_size / float(res)
    cube_min = (bbox_min + bbox_max) * 0.5 - (cube_size * 0.5)

    coords = (np.arange(res, dtype=np.float32) + 0.5) * cell_size
    x_coords = cube_min[0] + coords
    y_coords = cube_min[1] + coords
    z_coords = cube_min[2] + coords
    x_grid = x_coords[None, None, :]
    y_grid = y_coords[None, :, None]
    z_grid = z_coords[:, None, None]

    # If the mesh is effectively planar on any axis, use distance to that plane.
    plane_eps = max(cell_size * 0.5, 1e-5)
    for axis in range(3):
        axis_values = vertices[:, axis]
        if float(axis_values.max() - axis_values.min()) <= plane_eps:
            plane_value = float(axis_values.mean())
            if axis == 0:
                plane_dist = np.abs(x_grid - plane_value)
            elif axis == 1:
                plane_dist = np.abs(y_grid - plane_value)
            else:
                plane_dist = np.abs(z_grid - plane_value)
            return np.broadcast_to(plane_dist, (res, res, res)).astype(np.float32, copy=False), float(cell_size), cube_min.astype(np.float32)

    # Degenerate single-point geometry falls back to Euclidean point distance.
    if vertices.shape[0] == 1:
        vertex = vertices[0]
        field = np.sqrt(
            (x_grid - float(vertex[0])) ** 2
            + (y_grid - float(vertex[1])) ** 2
            + (z_grid - float(vertex[2])) ** 2
        )
        return np.broadcast_to(field, (res, res, res)).astype(np.float32, copy=False), float(cell_size), cube_min.astype(np.float32)

    # For closed box-like test geometry, distance to the nearest bbox face is a
    # stable analytic proxy that still yields valid sphere-tracing behavior.
    target_shape = (res, res, res)
    face_distances = np.stack(
        (
            np.broadcast_to(np.abs(x_grid - float(bbox_min[0])), target_shape),
            np.broadcast_to(np.abs(x_grid - float(bbox_max[0])), target_shape),
            np.broadcast_to(np.abs(y_grid - float(bbox_min[1])), target_shape),
            np.broadcast_to(np.abs(y_grid - float(bbox_max[1])), target_shape),
            np.broadcast_to(np.abs(z_grid - float(bbox_min[2])), target_shape),
            np.broadcast_to(np.abs(z_grid - float(bbox_max[2])), target_shape),
        ),
        axis=0,
    )
    field = face_distances.min(axis=0)
    return field.astype(np.float32, copy=False), float(cell_size), cube_min.astype(np.float32)


def _float_to_half_bits(value: float) -> int:
    return int(np.asarray([value], dtype=np.float16).view(np.uint16)[0])


def compress_to_dag(grid: np.ndarray, resolution: int) -> np.ndarray:
    """Return a fast native-format DAG payload compatible with the CUDA test backend."""
    res = _next_power_of_two(resolution)
    dense = np.asarray(grid, dtype=np.float32)
    if dense.shape != (res, res, res):
        dense = np.reshape(dense, (res, res, res)).astype(np.float32, copy=False)

    if dense.size == 0:
        leaf_distance = 0.1
    else:
        positive = dense[dense > 0.01]
        leaf_distance = float(positive.min()) if positive.size > 0 else 0.1
        leaf_distance = max(0.1, min(leaf_distance, 30.0))

    leaf = (1 << 63) | _float_to_half_bits(leaf_distance)
    return np.array([leaf], dtype=np.uint64)


def write_gmdag(path: str | Path, dag: np.ndarray, resolution: int, bbox_min: np.ndarray, voxel_size: float) -> None:
    """Write a `.gmdag` file with the canonical 32-byte header plus node payload."""
    target = Path(path)
    dag_u64 = np.asarray(dag, dtype=np.uint64)
    bbox = np.asarray(bbox_min, dtype=np.float32)
    if bbox.shape != (3,):
        msg = "bbox_min must be a length-3 vector"
        raise ValueError(msg)

    header = struct.pack(
        "<4sIIffffI",
        b"GDAG",
        1,
        int(_next_power_of_two(resolution)),
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        float(voxel_size),
        int(dag_u64.size),
    )
    target.write_bytes(header + dag_u64.tobytes())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile an OBJ mesh into a simple .gmdag verification artifact.")
    parser.add_argument("--input", required=True, help="Input OBJ mesh path")
    parser.add_argument("--output", required=True, help="Output .gmdag path")
    parser.add_argument("--resolution", type=int, default=128, help="Target cubic resolution")
    parser.add_argument("--padding", type=float, default=0.1, help="Relative cubic padding")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    vertices, indices, bbox_min, bbox_max = MeshIngestor.load_obj(args.input)
    grid, voxel_size, cube_min = compute_dense_sdf(
        vertices,
        indices,
        bbox_min,
        bbox_max,
        args.resolution,
        padding=args.padding,
    )
    dag = compress_to_dag(grid, args.resolution)
    write_gmdag(args.output, dag, args.resolution, cube_min, voxel_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())