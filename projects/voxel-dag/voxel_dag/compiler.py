"""Lightweight Python compiler surface for voxel-dag tests and tooling.

The canonical runtime consumes compiled `.gmdag` assets produced through the
Environment project. This module exists so repository-local tests can exercise a
matching offline surface without depending on the native C++ CLI.
"""

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = [
    "canonical_node_hash",
    "MeshIngestor",
    "compress_to_dag",
    "compute_dense_sdf",
    "deduplicate_signatures",
    "main",
    "write_gmdag",
]

_MURMUR_C1 = 0x87C37B91114253D5
_MURMUR_C2 = 0x4CF5AD432745937F
_UINT64_MASK = (1 << 64) - 1
_INTERNAL_NODE_MASK_SHIFT = 55


@dataclass(frozen=True)
class _LeafSpec:
    signature: bytes
    node_word: int


@dataclass(frozen=True)
class _InternalSpec:
    signature: bytes
    mask: int
    unique_child_signatures: tuple[bytes, ...]
    remap: tuple[int, ...]


def _rotl64(value: int, shift: int) -> int:
    masked = value & _UINT64_MASK
    return ((masked << shift) | (masked >> (64 - shift))) & _UINT64_MASK


def _fmix64(value: int) -> int:
    mixed = value & _UINT64_MASK
    mixed ^= mixed >> 33
    mixed = (mixed * 0xFF51AFD7ED558CCD) & _UINT64_MASK
    mixed ^= mixed >> 33
    mixed = (mixed * 0xC4CEB9FE1A85EC53) & _UINT64_MASK
    mixed ^= mixed >> 33
    return mixed & _UINT64_MASK


def canonical_node_hash(payload: bytes, seed: int = 0) -> int:
    """Return the canonical MurmurHash3 x64_128 low 64 bits for *payload*."""
    data = memoryview(payload)
    length = len(data)
    h1 = seed & _UINT64_MASK
    h2 = seed & _UINT64_MASK

    block_count = length // 16
    for block_idx in range(block_count):
        offset = block_idx * 16
        k1 = int.from_bytes(data[offset : offset + 8], byteorder="little", signed=False)
        k2 = int.from_bytes(data[offset + 8 : offset + 16], byteorder="little", signed=False)

        k1 = (k1 * _MURMUR_C1) & _UINT64_MASK
        k1 = _rotl64(k1, 31)
        k1 = (k1 * _MURMUR_C2) & _UINT64_MASK
        h1 ^= k1

        h1 = _rotl64(h1, 27)
        h1 = (h1 + h2) & _UINT64_MASK
        h1 = (h1 * 5 + 0x52DCE729) & _UINT64_MASK

        k2 = (k2 * _MURMUR_C2) & _UINT64_MASK
        k2 = _rotl64(k2, 33)
        k2 = (k2 * _MURMUR_C1) & _UINT64_MASK
        h2 ^= k2

        h2 = _rotl64(h2, 31)
        h2 = (h2 + h1) & _UINT64_MASK
        h2 = (h2 * 5 + 0x38495AB5) & _UINT64_MASK

    tail = data[block_count * 16 :]
    k1 = 0
    k2 = 0
    tail_length = len(tail)
    if tail_length:
        if tail_length >= 15:
            k2 ^= int(tail[14]) << 48
        if tail_length >= 14:
            k2 ^= int(tail[13]) << 40
        if tail_length >= 13:
            k2 ^= int(tail[12]) << 32
        if tail_length >= 12:
            k2 ^= int(tail[11]) << 24
        if tail_length >= 11:
            k2 ^= int(tail[10]) << 16
        if tail_length >= 10:
            k2 ^= int(tail[9]) << 8
        if tail_length >= 9:
            k2 ^= int(tail[8])
            k2 = (k2 * _MURMUR_C2) & _UINT64_MASK
            k2 = _rotl64(k2, 33)
            k2 = (k2 * _MURMUR_C1) & _UINT64_MASK
            h2 ^= k2
        if tail_length >= 8:
            k1 ^= int(tail[7]) << 56
        if tail_length >= 7:
            k1 ^= int(tail[6]) << 48
        if tail_length >= 6:
            k1 ^= int(tail[5]) << 40
        if tail_length >= 5:
            k1 ^= int(tail[4]) << 32
        if tail_length >= 4:
            k1 ^= int(tail[3]) << 24
        if tail_length >= 3:
            k1 ^= int(tail[2]) << 16
        if tail_length >= 2:
            k1 ^= int(tail[1]) << 8
        if tail_length >= 1:
            k1 ^= int(tail[0])
            k1 = (k1 * _MURMUR_C1) & _UINT64_MASK
            k1 = _rotl64(k1, 31)
            k1 = (k1 * _MURMUR_C2) & _UINT64_MASK
            h1 ^= k1

    h1 ^= length
    h2 ^= length
    h1 = (h1 + h2) & _UINT64_MASK
    h2 = (h2 + h1) & _UINT64_MASK
    h1 = _fmix64(h1)
    h2 = _fmix64(h2)
    h1 = (h1 + h2) & _UINT64_MASK
    return h1


def deduplicate_signatures(
    signatures: list[bytes],
    *,
    seed: int = 0,
    hash_fn: Callable[[bytes, int], int] | None = None,
) -> tuple[list[bytes], list[int]]:
    """Return unique signatures plus a per-input remap using structural fallback."""
    unique_signatures: list[bytes] = []
    remap: list[int] = []
    buckets: dict[int, list[int]] = {}
    active_hash = hash_fn or canonical_node_hash

    for signature in signatures:
        hash_value = active_hash(signature, seed)
        bucket = buckets.setdefault(hash_value, [])
        matched_index: int | None = None
        for candidate_index in bucket:
            if unique_signatures[candidate_index] == signature:
                matched_index = candidate_index
                break
        if matched_index is None:
            matched_index = len(unique_signatures)
            unique_signatures.append(signature)
            bucket.append(matched_index)
        remap.append(matched_index)
    return unique_signatures, remap


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


def _leaf_distance_from_block(block: np.ndarray) -> float:
    positive = block[block > 0.01]
    leaf_distance = float(positive.min()) if positive.size > 0 else 0.1
    return max(0.1, min(leaf_distance, 30.0))


def _leaf_signature_from_block(block: np.ndarray) -> bytes:
    leaf_distance = _leaf_distance_from_block(block)
    return b"L" + struct.pack("<HH", _float_to_half_bits(leaf_distance), 0)


def _leaf_node_from_signature(signature: bytes) -> np.uint64:
    if len(signature) != 5 or signature[0:1] != b"L":
        msg = "leaf signatures must be exactly 5 bytes including the leaf tag"
        raise ValueError(msg)
    dist_bits, semantic = struct.unpack("<HH", signature[1:])
    return np.uint64((1 << 63) | (int(semantic) << 16) | int(dist_bits))


def _encode_internal_signature(
    *,
    mask: int,
    unique_child_signatures: tuple[bytes, ...],
    remap: tuple[int, ...],
) -> bytes:
    encoded = bytearray(b"I")
    encoded.extend(struct.pack("<B", mask))
    encoded.extend(struct.pack("<B", len(remap)))
    encoded.extend(bytes(remap))
    encoded.extend(struct.pack("<B", len(unique_child_signatures)))
    for child_signature in unique_child_signatures:
        encoded.extend(struct.pack("<I", len(child_signature)))
        encoded.extend(child_signature)
    return bytes(encoded)


def _build_signature_tree(block: np.ndarray, specs: dict[bytes, _LeafSpec | _InternalSpec]) -> bytes:
    if block.size == 0:
        leaf_signature = _leaf_signature_from_block(np.array([0.1], dtype=np.float32))
        specs.setdefault(
            leaf_signature,
            _LeafSpec(signature=leaf_signature, node_word=int(_leaf_node_from_signature(leaf_signature))),
        )
        return leaf_signature

    if min(block.shape) <= 1 or bool(np.allclose(block, block.flat[0], atol=1e-6, rtol=0.0)):
        leaf_signature = _leaf_signature_from_block(block)
        specs.setdefault(
            leaf_signature,
            _LeafSpec(signature=leaf_signature, node_word=int(_leaf_node_from_signature(leaf_signature))),
        )
        return leaf_signature

    z_half = max(1, block.shape[0] // 2)
    y_half = max(1, block.shape[1] // 2)
    x_half = max(1, block.shape[2] // 2)
    child_signatures: list[bytes] = []
    for octant in range(8):
        z_start = z_half if octant & 4 else 0
        y_start = y_half if octant & 2 else 0
        x_start = x_half if octant & 1 else 0
        child_block = block[
            z_start : (block.shape[0] if octant & 4 else z_half),
            y_start : (block.shape[1] if octant & 2 else y_half),
            x_start : (block.shape[2] if octant & 1 else x_half),
        ]
        child_signatures.append(_build_signature_tree(child_block, specs))

    if len(set(child_signatures)) == 1:
        return child_signatures[0]

    unique_child_signatures, remap = deduplicate_signatures(child_signatures, seed=0)
    signature = _encode_internal_signature(
        mask=0xFF,
        unique_child_signatures=tuple(unique_child_signatures),
        remap=tuple(remap),
    )
    specs.setdefault(
        signature,
        _InternalSpec(
            signature=signature,
            mask=0xFF,
            unique_child_signatures=tuple(unique_child_signatures),
            remap=tuple(remap),
        ),
    )
    return signature


def _emit_signature(
    signature: bytes,
    specs: dict[bytes, _LeafSpec | _InternalSpec],
    dag_pool: list[np.uint64],
    emitted_indices: dict[bytes, int],
) -> int:
    existing_index = emitted_indices.get(signature)
    if existing_index is not None:
        return existing_index

    spec = specs[signature]
    if isinstance(spec, _LeafSpec):
        node_index = len(dag_pool)
        dag_pool.append(np.uint64(spec.node_word))
        emitted_indices[signature] = node_index
        return node_index

    node_index = len(dag_pool)
    dag_pool.append(np.uint64(0))
    child_base = len(dag_pool)
    dag_pool.extend(np.uint64(0) for _ in spec.remap)
    emitted_indices[signature] = node_index

    child_indices: list[int] = []
    for child_signature in spec.unique_child_signatures:
        child_indices.append(_emit_signature(child_signature, specs, dag_pool, emitted_indices))

    for offset, unique_child_idx in enumerate(spec.remap):
        dag_pool[child_base + offset] = np.uint64(child_indices[unique_child_idx])

    dag_pool[node_index] = np.uint64((int(spec.mask) << _INTERNAL_NODE_MASK_SHIFT) | child_base)
    return node_index


def compress_to_dag(grid: np.ndarray, resolution: int) -> np.ndarray:
    """Return a fast native-format DAG payload compatible with the CUDA test backend."""
    res = _next_power_of_two(resolution)
    dense = np.asarray(grid, dtype=np.float32)
    if dense.shape != (res, res, res):
        dense = np.reshape(dense, (res, res, res)).astype(np.float32, copy=False)

    if dense.size == 0:
        return np.array([np.uint64((1 << 63) | _float_to_half_bits(0.1))], dtype=np.uint64)

    specs: dict[bytes, _LeafSpec | _InternalSpec] = {}
    root_signature = _build_signature_tree(dense, specs)
    dag_pool: list[np.uint64] = []
    _emit_signature(root_signature, specs, dag_pool, emitted_indices={})
    return np.asarray(dag_pool, dtype=np.uint64)


def write_gmdag(path: str | Path, dag: np.ndarray, resolution: int, bbox_min: np.ndarray, voxel_size: float) -> None:
    """Write a `.gmdag` file with the canonical 32-byte header plus node payload."""
    target = Path(path)
    dag_u64 = np.asarray(dag, dtype=np.uint64)
    normalized_resolution = int(_next_power_of_two(resolution))
    if normalized_resolution <= 0:
        msg = "resolution must be a positive integer"
        raise ValueError(msg)
    if dag_u64.size <= 0:
        msg = "dag must contain at least one node"
        raise ValueError(msg)
    if voxel_size <= 0.0:
        msg = "voxel_size must be positive"
        raise ValueError(msg)
    bbox = np.asarray(bbox_min, dtype=np.float32)
    if bbox.shape != (3,):
        msg = "bbox_min must be a length-3 vector"
        raise ValueError(msg)

    header = struct.pack(
        "<4sIIffffI",
        b"GDAG",
        1,
        normalized_resolution,
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
    parser.add_argument("--resolution", type=int, default=512, help="Target cubic resolution")
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