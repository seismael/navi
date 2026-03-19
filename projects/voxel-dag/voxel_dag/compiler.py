"""Lightweight Python compiler surface for voxel-dag tests and tooling.

The canonical runtime consumes compiled `.gmdag` assets produced through the
Environment project. This module exists so repository-local tests can exercise a
matching offline surface without depending on the native C++ CLI.
"""

from __future__ import annotations

import argparse
import math
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
        k2 = int.from_bytes(
            data[offset + 8 : offset + 16], byteorder="little", signed=False
        )

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
    def load_obj(
        path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        index_array = (
            np.asarray(faces, dtype=np.int32)
            if faces
            else np.zeros((0, 3), dtype=np.int32)
        )
        return (
            vertex_array,
            index_array,
            bbox_min.astype(np.float32),
            bbox_max.astype(np.float32),
        )


def _point_triangle_distance_sq(
    px: float,
    py: float,
    pz: float,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Closest-point distance² from point to triangle (Eberly algorithm)."""
    diff = np.array([px - v0[0], py - v0[1], pz - v0[2]], dtype=np.float64)
    edge0 = (v1 - v0).astype(np.float64)
    edge1 = (v2 - v0).astype(np.float64)
    a00 = float(edge0 @ edge0)
    a01 = float(edge0 @ edge1)
    a11 = float(edge1 @ edge1)
    b0 = float(-diff @ edge0)
    b1 = float(-diff @ edge1)
    det = abs(a00 * a11 - a01 * a01)
    s = a01 * b1 - a11 * b0
    t = a01 * b0 - a00 * b1

    if s + t <= det:
        if s < 0:
            if t < 0:
                if b0 < 0:
                    t = 0.0
                    s = 1.0 if -b0 >= a00 else -b0 / max(a00, 1e-30)
                else:
                    s = 0.0
                    t = (
                        0.0
                        if b1 >= 0
                        else (1.0 if -b1 >= a11 else -b1 / max(a11, 1e-30))
                    )
            else:
                s = 0.0
                t = 0.0 if b1 >= 0 else (1.0 if -b1 >= a11 else -b1 / max(a11, 1e-30))
        elif t < 0:
            t = 0.0
            s = 0.0 if b0 >= 0 else (1.0 if -b0 >= a00 else -b0 / max(a00, 1e-30))
        else:
            inv = 1.0 / max(det, 1e-30)
            s *= inv
            t *= inv
    else:
        if s < 0:
            tmp0 = a01 + b0
            tmp1 = a11 + b1
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a00 - 2.0 * a01 + a11
                s = 1.0 if numer >= denom else numer / max(denom, 1e-30)
                t = 1.0 - s
            else:
                s = 0.0
                t = 0.0 if tmp1 >= 0 else (1.0 if -b1 >= a11 else -b1 / max(a11, 1e-30))
        elif t < 0:
            tmp0 = a01 + b1
            tmp1 = a00 + b0
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a00 - 2.0 * a01 + a11
                t = 1.0 if numer >= denom else numer / max(denom, 1e-30)
                s = 1.0 - t
            else:
                t = 0.0
                s = 0.0 if tmp1 >= 0 else (1.0 if -b0 >= a00 else -b0 / max(a00, 1e-30))
        else:
            numer = a11 + b1 - a01 - b0
            if numer <= 0:
                s = 0.0
                t = 1.0
            else:
                denom = a00 - 2.0 * a01 + a11
                s = 1.0 if numer >= denom else numer / max(denom, 1e-30)
                t = 1.0 - s

    rx = v0[0] + s * edge0[0] + t * edge1[0] - px
    ry = v0[1] + s * edge0[1] + t * edge1[1] - py
    rz = v0[2] + s * edge0[2] + t * edge1[2] - pz
    return float(rx * rx + ry * ry + rz * rz)


def _solve_eikonal(a: float, b: float, c: float, h: float) -> float:
    """1D/2D/3D Eikonal solver for fast-sweeping."""
    vals = sorted([a, b, c])
    a, b, c = vals[0], vals[1], vals[2]
    d = a + h
    if d <= b:
        return d
    d = 0.5 * (a + b + math.sqrt(max(0.0, 2.0 * h * h - (a - b) ** 2)))
    if d <= c:
        return d
    s = a + b + c
    ssq = a * a + b * b + c * c
    disc = s * s - 3.0 * (ssq - h * h)
    if disc >= 0.0:
        return (s + math.sqrt(disc)) / 3.0
    return d


def compute_dense_sdf(
    vertices: np.ndarray,
    indices: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int,
    *,
    padding: float = 0.1,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Return a cubic dense unsigned distance field using triangle-mesh distances + Eikonal propagation."""
    res = _next_power_of_two(resolution)
    if vertices.size == 0:
        return (
            np.zeros((res, res, res), dtype=np.float32),
            1.0 / float(res),
            np.zeros(3, dtype=np.float32),
        )

    bbox_min = np.asarray(bbox_min, dtype=np.float32)
    bbox_max = np.asarray(bbox_max, dtype=np.float32)
    extent = bbox_max - bbox_min
    max_extent = float(max(np.max(extent), 1e-6))
    pad = float(max(padding, 0.0)) * max_extent
    cube_size = max_extent + (2.0 * pad)
    cell_size = cube_size / float(res)
    cube_min = (bbox_min + bbox_max) * 0.5 - (cube_size * 0.5)

    # If the mesh is effectively planar on any axis, use distance to that plane.
    plane_eps = max(cell_size * 0.5, 1e-5)
    for axis in range(3):
        axis_values = vertices[:, axis]
        if float(axis_values.max() - axis_values.min()) <= plane_eps:
            plane_value = float(axis_values.mean())
            coords = (np.arange(res, dtype=np.float32) + 0.5) * cell_size
            if axis == 0:
                plane_dist = np.abs(cube_min[0] + coords - plane_value)
                field = np.broadcast_to(plane_dist[None, None, :], (res, res, res))
            elif axis == 1:
                plane_dist = np.abs(cube_min[1] + coords - plane_value)
                field = np.broadcast_to(plane_dist[None, :, None], (res, res, res))
            else:
                plane_dist = np.abs(cube_min[2] + coords - plane_value)
                field = np.broadcast_to(plane_dist[:, None, None], (res, res, res))
            return (
                field.astype(np.float32, copy=True),
                float(cell_size),
                cube_min.astype(np.float32),
            )

    # Degenerate single-point geometry falls back to Euclidean point distance.
    if vertices.shape[0] == 1:
        vertex = vertices[0]
        coords = (np.arange(res, dtype=np.float32) + 0.5) * cell_size
        x_coords = cube_min[0] + coords
        y_coords = cube_min[1] + coords
        z_coords = cube_min[2] + coords
        field = np.sqrt(
            (x_coords[None, None, :] - float(vertex[0])) ** 2
            + (y_coords[None, :, None] - float(vertex[1])) ** 2
            + (z_coords[:, None, None] - float(vertex[2])) ** 2
        )
        return (
            np.broadcast_to(field, (res, res, res)).astype(np.float32, copy=True),
            float(cell_size),
            cube_min.astype(np.float32),
        )

    # --- General mesh: seed with exact triangle distances, then Eikonal sweep ---
    field = np.full((res, res, res), float("inf"), dtype=np.float64)
    h = float(cell_size)
    indices_arr = np.asarray(indices, dtype=np.int32).reshape(-1, 3)

    # Seed: compute exact closest-point distances for voxels near each triangle
    for tri_row in indices_arr:
        i0, i1, i2 = int(tri_row[0]), int(tri_row[1]), int(tri_row[2])
        if i0 >= len(vertices) or i1 >= len(vertices) or i2 >= len(vertices):
            continue
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        tri_min = np.minimum(np.minimum(v0, v1), v2) - h
        tri_max = np.maximum(np.maximum(v0, v1), v2) + h

        ix_start = max(0, int((tri_min[0] - cube_min[0]) / cell_size))
        ix_end = min(res - 1, int((tri_max[0] - cube_min[0]) / cell_size))
        iy_start = max(0, int((tri_min[1] - cube_min[1]) / cell_size))
        iy_end = min(res - 1, int((tri_max[1] - cube_min[1]) / cell_size))
        iz_start = max(0, int((tri_min[2] - cube_min[2]) / cell_size))
        iz_end = min(res - 1, int((tri_max[2] - cube_min[2]) / cell_size))

        for iz in range(iz_start, iz_end + 1):
            pz = float(cube_min[2]) + iz * cell_size
            for iy in range(iy_start, iy_end + 1):
                py = float(cube_min[1]) + iy * cell_size
                for ix in range(ix_start, ix_end + 1):
                    px = float(cube_min[0]) + ix * cell_size
                    d2 = _point_triangle_distance_sq(px, py, pz, v0, v1, v2)
                    d = math.sqrt(d2)
                    if d < field[iz, iy, ix]:
                        field[iz, iy, ix] = d

    # Eikonal fast-sweeping propagation (8 diagonal passes)
    sweep_dirs = [
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (-1, -1, 1),
        (1, 1, -1),
        (-1, 1, -1),
        (1, -1, -1),
        (-1, -1, -1),
    ]
    for sx, sy, sz in sweep_dirs:
        xr = range(0, res, 1) if sx > 0 else range(res - 1, -1, -1)
        yr = range(0, res, 1) if sy > 0 else range(res - 1, -1, -1)
        zr = range(0, res, 1) if sz > 0 else range(res - 1, -1, -1)
        for iz in zr:
            for iy in yr:
                for ix in xr:
                    ax = field[iz, iy, ix - sx] if 0 <= ix - sx < res else float("inf")
                    ay = field[iz, iy - sy, ix] if 0 <= iy - sy < res else float("inf")
                    az = field[iz - sz, iy, ix] if 0 <= iz - sz < res else float("inf")
                    updated = _solve_eikonal(ax, ay, az, h)
                    if updated < field[iz, iy, ix]:
                        field[iz, iy, ix] = updated

    return (
        field.astype(np.float32, copy=False),
        float(cell_size),
        cube_min.astype(np.float32),
    )


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


def _build_signature_tree(
    block: np.ndarray,
    specs: dict[bytes, _LeafSpec | _InternalSpec],
    cell_size: float = 0.0,
) -> bytes:
    if block.size == 0:
        leaf_signature = _leaf_signature_from_block(np.array([0.1], dtype=np.float32))
        specs.setdefault(
            leaf_signature,
            _LeafSpec(
                signature=leaf_signature,
                node_word=int(_leaf_node_from_signature(leaf_signature)),
            ),
        )
        return leaf_signature

    if min(block.shape) <= 1 or bool(
        np.allclose(block, block.flat[0], atol=1e-6, rtol=0.0)
    ):
        leaf_signature = _leaf_signature_from_block(block)
        specs.setdefault(
            leaf_signature,
            _LeafSpec(
                signature=leaf_signature,
                node_word=int(_leaf_node_from_signature(leaf_signature)),
            ),
        )
        return leaf_signature

    z_half = max(1, block.shape[0] // 2)
    y_half = max(1, block.shape[1] // 2)
    x_half = max(1, block.shape[2] // 2)

    # Octant diagonal extent for void detection
    octant_extent = (
        max(z_half, y_half, x_half) * cell_size * 1.7321 if cell_size > 0.0 else 0.0
    )

    child_signatures: list[bytes | None] = []
    mask = 0
    for octant in range(8):
        z_start = z_half if octant & 4 else 0
        y_start = y_half if octant & 2 else 0
        x_start = x_half if octant & 1 else 0
        child_block = block[
            z_start : (block.shape[0] if octant & 4 else z_half),
            y_start : (block.shape[1] if octant & 2 else y_half),
            x_start : (block.shape[2] if octant & 1 else x_half),
        ]

        # Void octant detection: skip if min SDF > octant diagonal
        if (
            octant_extent > 0.0
            and child_block.size > 0
            and float(child_block.min()) > octant_extent
        ):
            child_signatures.append(None)
            continue

        child_signatures.append(_build_signature_tree(child_block, specs, cell_size))
        mask |= 1 << octant

    # If all octants are void, fall back to a leaf
    if mask == 0:
        leaf_signature = _leaf_signature_from_block(block)
        specs.setdefault(
            leaf_signature,
            _LeafSpec(
                signature=leaf_signature,
                node_word=int(_leaf_node_from_signature(leaf_signature)),
            ),
        )
        return leaf_signature

    present_sigs = [s for s in child_signatures if s is not None]
    if len(set(present_sigs)) == 1 and mask == 0xFF:
        return present_sigs[0]

    unique_child_signatures, remap = deduplicate_signatures(present_sigs, seed=0)
    signature = _encode_internal_signature(
        mask=mask,
        unique_child_signatures=tuple(unique_child_signatures),
        remap=tuple(remap),
    )
    specs.setdefault(
        signature,
        _InternalSpec(
            signature=signature,
            mask=mask,
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
        child_indices.append(
            _emit_signature(child_signature, specs, dag_pool, emitted_indices)
        )

    for offset, unique_child_idx in enumerate(spec.remap):
        dag_pool[child_base + offset] = np.uint64(child_indices[unique_child_idx])

    dag_pool[node_index] = np.uint64(
        (int(spec.mask) << _INTERNAL_NODE_MASK_SHIFT) | child_base
    )
    return node_index


def compress_to_dag(
    grid: np.ndarray, resolution: int, cell_size: float = 0.0
) -> np.ndarray:
    """Return a fast native-format DAG payload compatible with the CUDA test backend."""
    res = _next_power_of_two(resolution)
    dense = np.asarray(grid, dtype=np.float32)
    if dense.shape != (res, res, res):
        dense = np.reshape(dense, (res, res, res)).astype(np.float32, copy=False)

    if dense.size == 0:
        return np.array(
            [np.uint64((1 << 63) | _float_to_half_bits(0.1))], dtype=np.uint64
        )

    specs: dict[bytes, _LeafSpec | _InternalSpec] = {}
    root_signature = _build_signature_tree(dense, specs, cell_size=cell_size)
    dag_pool: list[np.uint64] = []
    _emit_signature(root_signature, specs, dag_pool, emitted_indices={})
    return np.asarray(dag_pool, dtype=np.uint64)


def write_gmdag(
    path: str | Path,
    dag: np.ndarray,
    resolution: int,
    bbox_min: np.ndarray,
    voxel_size: float,
) -> None:
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
    if not bool(np.isfinite(float(voxel_size))):
        msg = "voxel_size must be finite"
        raise ValueError(msg)
    if not bool(np.all(np.isfinite(bbox))):
        msg = "bbox_min must contain only finite values"
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
    parser = argparse.ArgumentParser(
        description="Compile an OBJ mesh into a simple .gmdag verification artifact."
    )
    parser.add_argument("--input", required=True, help="Input OBJ mesh path")
    parser.add_argument("--output", required=True, help="Output .gmdag path")
    parser.add_argument(
        "--resolution", type=int, default=512, help="Target cubic resolution"
    )
    parser.add_argument(
        "--padding", type=float, default=0.1, help="Relative cubic padding"
    )
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
    dag = compress_to_dag(grid, args.resolution, cell_size=voxel_size)
    write_gmdag(args.output, dag, args.resolution, cube_min, voxel_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
