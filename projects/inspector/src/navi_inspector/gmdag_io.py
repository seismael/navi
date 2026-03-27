"""Standalone .gmdag binary reader.

Self-contained — no imports from navi-environment or navi-contracts.
Replicates the 32-byte header struct and uint64 node pool loading
from the canonical environment loader.

Binary Format (v1):
    Header (32 bytes, little-endian):
        magic       char[4]   "GDAG"
        version     uint32    1
        resolution  uint32    grid resolution (e.g. 512)
        bmin_x      float32   bounding-box minimum X
        bmin_y      float32   bounding-box minimum Y
        bmin_z      float32   bounding-box minimum Z
        voxel_size  float32   side length of one voxel (metres)
        node_count  uint32    number of 64-bit nodes
    Node pool:
        node_count × uint64   contiguous little-endian
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__: list[str] = ["GmdagAsset", "load_gmdag", "gmdag_info"]

_HEADER_STRUCT = struct.Struct("<4sIIffffI")
_HEADER_SIZE = _HEADER_STRUCT.size  # 32 bytes


@dataclass(frozen=True, slots=True)
class GmdagAsset:
    """Loaded .gmdag asset: metadata plus contiguous uint64 node array."""

    path: Path
    version: int
    resolution: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    voxel_size: float
    nodes: np.ndarray  # dtype=uint64, shape=(node_count,)


def load_gmdag(path: str | Path) -> GmdagAsset:
    """Load a .gmdag binary into a GmdagAsset.

    Validates magic, version, resolution, bounds, voxel_size, node count,
    and absence of trailing bytes.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        msg = f"gmdag file does not exist: {resolved}"
        raise FileNotFoundError(msg)

    with resolved.open("rb") as fh:
        header_bytes = fh.read(_HEADER_SIZE)
        if len(header_bytes) != _HEADER_SIZE:
            msg = f"Invalid gmdag header in {resolved}: file too small"
            raise RuntimeError(msg)

        magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = (
            _HEADER_STRUCT.unpack(header_bytes)
        )

        if magic != b"GDAG":
            msg = f"Unsupported gmdag magic in {resolved}: {magic!r}"
            raise RuntimeError(msg)
        if version != 1:
            msg = f"Unsupported gmdag version in {resolved}: {version}"
            raise RuntimeError(msg)
        if resolution <= 0:
            msg = f"Invalid gmdag resolution in {resolved}: {resolution}"
            raise RuntimeError(msg)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            msg = f"Invalid gmdag voxel size in {resolved}: {voxel_size}"
            raise RuntimeError(msg)
        if node_count <= 0:
            msg = f"Invalid gmdag node count in {resolved}: {node_count}"
            raise RuntimeError(msg)
        bbox_values = (bmin_x, bmin_y, bmin_z)
        if not all(np.isfinite(v) for v in bbox_values):
            msg = f"Non-finite bbox_min in {resolved}"
            raise RuntimeError(msg)

        nodes = np.fromfile(fh, dtype=np.uint64, count=node_count)
        if nodes.shape[0] != node_count:
            msg = (
                f"Truncated node payload in {resolved}: "
                f"expected {node_count}, got {nodes.shape[0]}"
            )
            raise RuntimeError(msg)

        trailing = fh.read(1)
        if trailing:
            msg = f"Trailing bytes in {resolved}"
            raise RuntimeError(msg)

    bbox_min = (float(bmin_x), float(bmin_y), float(bmin_z))
    extent = float(voxel_size) * int(resolution)
    bbox_max = (
        bbox_min[0] + extent,
        bbox_min[1] + extent,
        bbox_min[2] + extent,
    )

    return GmdagAsset(
        path=resolved,
        version=int(version),
        resolution=int(resolution),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size=float(voxel_size),
        nodes=np.ascontiguousarray(nodes),
    )


def gmdag_info(path: str | Path) -> dict[str, Any]:
    """Read only the header metadata (no node pool) for quick inspection."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        msg = f"gmdag file does not exist: {resolved}"
        raise FileNotFoundError(msg)

    file_size = resolved.stat().st_size
    with resolved.open("rb") as fh:
        header_bytes = fh.read(_HEADER_SIZE)
        if len(header_bytes) != _HEADER_SIZE:
            msg = f"Invalid gmdag header in {resolved}"
            raise RuntimeError(msg)

        magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = (
            _HEADER_STRUCT.unpack(header_bytes)
        )

    bbox_min = (float(bmin_x), float(bmin_y), float(bmin_z))
    extent = float(voxel_size) * int(resolution)
    bbox_max = (
        bbox_min[0] + extent,
        bbox_min[1] + extent,
        bbox_min[2] + extent,
    )

    return {
        "path": str(resolved),
        "magic": magic.decode("ascii", errors="replace"),
        "version": int(version),
        "resolution": int(resolution),
        "voxel_size": float(voxel_size),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "node_count": int(node_count),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "expected_payload_bytes": int(node_count) * 8,
    }
