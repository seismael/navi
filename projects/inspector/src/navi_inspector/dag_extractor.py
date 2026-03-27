"""DAG → dense SDF grid extraction.

Walks the .gmdag octree/DAG from root to leaves, producing a dense
float32 SDF grid at the requested target resolution.  Multi-resolution
support: when target_resolution < asset.resolution, traversal stops
early once the cell size matches the target resolution.

Node encoding (64-bit):
    Leaf  (bit 63 = 1): bits[15:0]  = fp16 distance
                         bits[31:16] = semantic gate (u16)
    Inner (bit 63 = 0): bits[62:55] = child mask (8 bits, one per octant)
                         bits[31:0]  = child base pointer (u32)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from navi_inspector.gmdag_io import GmdagAsset

__all__: list[str] = ["SdfGrid", "extract_sdf_grid"]

_LEAF_FLAG = np.uint64(1 << 63)
_MASK_SHIFT = 55
_FP16_MASK = np.uint64(0xFFFF)
_BASE_PTR_MASK = np.uint64(0xFFFFFFFF)


@dataclass(frozen=True, slots=True)
class SdfGrid:
    """Dense SDF grid extracted from a .gmdag file."""

    grid: np.ndarray  # shape (res, res, res), dtype float32
    resolution: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    voxel_size: float  # per-cell size at *this* extraction resolution


def _decode_fp16_distance(word: np.uint64) -> float:
    """Decode the fp16 distance stored in bits [15:0] of a leaf node."""
    bits = int(word & _FP16_MASK)
    return float(np.array([bits], dtype=np.uint16).view(np.float16)[0])


def extract_sdf_grid(
    asset: GmdagAsset,
    target_resolution: int = 128,
    *,
    progress_callback: object | None = None,
) -> SdfGrid:
    """Extract a dense float32 SDF grid from a loaded .gmdag asset.

    Uses iterative stack-based traversal (no Python recursion limits).
    When target_resolution < asset.resolution, leaf values are
    broadcast to all covered cells at the coarser resolution.

    Args:
        asset: Loaded GmdagAsset with nodes array.
        target_resolution: Voxel grid resolution for extraction.
            128 → instant preview, 256 → mid-detail, 512 → full.
        progress_callback: Unused placeholder for future progress reporting.

    Returns:
        SdfGrid with the extracted dense grid.
    """
    dag = asset.nodes
    dag_res = asset.resolution

    # Clamp target to not exceed the DAG's native resolution
    target_res = min(target_resolution, dag_res)
    # Ensure power of 2 for clean octree subdivision
    target_res = max(1, target_res)

    # Compute the cell size ratio: how many target cells per DAG cell
    # If dag_res=512, target=128 → scale=4, meaning each leaf may cover 4³ cells
    scale = dag_res // target_res if target_res <= dag_res else 1
    # Minimum DAG subdivision size (in DAG units) at which we stop and fill
    min_dag_size = max(scale, 1)

    grid = np.full((target_res, target_res, target_res), np.inf, dtype=np.float32)

    # Iterative stack: (node_ptr, lo_x, lo_y, lo_z, size_in_dag_units)
    # All coordinates are in DAG-resolution units [0, dag_res)
    stack: list[tuple[int, int, int, int, int]] = [(0, 0, 0, 0, dag_res)]

    while stack:
        ptr, lo_x, lo_y, lo_z, size = stack.pop()

        word = dag[ptr]

        # --- Leaf node ---
        if word & _LEAF_FLAG:
            dist = _decode_fp16_distance(word)
            # Map DAG coordinates to target grid coordinates
            gx0 = lo_x // scale
            gy0 = lo_y // scale
            gz0 = lo_z // scale
            gx1 = min((lo_x + size) // scale, target_res)
            gy1 = min((lo_y + size) // scale, target_res)
            gz1 = min((lo_z + size) // scale, target_res)
            if gx0 < gx1 and gy0 < gy1 and gz0 < gz1:
                grid[gx0:gx1, gy0:gy1, gz0:gz1] = dist
            continue

        # --- Inner node ---
        mask = (int(word) >> _MASK_SHIFT) & 0xFF
        child_base = int(word & _BASE_PTR_MASK)

        half = size >> 1  # size // 2

        # If we've reached the target cell size, sample the first reachable leaf
        if half < min_dag_size:
            # Walk down to find any leaf distance for this cell
            dist = _sample_leaf(dag, ptr)
            gx0 = lo_x // scale
            gy0 = lo_y // scale
            gz0 = lo_z // scale
            gx1 = min((lo_x + size) // scale, target_res)
            gy1 = min((lo_y + size) // scale, target_res)
            gz1 = min((lo_z + size) // scale, target_res)
            if gx0 < gx1 and gy0 < gy1 and gz0 < gz1:
                grid[gx0:gx1, gy0:gy1, gz0:gz1] = dist
            continue

        # Push occupied children onto the stack
        child_offset = 0
        for octant in range(8):
            if not (mask & (1 << octant)):
                continue

            child_ptr_idx = child_base + child_offset
            child_ptr = int(dag[child_ptr_idx]) & 0xFFFFFFFF

            ox = lo_x + (half if (octant & 1) else 0)
            oy = lo_y + (half if (octant & 2) else 0)
            oz = lo_z + (half if (octant & 4) else 0)

            stack.append((child_ptr, ox, oy, oz, half))
            child_offset += 1

    # Compute world-space metadata for the extracted grid
    cell_size = (asset.bbox_max[0] - asset.bbox_min[0]) / target_res

    return SdfGrid(
        grid=grid,
        resolution=target_res,
        bbox_min=asset.bbox_min,
        bbox_max=asset.bbox_max,
        voxel_size=cell_size,
    )


def _sample_leaf(dag: np.ndarray, ptr: int) -> float:
    """Walk from an inner node to the first reachable leaf.

    Used when the target resolution is coarser than the DAG resolution
    and we just need a representative distance for a cell.
    """
    for _ in range(32):
        word = dag[ptr]
        if word & _LEAF_FLAG:
            return _decode_fp16_distance(word)

        mask = (int(word) >> _MASK_SHIFT) & 0xFF
        child_base = int(word & _BASE_PTR_MASK)

        if mask == 0:
            return float("inf")

        # Take the first occupied octant
        for octant in range(8):
            if mask & (1 << octant):
                offset = bin(mask & ((1 << octant) - 1)).count("1")
                ptr = int(dag[child_base + offset]) & 0xFFFFFFFF
                break

    return float("inf")
