"""Deep structural diagnostic for .gmdag assets.

Probes each scene by:
1. Reading header and validating metadata
2. Walking the DAG to check node encoding consistency
3. Checking for degenerate scenes (allvoid, all-surface, extreme distributions)
4. Reporting per-dataset statistics to identify dataset-specific issues
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

HEADER_STRUCT = struct.Struct("<4sIIffffI")

LEAF_FLAG = np.uint64(1) << np.uint64(63)


def walk_dag_stats(nodes: np.ndarray) -> dict:
    """Walk the DAG from root to collect structural statistics."""
    node_count = len(nodes)
    
    internal_count = 0
    leaf_count = 0
    void_octants = 0
    total_octants = 0
    min_leaf_dist = float("inf")
    max_leaf_dist = float("-inf")
    surface_leaf_count = 0  # semantic != 0
    empty_leaf_count = 0    # semantic == 0
    
    # Leaf distance histogram
    dist_bins = np.zeros(20, dtype=np.int64)  # 0-0.5, 0.5-1, ..., 9.5-10+ meters
    
    visited = set()
    stack = [0]
    
    max_visits = min(node_count, 2_000_000)  # cap to avoid stack overflow on huge DAGs
    visit_count = 0
    
    while stack and visit_count < max_visits:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        visit_count += 1
        
        if idx < 0 or idx >= node_count:
            continue
        
        word = int(nodes[idx])
        
        try:
            if (word >> 63) & 1:
                # Leaf node
                leaf_count += 1
                dist_bits = word & 0xFFFF
                # Decode float16
                dist_array = np.array([dist_bits], dtype=np.uint16).view(np.float16)
                dist_val = float(dist_array[0])
                
                semantic = (word >> 16) & 0xFFFF
                if semantic != 0:
                    surface_leaf_count += 1
                else:
                    empty_leaf_count += 1
                
                if math.isfinite(dist_val) and dist_val >= 0:
                    if dist_val < min_leaf_dist:
                        min_leaf_dist = dist_val
                    if dist_val > max_leaf_dist:
                        max_leaf_dist = dist_val
                    
                    bin_idx = min(19, int(dist_val / 0.5))
                    if bin_idx >= 0:
                        dist_bins[bin_idx] += 1
            else:
                # Internal node
                internal_count += 1
                mask = (word >> 55) & 0xFF
                child_base = word & 0xFFFFFFFF
                child_count = bin(mask).count("1")
                void_octants += (8 - child_count)
                total_octants += 8
                
                for offset in range(child_count):
                    child_ptr_idx = int(child_base) + offset
                    if 0 <= child_ptr_idx < node_count:
                        child_ptr = int(nodes[child_ptr_idx]) & 0xFFFFFFFF
                        if 0 <= child_ptr < node_count:
                            stack.append(child_ptr)
        except (OverflowError, ValueError):
            continue
    
    return {
        "visited": visit_count,
        "internal_nodes": internal_count,
        "leaf_nodes": leaf_count,
        "void_octants": void_octants,
        "total_octants": total_octants,
        "void_ratio": void_octants / max(1, total_octants),
        "surface_leaves": surface_leaf_count,
        "empty_leaves": empty_leaf_count,
        "surface_ratio": surface_leaf_count / max(1, leaf_count),
        "min_leaf_dist": min_leaf_dist if math.isfinite(min_leaf_dist) else None,
        "max_leaf_dist": max_leaf_dist if math.isfinite(max_leaf_dist) else None,
        "dist_bins": dist_bins.tolist(),
    }


def diagnose_scene(path: Path) -> dict:
    """Full structural diagnostic for one .gmdag file."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header_bytes = f.read(HEADER_STRUCT.size)
        magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = (
            HEADER_STRUCT.unpack(header_bytes)
        )
        nodes = np.fromfile(f, dtype=np.uint64, count=node_count)

    extent = voxel_size * resolution
    
    # Walk DAG
    dag_stats = walk_dag_stats(nodes)
    
    issues = []
    
    # Check: Is the scene mostly void? (could indicate bad compilation)
    if dag_stats["surface_leaves"] == 0:
        issues.append("NO_SURFACE_LEAVES")
    
    if dag_stats["surface_ratio"] < 0.001 and dag_stats["leaf_nodes"] > 100:
        issues.append(f"VERY_LOW_SURFACE_RATIO={dag_stats['surface_ratio']:.6f}")
    
    # Check: very high void ratio might mean the mesh was tiny relative to
    # the grid, losing most resolution to padding
    if dag_stats["void_ratio"] > 0.99:
        issues.append(f"EXTREME_VOID={dag_stats['void_ratio']:.4f}")
    
    # Check: min distance too small or negative
    if dag_stats["min_leaf_dist"] is not None and dag_stats["min_leaf_dist"] < 0:
        issues.append(f"NEGATIVE_DIST={dag_stats['min_leaf_dist']:.4f}")
    
    # Check: root node format
    root_word = int(nodes[0])
    root_is_leaf = bool((root_word >> 63) & 1)
    if root_is_leaf:
        issues.append("ROOT_IS_LEAF")
    
    return {
        "path": str(path),
        "resolution": resolution,
        "voxel_size": voxel_size,
        "extent": extent,
        "bbox_min": (bmin_x, bmin_y, bmin_z),
        "node_count": node_count,
        "dag_stats": dag_stats,
        "issues": issues,
    }


def main() -> None:
    manifest_path = Path("artifacts/gmdag/corpus/gmdag_manifest.json")
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with manifest_path.open() as f:
        manifest = json.load(f)

    gmdag_root = Path(manifest["gmdag_root"])
    scenes = manifest["scenes"]
    
    print(f"Deep structural analysis of {len(scenes)} scenes")
    print("=" * 120)
    
    for scene in scenes:
        gmdag_path = gmdag_root / scene["gmdag_path"]
        name = scene["scene_name"]
        dataset = scene["dataset"]
        
        if not gmdag_path.exists():
            print(f"MISSING: {name}")
            continue
        
        try:
            result = diagnose_scene(gmdag_path)
        except Exception as exc:
            print(f"\n{dataset} / {name}")
            print(f"  ERROR: {exc}")
            continue
        stats = result["dag_stats"]
        issue_str = " | ".join(result["issues"]) if result["issues"] else "OK"
        
        print(f"\n{dataset} / {name}")
        print(f"  Extent: {result['extent']:.2f}m, Voxel: {result['voxel_size']:.4f}m")
        print(f"  Nodes: {result['node_count']}, Visited: {stats['visited']}")
        print(f"  Internal: {stats['internal_nodes']}, Leaves: {stats['leaf_nodes']}")
        print(f"  Surface leaves: {stats['surface_leaves']}, Empty leaves: {stats['empty_leaves']}")
        print(f"  Surface ratio: {stats['surface_ratio']:.4f}, Void ratio: {stats['void_ratio']:.4f}")
        if stats["min_leaf_dist"] is not None:
            print(f"  Dist range: [{stats['min_leaf_dist']:.4f}, {stats['max_leaf_dist']:.4f}]m")
        
        # Print distance histogram  
        bins = stats["dist_bins"]
        total_binned = sum(bins)
        if total_binned > 0:
            hist_str = " ".join(f"{b/total_binned*100:.0f}%" for b in bins[:10])
            print(f"  Dist hist (0-5m, 0.5m bins): {hist_str}")
        
        print(f"  Status: {issue_str}")
    
    print("\n" + "=" * 120)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
