"""Diagnostic tool: probe all compiled .gmdag assets for geometry anomalies.

Reports bounding box, scale, voxel resolution, compression, and flags
potential issues that can cause distorted observations in the actor view.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

HEADER_STRUCT = struct.Struct("<4sIIffffI")


def diagnose_gmdag(path: Path) -> dict:
    """Read a .gmdag header and return diagnostic metadata."""
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header_bytes = f.read(HEADER_STRUCT.size)
        if len(header_bytes) != HEADER_STRUCT.size:
            return {"path": str(path), "error": "header too short"}
        magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = (
            HEADER_STRUCT.unpack(header_bytes)
        )

    extent = voxel_size * resolution
    bmax_x = bmin_x + extent
    bmax_y = bmin_y + extent
    bmax_z = bmin_z + extent

    # Compute mesh-centric metrics
    center_x = bmin_x + extent / 2
    center_y = bmin_y + extent / 2
    center_z = bmin_z + extent / 2

    dense_voxels = resolution**3
    compression_ratio = node_count / dense_voxels if dense_voxels > 0 else 0

    expected_payload = node_count * 8
    actual_payload = file_size - HEADER_STRUCT.size

    issues: list[str] = []

    # Check magic
    if magic != b"GDAG":
        issues.append(f"BAD_MAGIC={magic!r}")

    # Check version
    if version != 1:
        issues.append(f"BAD_VERSION={version}")

    # Check finite values
    for name_val, val in [
        ("bmin_x", bmin_x),
        ("bmin_y", bmin_y),
        ("bmin_z", bmin_z),
        ("voxel_size", voxel_size),
    ]:
        if not math.isfinite(val):
            issues.append(f"NON_FINITE_{name_val}={val}")

    # Scale checks
    if extent > 500.0:
        issues.append(f"HUGE_EXTENT={extent:.1f}m")
    elif extent > 200.0:
        issues.append(f"LARGE_EXTENT={extent:.1f}m")
    if extent < 1.0:
        issues.append(f"TINY_EXTENT={extent:.3f}m")
    elif extent < 3.0:
        issues.append(f"SMALL_EXTENT={extent:.2f}m")

    # Voxel size check (for resolution=512, typical scenes 5-50m => voxel 0.01-0.1m)
    if voxel_size > 1.0:
        issues.append(f"COARSE_VOXEL={voxel_size:.3f}m")
    elif voxel_size > 0.5:
        issues.append(f"ROUGH_VOXEL={voxel_size:.3f}m")
    if voxel_size < 0.001:
        issues.append(f"MICRO_VOXEL={voxel_size:.6f}m")

    # Center-of-mass: if center is very far from origin, may indicate coordinate issues
    dist_from_origin = math.sqrt(center_x**2 + center_y**2 + center_z**2)
    if dist_from_origin > 200.0:
        issues.append(f"FAR_CENTER={dist_from_origin:.1f}m")

    # Y-range check: typical indoor scenes have Y from ~0 to ~3-5m
    # If Y-min is very negative or Y-max is very large, something may be wrong
    if bmin_y < -100.0:
        issues.append(f"DEEP_YMIN={bmin_y:.1f}m")
    if bmax_y > 100.0:
        issues.append(f"HIGH_YMAX={bmax_y:.1f}m")

    # Payload integrity
    if expected_payload != actual_payload:
        issues.append(f"PAYLOAD_MISMATCH(expected={expected_payload},actual={actual_payload})")

    # Compression ratio: very high suggests mostly empty space (fine)
    # Very low suggests nearly dense (unusual for scenes)
    if compression_ratio > 0.5:
        issues.append(f"LOW_COMPRESSION={compression_ratio:.3f}")
    if node_count < 100:
        issues.append(f"VERY_FEW_NODES={node_count}")

    return {
        "path": str(path),
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "resolution": resolution,
        "voxel_size": voxel_size,
        "extent": extent,
        "bbox_min": (bmin_x, bmin_y, bmin_z),
        "bbox_max": (bmax_x, bmax_y, bmax_z),
        "center": (center_x, center_y, center_z),
        "node_count": node_count,
        "compression_ratio": compression_ratio,
        "file_size_mb": file_size / (1024 * 1024),
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
    print(f"Corpus root: {gmdag_root}")
    print(f"Scene count: {len(scenes)}")
    print()
    print(
        f"{'DATASET':45s} | {'NAME':35s} | {'EXTENT':>8s} | {'VOXEL':>7s} | "
        f"{'BBOX_MIN':>28s} | {'BBOX_MAX':>28s} | {'NODES':>8s} | {'CR':>6s} | STATUS"
    )
    print("-" * 200)

    all_results = []
    issue_count = 0

    for scene in scenes:
        gmdag_path = gmdag_root / scene["gmdag_path"]
        if not gmdag_path.exists():
            print(f"MISSING: {scene['scene_name']} from {scene['dataset']}")
            continue

        result = diagnose_gmdag(gmdag_path)
        result["dataset"] = scene["dataset"]
        result["scene_name"] = scene["scene_name"]
        all_results.append(result)

        if result.get("error"):
            print(f"ERROR: {scene['scene_name']}: {result['error']}")
            continue

        bmin = result["bbox_min"]
        bmax = result["bbox_max"]
        status = " | ".join(result["issues"]) if result["issues"] else "OK"
        if result["issues"]:
            issue_count += 1

        print(
            f"{result['dataset']:45s} | {result['scene_name']:35s} | "
            f"{result['extent']:7.2f}m | {result['voxel_size']:6.4f} | "
            f"[{bmin[0]:8.2f},{bmin[1]:8.2f},{bmin[2]:8.2f}] | "
            f"[{bmax[0]:8.2f},{bmax[1]:8.2f},{bmax[2]:8.2f}] | "
            f"{result['node_count']:>8d} | {result['compression_ratio']:.4f} | {status}"
        )

    print()
    print("=" * 100)
    print(f"SUMMARY: {len(all_results)} scenes analyzed, {issue_count} with issues")
    print()

    # Group by dataset
    datasets: dict[str, list[dict]] = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        datasets.setdefault(ds, []).append(r)

    for ds in sorted(datasets):
        ds_scenes = datasets[ds]
        extents = [r["extent"] for r in ds_scenes if "extent" in r]
        voxels = [r["voxel_size"] for r in ds_scenes if "voxel_size" in r]
        y_mins = [r["bbox_min"][1] for r in ds_scenes if "bbox_min" in r]
        y_maxs = [r["bbox_max"][1] for r in ds_scenes if "bbox_max" in r]

        print(f"Dataset: {ds}")
        print(f"  Scenes: {len(ds_scenes)}")
        if extents:
            print(f"  Extent range: {min(extents):.2f}m - {max(extents):.2f}m")
            print(f"  Voxel range:  {min(voxels):.4f}m - {max(voxels):.4f}m")
            print(f"  Y range:      [{min(y_mins):.2f}, {max(y_maxs):.2f}]m")
        for r in ds_scenes:
            if r.get("issues"):
                print(f"  ** {r['scene_name']}: {r['issues']}")
        print()


if __name__ == "__main__":
    main()
