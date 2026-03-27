"""Quick smoke test against a real .gmdag file."""
import sys
import time
from pathlib import Path

# Test against the smallest corpus file
GMDAG = Path(r"C:\dev\projects\navi\artifacts\gmdag\corpus\ai-habitat_habitat_test_scenes\van-gogh-room.gmdag")

if not GMDAG.exists():
    print(f"SKIP: {GMDAG} not found")
    sys.exit(0)

from navi_inspector.gmdag_io import load_gmdag, gmdag_info
from navi_inspector.dag_extractor import extract_sdf_grid
from navi_inspector.mesh_builder import build_mesh, export_mesh

# Phase 1: info
info = gmdag_info(GMDAG)
print(f"[INFO] version={info['version']}, resolution={info['resolution']}, nodes={info['node_count']}, size={info['file_size_mb']}MB")

# Phase 2: load
t0 = time.perf_counter()
asset = load_gmdag(GMDAG)
t1 = time.perf_counter()
print(f"[LOAD] {t1-t0:.3f}s — {asset.nodes.shape[0]} nodes, voxel_size={asset.voxel_size:.6f}")

# Phase 3: extract
grid = extract_sdf_grid(asset, target_resolution=64)
t2 = time.perf_counter()
print(f"[EXTRACT] {t2-t1:.3f}s — grid {grid.grid.shape}, voxel_size={grid.voxel_size:.4f}")

# Phase 4: mesh
mesh = build_mesh(grid)
t3 = time.perf_counter()
print(f"[MESH] {t3-t2:.3f}s — {mesh.n_points} verts, {mesh.n_cells} faces")

# Phase 5: export
out_path = Path("tests/van-gogh-test.ply")
out = export_mesh(mesh, out_path)
t4 = time.perf_counter()
print(f"[EXPORT] {t4-t3:.3f}s — {out} ({out.stat().st_size / 1024:.1f} KB)")

print(f"\nTotal pipeline: {t4-t0:.3f}s")
print("SMOKE TEST PASSED")
