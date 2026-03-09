import numpy as np
import time
import os
import struct
import pytest
from voxel_dag.compiler import MeshIngestor, compute_dense_sdf, compress_to_dag, write_gmdag

# --- Test Utilities ---

def create_test_obj(filename, shape='cube'):
    if shape == 'cube':
        content = """v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""
    elif shape == 'plane':
        content = """v -1 0 -1
v 1 0 -1
v 1 0 1
v -1 0 1
f 1 2 3
f 1 3 4
"""
    else:
        content = "v 0 0 0\n" # Degenerate
    
    with open(filename, 'w') as f:
        f.write(content)

# --- 1. Functional Tests ---

def test_mesh_ingestion():
    create_test_obj("test_cube.obj", 'cube')
    v, i, bmin, bmax = MeshIngestor.load_obj("test_cube.obj")
    assert len(v) == 8
    assert len(i) == 12
    assert np.all(bmin == 0)
    assert np.all(bmax == 1)
    os.remove("test_cube.obj")

def test_eikonal_accuracy():
    """Verify FSM produces correct Euclidean distances for a plane."""
    create_test_obj("test_plane.obj", 'plane')
    v, i, bmin, bmax = MeshIngestor.load_obj("test_plane.obj")
    res = 32
    grid, h, cmin = compute_dense_sdf(v, i, bmin, bmax, res, padding=0.5)
    
    # Grid is now a cubic padding around the plane at y=0.
    # We find the center voxel and check its distance.
    mid = res // 2
    dist_at_center = grid[mid, mid, mid]
    assert dist_at_center < h * 2 
    
    # Check linear growth
    dist_above = grid[mid, mid + 5, mid] # y is the second dimension in (z, y, x)
    expected_dist = 5 * h
    assert np.isclose(dist_above, expected_dist, atol=h)
    os.remove("test_plane.obj")

def test_dag_compression_efficiency():
    """Verify that uniform space is aggressively deduplicated."""
    res = 32
    # Uniform distance field
    grid = np.full((res, res, res), 5.0, dtype=np.float32)
    dag = compress_to_dag(grid, res)
    
    # A uniform 32^3 grid (32,768 voxels) should fold into a very small number of unique nodes
    # (Log2(32) = 5 levels of internal nodes + 1 leaf node)
    assert len(dag) < 100
    print(f"Compression Ratio: { (res**3) / len(dag) :.1f}x")

# --- 2. Edge Case Tests ---

def test_degenerate_mesh():
    """Handle meshes with single vertices or no faces."""
    create_test_obj("degenerate.obj", 'degenerate')
    try:
        v, i, bmin, bmax = MeshIngestor.load_obj("degenerate.obj")
        grid, h, cmin = compute_dense_sdf(v, i, bmin, bmax, 16)
        assert grid.shape == (16, 16, 16)
    except Exception as e:
        pytest.fail(f"Degenerate mesh caused crash: {e}")
    os.remove("degenerate.obj")

def test_extreme_resolution():
    """Verify memory handling at higher resolutions."""
    res = 128
    grid = np.zeros((res, res, res), dtype=np.float32)
    start = time.time()
    dag = compress_to_dag(grid, res)
    end = time.time()
    assert len(dag) > 0
    print(f"Deduplication for {res}^3 took {end - start:.2f}s")

# --- 3. Performance Benchmarks ---

def benchmark_fsm_scaling():
    resolutions = [32, 64, 128]
    create_test_obj("bench.obj", 'cube')
    v, i, bmin, bmax = MeshIngestor.load_obj("bench.obj")
    
    print("\nFSM Performance Scaling (Agnostic & Cubic):")
    for res in resolutions:
        start = time.time()
        compute_dense_sdf(v, i, bmin, bmax, res)
        duration = time.time() - start
        print(f"  {res:3d}^3: {duration:6.3f}s ({(res**3)/duration/1e6:.2f} MVoxels/s)")
    os.remove("bench.obj")

# --- 4. Format Verification ---

def test_serialization_integrity():
    res = 16
    dag = np.arange(10, dtype=np.uint64)
    bmin = np.array([0, 0, 0], dtype=np.float32)
    write_gmdag("test.gmdag", dag, res, bmin, 0.1)
    
    with open("test.gmdag", 'rb') as f:
        magic = f.read(4)
        assert magic == b'GDAG'
        version = struct.unpack('<I', f.read(4))[0]
        assert version == 1
        read_res = struct.unpack('<I', f.read(4))[0]
        assert read_res == res
    os.remove("test.gmdag")

if __name__ == "__main__":
    print("Running TopoNav Voxel-DAG Comprehensive Test Suite...")
    test_mesh_ingestion()
    print("[PASS] Mesh Ingestion")
    test_eikonal_accuracy()
    print("[PASS] Eikonal Mathematical Accuracy")
    test_dag_compression_efficiency()
    print("[PASS] DAG Compression Deduplication")
    test_degenerate_mesh()
    print("[PASS] Edge Case: Degenerate Geometry")
    test_extreme_resolution()
    print("[PASS] Scale Test: 128^3 Resolution")
    test_serialization_integrity()
    print("[PASS] Serialization Binary Integrity")
    
    benchmark_fsm_scaling()
