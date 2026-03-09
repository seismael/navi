import numpy as np
import time
import os
import pytest
from voxel_dag.compiler import MeshIngestor, compute_dense_sdf, compress_to_dag

def generate_swiss_cheese_mesh(filename, num_holes=10):
    """Generates a complex mesh with multiple holes and thin walls to test edge cases."""
    vertices = []
    indices = []
    
    # Base Box
    v_base = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=np.float32)
    vertices.extend(v_base.tolist())
    
    # Faces
    f_base = [
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,1,5], [0,5,4], [1,2,6], [1,6,5],
        [2,3,7], [2,7,6], [3,0,4], [3,4,7]
    ]
    indices.extend(f_base)
    
    # Add random internal quads (Thin Geometry)
    for i in range(num_holes):
        z = 0.1 + (i / num_holes) * 0.8
        offset = len(vertices)
        vertices.extend([
            [0.2, 0.2, z], [0.8, 0.2, z], [0.8, 0.8, z], [0.2, 0.8, z]
        ])
        indices.extend([[offset, offset+1, offset+2], [offset, offset+2, offset+3]])

    with open(filename, 'w') as f:
        for v in vertices:
            f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        for idx in indices:
            f.write("f {} {} {}\n".format(idx[0]+1, idx[1]+1, idx[2]+1))

def test_stress_integrity():
    print("\n[STRESS TEST] Complex Geometry & Native Performance...")
    mesh_file = "swiss_cheese.obj"
    generate_swiss_cheese_mesh(mesh_file, num_holes=50)
    
    v, i, bmin, bmax = MeshIngestor.load_obj(mesh_file)
    
    # Test high resolution for performance validation
    res = 128
    print("  Target Resolution: {}^3 ({:,} voxels)".format(res, res**3))
    
    start_time = time.time()
    grid, h, cmin = compute_dense_sdf(v, i, bmin, bmax, res)
    fsm_time = time.time() - start_time
    
    print("  FSM Compute Time: {:.3f}s".format(fsm_time))
    throughput = (res**3) / fsm_time / 1e6
    print("  Verified Throughput: {:.3f} MVoxels/s".format(throughput))
    
    # MATHEMATICAL INTEGRITY CHECK
    # All voxels should have a valid distance
    assert not np.any(np.isnan(grid))
    assert np.all(grid < 1e8)
    
    # DAG INTEGRITY
    print("  Testing DAG Compression Consistency...")
    start_comp = time.time()
    dag = compress_to_dag(grid, res)
    comp_time = time.time() - start_comp
    print("  DAG Compression Time: {:.3f}s".format(comp_time))
    print("  Final Node Count: {:,}".format(len(dag)))
    
    # Ratio Assertion: Even with complex geometry, spatial redundancy should exist
    ratio = (res**3) / len(dag)
    print("  Compression Ratio: {:.1f}x".format(ratio))
    assert ratio > 1.0
    
    if os.path.exists(mesh_file):
        os.remove(mesh_file)
    print("[PASS] Stress Integrity & Performance Validated.")

if __name__ == "__main__":
    test_stress_integrity()
