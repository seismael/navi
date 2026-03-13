from pathlib import Path

import numpy as np
import os
import subprocess
import struct
import time

import pytest

from voxel_dag.compiler import MeshIngestor, compute_dense_sdf, compress_to_dag, write_gmdag


_HEADER = struct.Struct("<4sIIffffI")


def _resolve_native_compiler() -> Path | None:
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        project_root / "build" / "Release" / "voxel-dag.exe",
        project_root / "build-local" / "Release" / "voxel-dag.exe",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_gmdag_header(path: Path) -> tuple[bytes, int, int, tuple[float, float, float], float, int]:
    with path.open("rb") as handle:
        raw = handle.read(_HEADER.size)
    magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = _HEADER.unpack(raw)
    return magic, version, resolution, (bmin_x, bmin_y, bmin_z), voxel_size, node_count

# --- Test Utilities ---

def create_test_obj(filename: str | Path, shape: str = "cube") -> None:
    if shape == "cube":
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
    elif shape == "plane":
        content = """v -1 0 -1
v 1 0 -1
v 1 0 1
v -1 0 1
f 1 2 3
f 1 3 4
"""
    else:
        content = "v 0 0 0\n"  # Degenerate

    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

# --- 1. Functional Tests ---

def test_mesh_ingestion() -> None:
    create_test_obj("test_cube.obj", "cube")
    v, i, bmin, bmax = MeshIngestor.load_obj("test_cube.obj")
    assert len(v) == 8
    assert len(i) == 12
    assert np.all(bmin == 0)
    assert np.all(bmax == 1)
    os.remove("test_cube.obj")

def test_eikonal_accuracy() -> None:
    """Verify FSM produces correct Euclidean distances for a plane."""
    create_test_obj("test_plane.obj", "plane")
    v, i, bmin, bmax = MeshIngestor.load_obj("test_plane.obj")
    res = 32
    grid, h, _cmin = compute_dense_sdf(v, i, bmin, bmax, res, padding=0.5)
    
    # Grid is now a cubic padding around the plane at y=0.
    # We find the center voxel and check its distance.
    mid = res // 2
    dist_at_center = grid[mid, mid, mid]
    assert dist_at_center < h * 2
    
    # Check linear growth
    dist_above = grid[mid, mid + 5, mid]  # y is the second dimension in (z, y, x)
    expected_dist = 5 * h
    assert np.isclose(dist_above, expected_dist, atol=h)
    os.remove("test_plane.obj")

def test_dag_compression_efficiency() -> None:
    """Verify that uniform space is aggressively deduplicated."""
    res = 32
    # Uniform distance field
    grid = np.full((res, res, res), 5.0, dtype=np.float32)
    dag = compress_to_dag(grid, res)
    
    # A uniform 32^3 grid (32,768 voxels) should fold into a very small number of unique nodes
    # (Log2(32) = 5 levels of internal nodes + 1 leaf node)
    assert len(dag) < 100
    print(f"Compression Ratio: {(res**3) / len(dag):.1f}x")


def test_box_distance_matches_nearest_face_at_voxel_centers() -> None:
    create_test_obj("test_cube.obj", "cube")
    vertices, indices, bbox_min, bbox_max = MeshIngestor.load_obj("test_cube.obj")
    resolution = 32
    grid, voxel_size, cube_min = compute_dense_sdf(vertices, indices, bbox_min, bbox_max, resolution, padding=0.0)

    x_index = 7
    y_index = 10
    z_index = 21
    sample_point = cube_min + (np.array([x_index, y_index, z_index], dtype=np.float32) + 0.5) * voxel_size
    expected_distance = min(
        sample_point[0] - bbox_min[0],
        bbox_max[0] - sample_point[0],
        sample_point[1] - bbox_min[1],
        bbox_max[1] - sample_point[1],
        sample_point[2] - bbox_min[2],
        bbox_max[2] - sample_point[2],
    )

    assert np.isclose(grid[z_index, y_index, x_index], expected_distance, atol=voxel_size)
    os.remove("test_cube.obj")

# --- 2. Edge Case Tests ---

def test_degenerate_mesh() -> None:
    """Handle meshes with single vertices or no faces."""
    create_test_obj("degenerate.obj", "degenerate")
    try:
        v, i, bmin, bmax = MeshIngestor.load_obj("degenerate.obj")
        grid, _h, _cmin = compute_dense_sdf(v, i, bmin, bmax, 16)
        assert grid.shape == (16, 16, 16)
    except Exception as e:
        pytest.fail(f"Degenerate mesh caused crash: {e}")
    os.remove("degenerate.obj")

def test_extreme_resolution() -> None:
    """Verify memory handling at higher resolutions."""
    res = 128
    grid = np.zeros((res, res, res), dtype=np.float32)
    start = time.time()
    dag = compress_to_dag(grid, res)
    end = time.time()
    assert len(dag) > 0
    print(f"Deduplication for {res}^3 took {end - start:.2f}s")

# --- 3. Performance Benchmarks ---

def benchmark_fsm_scaling() -> None:
    resolutions = [32, 64, 128]
    create_test_obj("bench.obj", "cube")
    v, i, bmin, bmax = MeshIngestor.load_obj("bench.obj")
    
    print("\nFSM Performance Scaling (Agnostic & Cubic):")
    for res in resolutions:
        start = time.time()
        compute_dense_sdf(v, i, bmin, bmax, res)
        duration = time.time() - start
        print(f"  {res:3d}^3: {duration:6.3f}s ({(res**3)/duration/1e6:.2f} MVoxels/s)")
    os.remove("bench.obj")

# --- 4. Format Verification ---

def test_serialization_integrity() -> None:
    res = 16
    dag = np.arange(10, dtype=np.uint64)
    bmin = np.array([0, 0, 0], dtype=np.float32)
    write_gmdag("test.gmdag", dag, res, bmin, 0.1)
    
    with open("test.gmdag", "rb") as f:
        magic = f.read(4)
        assert magic == b"GDAG"
        version = struct.unpack("<I", f.read(4))[0]
        assert version == 1
        read_res = struct.unpack("<I", f.read(4))[0]
        assert read_res == res
    os.remove("test.gmdag")


def test_write_gmdag_rejects_empty_dag(tmp_path: Path) -> None:
    target = tmp_path / "empty.gmdag"

    with pytest.raises(ValueError, match="at least one node"):
        write_gmdag(target, np.array([], dtype=np.uint64), 16, np.zeros(3, dtype=np.float32), 0.1)


def test_write_gmdag_rejects_nonpositive_voxel_size(tmp_path: Path) -> None:
    target = tmp_path / "bad_voxel_size.gmdag"

    with pytest.raises(ValueError, match="voxel_size must be positive"):
        write_gmdag(target, np.array([1], dtype=np.uint64), 16, np.zeros(3, dtype=np.float32), 0.0)


def test_write_gmdag_rejects_nonfinite_bbox_min(tmp_path: Path) -> None:
    target = tmp_path / "bad_bbox.gmdag"

    with pytest.raises(ValueError, match="bbox_min must contain only finite values"):
        write_gmdag(
            target,
            np.array([1], dtype=np.uint64),
            16,
            np.array([0.0, np.nan, 1.0], dtype=np.float32),
            0.25,
        )


def test_write_gmdag_is_deterministic(tmp_path: Path) -> None:
    dag = np.array([1, 2, 3], dtype=np.uint64)
    bmin = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    first = tmp_path / "first.gmdag"
    second = tmp_path / "second.gmdag"

    write_gmdag(first, dag, 16, bmin, 0.25)
    write_gmdag(second, dag, 16, bmin, 0.25)

    assert first.read_bytes() == second.read_bytes()


def test_native_compiler_is_deterministic_for_fixed_fixture(tmp_path: Path) -> None:
    compiler = _resolve_native_compiler()
    if compiler is None:
        pytest.skip("Native voxel-dag compiler executable is not available")

    source = tmp_path / "cube.obj"
    create_test_obj(source, "cube")
    first = tmp_path / "native_first.gmdag"
    second = tmp_path / "native_second.gmdag"

    env = os.environ.copy()
    path_entries = [str(compiler.parent)]
    sibling_release = compiler.parents[1] / "Release"
    if sibling_release.exists():
        path_entries.append(str(sibling_release))
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([existing_path] if existing_path else []))

    first_run = subprocess.run(
        [str(compiler), "--input", str(source), "--output", str(first), "--resolution", "32"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert first_run.returncode == 0, first_run.stderr or first_run.stdout

    second_run = subprocess.run(
        [str(compiler), "--input", str(source), "--output", str(second), "--resolution", "32"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert second_run.returncode == 0, second_run.stderr or second_run.stdout

    assert first.read_bytes() == second.read_bytes()


def test_native_compiler_matches_python_header_contract_for_cube_fixture(tmp_path: Path) -> None:
    compiler = _resolve_native_compiler()
    if compiler is None:
        pytest.skip("Native voxel-dag compiler executable is not available")

    source = tmp_path / "cube.obj"
    create_test_obj(source, "cube")
    native_output = tmp_path / "native.gmdag"
    python_output = tmp_path / "python.gmdag"
    resolution = 32

    env = os.environ.copy()
    path_entries = [str(compiler.parent)]
    sibling_release = compiler.parents[1] / "Release"
    if sibling_release.exists():
        path_entries.append(str(sibling_release))
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_entries + ([existing_path] if existing_path else []))

    native_run = subprocess.run(
        [str(compiler), "--input", str(source), "--output", str(native_output), "--resolution", str(resolution)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert native_run.returncode == 0, native_run.stderr or native_run.stdout

    vertices, indices, bbox_min, bbox_max = MeshIngestor.load_obj(source)
    grid, voxel_size, cube_min = compute_dense_sdf(
        vertices,
        indices,
        bbox_min,
        bbox_max,
        resolution,
        padding=1.0,
    )
    write_gmdag(python_output, compress_to_dag(grid, resolution), resolution, cube_min, voxel_size)

    native_header = _read_gmdag_header(native_output)
    python_header = _read_gmdag_header(python_output)

    assert native_header[0] == python_header[0] == b"GDAG"
    assert native_header[1] == python_header[1] == 1
    assert native_header[2] == python_header[2] == resolution
    assert native_header[3] == pytest.approx(python_header[3], rel=0.0, abs=1e-6)
    assert native_header[4] == pytest.approx(python_header[4], rel=0.0, abs=1e-6)
    assert native_header[5] > 0
    assert python_header[5] > 0

if __name__ == "__main__":
    print("Running TopoNav Voxel-DAG Comprehensive Test Suite...")
    benchmark_fsm_scaling()
