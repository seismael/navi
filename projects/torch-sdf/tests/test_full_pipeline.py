import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
import torch
import torch_sdf


def _require_voxel_dag() -> tuple[Any, Any, Any]:
    compiler_path = (
        _repo_root() / "projects" / "voxel-dag" / "voxel_dag" / "compiler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "navi_test_voxel_dag_compiler", compiler_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load voxel-dag compiler from {compiler_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MeshIngestor, module.compute_dense_sdf, module.compress_to_dag


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve repo root for torch-sdf full pipeline tests"
    raise RuntimeError(msg)


def _load_oracle_house_writer() -> Callable[[str | Path], None]:
    helper_path = (
        _repo_root()
        / "projects"
        / "contracts"
        / "src"
        / "navi_contracts"
        / "testing"
        / "oracle_house.py"
    )
    spec = importlib.util.spec_from_file_location("navi_test_oracle_house", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load oracle fixture helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast("Callable[[str | Path], None]", module.write_square_house_obj)


def test_full_pipeline_integrity():
    """
    End-to-End Test:
    1. Generate complex geometry.
    2. Compile to .gmdag.
    3. Run raycasting and verify results.
    """
    MeshIngestor, compute_dense_sdf, compress_to_dag = _require_voxel_dag()
    MeshIngestor = cast(Any, MeshIngestor)
    compute_dense_sdf = cast(Any, compute_dense_sdf)
    compress_to_dag = cast(Any, compress_to_dag)
    # 1. Create a "Hallway" Mesh
    hallway_obj = "hallway.obj"
    with open(hallway_obj, "w") as f:
        f.write(
            "v -1 -1 0\n"
            "v 1 -1 0\n"
            "v 1 1 0\n"
            "v -1 1 0\n"
            "v -1 -1 20\n"
            "v 1 -1 20\n"
            "v 1 1 20\n"
            "v -1 1 20\n"
            "f 1 2 3\n"
            "f 1 3 4\n"
            "f 5 6 7\n"
            "f 5 7 8\n"
            "f 1 2 6\n"
            "f 1 6 5\n"
            "f 3 4 8\n"
            "f 3 8 7\n"
        )

    # 2. Compile to DAG
    res = 128
    v, i, bmin, bmax = MeshIngestor.load_obj(hallway_obj)
    grid, h, cmin = compute_dense_sdf(v, i, bmin, bmax, res)
    dag_np = compress_to_dag(grid, res)

    # 3. Transfer to GPU
    if not torch.cuda.is_available():
        os.remove(hallway_obj)
        pytest.skip("CUDA not available")

    dag_torch = torch.from_numpy(dag_np.view(np.int64)).cuda()

    # --- EDGE CASE 1: Ray Down the Hallway (Long Trace) ---
    origins = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda", dtype=torch.float32)
    dirs = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda", dtype=torch.float32)
    out_d = torch.zeros((1, 1), device="cuda")
    out_s = torch.zeros((1, 1), dtype=torch.int32, device="cuda")

    bbox_max = cmin + (res * h)

    torch_sdf.cast_rays(
        dag_torch,
        origins,
        dirs,
        out_d,
        out_s,
        128,
        30.0,
        cmin.tolist(),
        bbox_max.tolist(),
        res,
    )

    # Hit wall logic verification
    print(f"Distance detected: {out_d[0, 0].item():.2f}")
    assert out_d[0, 0].item() > 0

    os.remove(hallway_obj)
    print("\n[PASS] torch-sdf Hallway & Integrity Validated.")


def test_square_house_pipeline_reduces_forward_distance_after_motion(
    tmp_path: Path,
) -> None:
    """Compiled oracle house should report a shorter forward hit after forward motion."""
    MeshIngestor, compute_dense_sdf, compress_to_dag = _require_voxel_dag()
    MeshIngestor = cast(Any, MeshIngestor)
    compute_dense_sdf = cast(Any, compute_dense_sdf)
    compress_to_dag = cast(Any, compress_to_dag)
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    source = tmp_path / "square_house.obj"
    _load_oracle_house_writer()(source)

    vertices, indices, bbox_min, bbox_max = MeshIngestor.load_obj(str(source))
    grid, voxel_size, cube_min = compute_dense_sdf(
        vertices, indices, bbox_min, bbox_max, 64, padding=0.0
    )
    dag_np = compress_to_dag(grid, 64)
    dag_torch = torch.from_numpy(dag_np.view(np.int64)).cuda()
    cube_max = cube_min + (64 * voxel_size)

    origins = torch.tensor(
        [
            [[0.0, 1.0, 0.0]],
            [[0.75, 1.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    dirs = torch.tensor(
        [
            [[1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    out_d = torch.full((2, 1), -1.0, device="cuda", dtype=torch.float32)
    out_s = torch.full((2, 1), -1, device="cuda", dtype=torch.int32)

    torch_sdf.cast_rays(
        dag_torch,
        origins,
        dirs,
        out_d,
        out_s,
        128,
        10.0,
        cube_min.tolist(),
        cube_max.tolist(),
        64,
    )

    distances = out_d[:, 0].detach().cpu().numpy()
    assert np.all(distances > 0.0)
    assert float(distances[1]) < float(distances[0])
    assert float(distances[0] - distances[1]) > 0.35


def benchmark_throughput():
    """Verify performance under load (planned 10,000 SPS)."""
    if not torch.cuda.is_available():
        print("Skipping benchmark: CUDA not available")
        return

    # Mock massive batch (Topological Navigation Standard Load)
    batch_size = 64
    rays_per_actor = 3072
    total_rays = batch_size * rays_per_actor

    origins = torch.zeros((batch_size, rays_per_actor, 3), device="cuda")
    dirs = torch.zeros((batch_size, rays_per_actor, 3), device="cuda")
    dirs[..., 0] = 1.0  # All rays point X+

    dag = torch.zeros(1024, dtype=torch.int64, device="cuda")  # Empty space DAG
    out_d = torch.zeros((batch_size, rays_per_actor), device="cuda")
    out_s = torch.zeros((batch_size, rays_per_actor), dtype=torch.int32, device="cuda")

    # Warmup
    torch_sdf.cast_rays(
        dag, origins, dirs, out_d, out_s, 64, 30.0, [0, 0, 0], [1, 1, 1], 128
    )
    torch.cuda.synchronize()

    print(f"Measuring throughput for {total_rays:,} rays...")
    start = time.time()
    iters = 100
    for _ in range(iters):
        torch_sdf.cast_rays(
            dag, origins, dirs, out_d, out_s, 64, 30.0, [0, 0, 0], [1, 1, 1], 128
        )
    torch.cuda.synchronize()
    end = time.time()

    duration = end - start
    sps = (iters * batch_size) / duration
    print(f"[BENCHMARK] Throughput: {sps:.1f} Steps-Per-Second")
    print(f"[BENCHMARK] Compute Latency: {duration / iters * 1000:.3f} ms per step")


if __name__ == "__main__":
    test_full_pipeline_integrity()
    benchmark_throughput()
