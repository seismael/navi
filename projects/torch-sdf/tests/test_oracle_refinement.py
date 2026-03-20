"""Refined oracle validation for the complete flow from mesh to raycasting.

This test uses the 'oracle house' mesh fixture to validate the mathematical 
consistency of the entire pipeline: mesh loading, SDF computation, DAG 
compression, and CUDA-accelerated raycasting.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pytest
import torch
import torch_sdf


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    raise RuntimeError("Could not resolve repo root")


def _setup_sys_path() -> None:
    root = _repo_root()
    paths = [
        root / "projects" / "contracts" / "src",
        root / "projects" / "environment" / "src",
        root / "projects" / "voxel-dag",
    ]
    for p in paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _require_voxel_dag() -> Any:
    _setup_sys_path()
    import voxel_dag.compiler as compiler
    return compiler


def _require_oracle_house() -> Any:
    _setup_sys_path()
    import navi_contracts.testing.oracle_house as oracle_house
    return oracle_house


def _require_sdfdag_backend() -> Any:
    _setup_sys_path()
    import navi_environment.backends.sdfdag_backend as backend
    return backend


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_oracle_house_complete_flow_is_mathematically_consistent(tmp_path: Path) -> None:
    """Validate that the full pipeline produces mathematically sound results for the oracle house."""
    _setup_sys_path()
    v_dag = _require_voxel_dag()
    oracle_mod = _require_oracle_house()
    backend_mod = _require_sdfdag_backend()
    
    # 1. Generate Oracle House Mesh
    source = tmp_path / "oracle_house.obj"
    oracle_mod.write_square_house_obj(source)
    
    # 2. Compile to .gmdag
    resolution = 64
    vertices, indices, bbox_min, bbox_max = v_dag.MeshIngestor.load_obj(source)
    
    # Use refactored OOD classes
    computer = v_dag.EikonalSdfComputer()
    grid, voxel_size, cube_min = computer.compute(
        vertices, indices, bbox_min, bbox_max, resolution, padding=0.0
    )
    
    # Create a semantic grid matching the oracle's "wall" ID (1)
    semantic_grid = np.ones_like(grid, dtype=np.int32)
    
    compressor = v_dag.SvoDagCompressor()
    dag_np = compressor.compress(grid, resolution, cell_size=voxel_size, semantic_grid=semantic_grid)
    dag_torch = torch.from_numpy(dag_np.view(np.int64)).cuda()
    
    cube_max = cube_min + (resolution * voxel_size)
    
    # 3. Setup Raycasting Pose
    # Origin at center of house: x=0, z=0, y=1.25 (middle of 2.5m height)
    origin_val = [0.0, 1.25, 0.0]
    max_distance = 10.0
    az_bins = 12
    el_bins = 3
    
    # Get directions (spherical distribution)
    dirs_np = backend_mod.build_spherical_ray_directions(az_bins, el_bins)
    dirs_torch = torch.from_numpy(dirs_np).cuda().unsqueeze(0)
    origins_torch = torch.tensor([origin_val], device="cuda", dtype=torch.float32).repeat(1, dirs_torch.shape[1], 1)
    
    out_d = torch.full((1, dirs_torch.shape[1]), -1.0, device="cuda", dtype=torch.float32)
    out_s = torch.full((1, dirs_torch.shape[1]), -1, device="cuda", dtype=torch.int32)
    
    # 4. Execute Raycast
    torch_sdf.cast_rays(
        dag_torch,
        origins_torch,
        dirs_torch,
        out_d,
        out_s,
        128, # max_steps
        max_distance,
        cube_min.tolist(),
        cube_max.tolist(),
        resolution,
    )
    
    # 5. Logical Validation
    actual_depth = out_d[0].detach().cpu().numpy()
    actual_semantic = out_s[0].detach().cpu().numpy()
    
    # All rays should hit (it's a closed box)
    assert np.all(actual_semantic == 1), "Every ray should hit the wall/floor/ceiling (semantic 1)"
    
    # Metric distance checks:
    # Origin is at y=1.25. Ceiling is at y=2.5. Distance UP is 1.25.
    # Floor is at y=0.0. Distance DOWN is 1.25.
    # Horizontal distance to walls is 2.0.
    
    # Elevation bins: 0=DOWN, 1=HORIZONTAL, 2=UP (typical for build_spherical_ray_directions)
    actual_depth_2d = actual_depth.reshape(az_bins, el_bins)
    
    # Horizontal rays (elevation index 1)
    # They should hit walls at distance >= 2.0
    horizontal_depths = actual_depth_2d[:, 1]
    assert np.all(horizontal_depths >= 1.9), "Horizontal rays should hit walls at ~2.0m"
    assert np.all(horizontal_depths <= 2.9), "Horizontal rays should hit walls within house bounds"
    
    # Vertical rays (elevation index 0 and 2)
    # At 45 degrees, distance to floor/ceiling is 1.25 / cos(45) = 1.76
    vertical_depths = actual_depth_2d[:, [0, 2]]
    assert np.all(vertical_depths >= 1.1), "Vertical rays should hit ceiling/floor at >= 1.25m"
    assert np.all(vertical_depths <= 1.9), "Vertical rays at 45 deg should hit ceiling/floor at ~1.76m"

    print("\n[PASS] Pipeline consistency verified with oracle house.")
