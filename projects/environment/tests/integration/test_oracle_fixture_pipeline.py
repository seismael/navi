"""Golden-path validation from compiled fixture to materialized auditor-facing signal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from navi_environment.backends.sdfdag_backend import SdfDagBackend
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.voxel_dag import probe_sdfdag_runtime


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve repo root for oracle fixture integration test"
    raise RuntimeError(msg)


def _center_forward_azimuth(array: np.ndarray) -> np.ndarray:
    """Mirror the auditor's forward-centering seam alignment without GUI deps."""
    return np.roll(array, shift=array.shape[0] // 2, axis=0)


def _load_voxel_dag_compiler() -> Any:
    compiler_path = _repo_root() / "projects" / "voxel-dag" / "voxel_dag" / "compiler.py"
    spec = importlib.util.spec_from_file_location("navi_test_voxel_dag_compiler", compiler_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load voxel-dag compiler from {compiler_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_forward_wall_obj(path: Path) -> None:
    vertices = [
        (1.5, 0.0, -1.0),
        (1.7, 0.0, -1.0),
        (1.7, 2.0, -1.0),
        (1.5, 2.0, -1.0),
        (1.5, 0.0, 1.0),
        (1.7, 0.0, 1.0),
        (1.7, 2.0, 1.0),
        (1.5, 2.0, 1.0),
    ]
    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7),
        (4, 1, 5), (4, 5, 8),
    ]
    lines = [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a} {b} {c}" for a, b, c in faces)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compile_forward_wall_gmdag(output_dir: Path) -> Path:
    compiler = _load_voxel_dag_compiler()
    mesh_ingestor = cast("Any", compiler.MeshIngestor)
    compute_dense_sdf = cast("Any", compiler.compute_dense_sdf)
    compress_to_dag = cast("Any", compiler.compress_to_dag)
    write_gmdag = cast("Any", compiler.write_gmdag)

    source = output_dir / "forward_wall.obj"
    output = output_dir / "forward_wall.gmdag"
    _write_forward_wall_obj(source)

    vertices, indices, bbox_min, bbox_max = mesh_ingestor.load_obj(str(source))
    grid, voxel_size, cube_min = compute_dense_sdf(vertices, indices, bbox_min, bbox_max, 64, padding=0.0)
    dag = compress_to_dag(grid, 64)
    write_gmdag(output, dag, 64, cube_min, voxel_size)
    return output


def test_compiled_forward_wall_materializes_centered_auditor_signal(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gmdag_path = _compile_forward_wall_gmdag(tmp_path)
    runtime_status = probe_sdfdag_runtime(gmdag_path)
    if runtime_status.issues:
        pytest.skip("Compiled sdfdag runtime unavailable: " + "; ".join(runtime_status.issues))

    backend = SdfDagBackend(
        EnvironmentConfig(
            backend="sdfdag",
            gmdag_file=str(gmdag_path),
            n_actors=1,
            azimuth_bins=64,
            elevation_bins=16,
            max_distance=10.0,
            training_mode=True,
            sdfdag_torch_compile=False,
        )
    )
    try:
        initial_pose = torch.tensor([0.0, 1.0, 0.0], device=backend._device, dtype=torch.float32)
        backend._spawn_positions[0].copy_(initial_pose)
        observation_tensor, published = backend.reset_tensor(episode_id=1, actor_id=0, materialize=True)
        assert published is not None

        centered_depth = _center_forward_azimuth(published.depth[0])
        centered_valid = _center_forward_azimuth(published.valid_mask[0])

        assert tuple(observation_tensor.shape) == (3, 64, 16)
        assert published.matrix_shape == (64, 16)
        assert published.depth.shape == (1, 64, 16)
        assert published.delta_depth.shape == (1, 64, 16)
        assert centered_depth.shape == (64, 16)
        assert centered_valid.shape == (64, 16)
        assert np.all(np.isfinite(centered_depth))
        assert np.all(centered_depth >= 0.0)
        assert np.all(centered_depth <= 1.0)
    finally:
        backend.close()
