"""Thin orchestration wrapper around the internal `projects/voxel-dag` compiler."""

from __future__ import annotations

import importlib
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "GmDagAsset",
    "GmDagCompileResult",
    "GmDagRuntimeStatus",
    "compile_gmdag_world",
    "load_gmdag_asset",
    "probe_sdfdag_runtime",
]

_HEADER_STRUCT = struct.Struct("<4sIIffffI")


def _validate_gmdag_dag_layout(nodes: np.ndarray) -> None:
    node_count = int(nodes.shape[0])
    visited: set[int] = set()
    active_stack: set[int] = set()

    def visit(node_index: int) -> None:
        if node_index in visited:
            return
        if node_index in active_stack:
            msg = f"Invalid gmdag DAG layout: cycle detected at node index {node_index}"
            raise RuntimeError(msg)
        if node_index < 0 or node_index >= node_count:
            msg = f"Invalid gmdag DAG layout: node index out of range: {node_index}"
            raise RuntimeError(msg)

        active_stack.add(node_index)
        node_word = int(nodes[node_index])
        if (node_word >> 63) == 0:
            mask = (node_word >> 55) & 0xFF
            if mask == 0:
                msg = f"Invalid gmdag DAG layout: internal node {node_index} has empty child mask"
                raise RuntimeError(msg)
            child_base = node_word & 0xFFFFFFFF
            child_count = int(mask.bit_count())
            if child_base >= node_count:
                msg = (
                    "Invalid gmdag DAG layout: child pointer table starts beyond payload "
                    f"at index {child_base}"
                )
                raise RuntimeError(msg)
            if child_base + child_count > node_count:
                msg = (
                    "Invalid gmdag DAG layout: child pointer table exceeds payload bounds "
                    f"for node {node_index}"
                )
                raise RuntimeError(msg)
            for offset in range(child_count):
                child_index = int(nodes[child_base + offset])
                if child_index < 0 or child_index >= node_count:
                    msg = (
                        "Invalid gmdag DAG layout: child reference out of range "
                        f"({child_index}) from node {node_index}"
                    )
                    raise RuntimeError(msg)
                visit(child_index)

        active_stack.remove(node_index)
        visited.add(node_index)

    visit(0)


@dataclass(frozen=True, slots=True)
class GmDagAsset:
    """Loaded `.gmdag` asset metadata and node payload."""

    path: Path
    version: int
    resolution: int
    bbox_min: tuple[float, float, float]
    bbox_max: tuple[float, float, float]
    voxel_size: float
    nodes: np.ndarray


@dataclass(frozen=True, slots=True)
class GmDagCompileResult:
    """Outcome of invoking the internal voxel-dag compiler."""

    source_path: Path
    output_path: Path
    resolution: int
    command: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GmDagRuntimeStatus:
    """Readiness status for the canonical SDF/DAG compiler and runtime stack."""

    compiler_ready: bool
    torch_ready: bool
    cuda_ready: bool
    torch_sdf_ready: bool
    asset_loaded: bool
    compiler_path: Path | None
    gmdag_path: Path | None
    resolution: int | None
    node_count: int | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    issues: tuple[str, ...]


def probe_sdfdag_runtime(
    gmdag_path: Path | None = None,
    *,
    validate_asset_layout: bool = True,
) -> GmDagRuntimeStatus:
    """Check compiler/runtime readiness and optionally inspect a `.gmdag` asset."""
    issues: list[str] = []

    compiler_path: Path | None = None
    try:
        compiler_path = _resolve_voxel_dag_executable()
        compiler_ready = True
    except RuntimeError as exc:
        compiler_ready = False
        issues.append(str(exc))

    torch_module = None
    try:
        torch_module = importlib.import_module("torch")
        torch_ready = True
    except ImportError as exc:
        torch_ready = False
        cuda_ready = False
        issues.append(f"torch import failed: {exc}")
    else:
        cuda_ready = bool(torch_module.cuda.is_available())
        if not cuda_ready:
            issues.append("CUDA is not available. Canonical sdfdag execution requires CUDA.")

    try:
        importlib.import_module("torch_sdf")
        torch_sdf_ready = True
    except ImportError as exc:
        torch_sdf_ready = False
        issues.append(f"torch_sdf import failed: {exc}")

    asset_loaded = False
    asset_path: Path | None = None
    resolution: int | None = None
    node_count: int | None = None
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None
    if gmdag_path is not None:
        asset_path = gmdag_path.expanduser().resolve()
        try:
            asset = load_gmdag_asset(asset_path, validate_layout=validate_asset_layout)
        except (FileNotFoundError, RuntimeError) as exc:
            issues.append(str(exc))
        else:
            asset_loaded = True
            resolution = asset.resolution
            node_count = int(asset.nodes.shape[0])
            bbox_min = asset.bbox_min
            bbox_max = asset.bbox_max

    return GmDagRuntimeStatus(
        compiler_ready=compiler_ready,
        torch_ready=torch_ready,
        cuda_ready=cuda_ready,
        torch_sdf_ready=torch_sdf_ready,
        asset_loaded=asset_loaded,
        compiler_path=compiler_path,
        gmdag_path=asset_path,
        resolution=resolution,
        node_count=node_count,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        issues=tuple(issues),
    )


def load_gmdag_asset(path: Path, *, validate_layout: bool = True) -> GmDagAsset:
    """Load a `.gmdag` binary into metadata plus a contiguous node array."""
    path = path.expanduser().resolve()
    if not path.exists():
        msg = f"gmdag asset does not exist: {path}"
        raise FileNotFoundError(msg)

    with path.open("rb") as handle:
        header_bytes = handle.read(_HEADER_STRUCT.size)
        if len(header_bytes) != _HEADER_STRUCT.size:
            msg = f"Invalid gmdag header in {path}"
            raise RuntimeError(msg)

        magic, version, resolution, bmin_x, bmin_y, bmin_z, voxel_size, node_count = (
            _HEADER_STRUCT.unpack(header_bytes)
        )
        if magic != b"GDAG":
            msg = f"Unsupported gmdag magic in {path}: {magic!r}"
            raise RuntimeError(msg)
        if version != 1:
            msg = f"Unsupported gmdag version in {path}: {version}"
            raise RuntimeError(msg)
        if resolution <= 0:
            msg = f"Invalid gmdag resolution in {path}: {resolution}"
            raise RuntimeError(msg)
        if not np.isfinite(voxel_size) or voxel_size <= 0.0:
            msg = f"Invalid gmdag voxel size in {path}: {voxel_size}"
            raise RuntimeError(msg)
        if node_count <= 0:
            msg = f"Invalid gmdag node count in {path}: {node_count}"
            raise RuntimeError(msg)
        bbox_values = (bmin_x, bmin_y, bmin_z)
        if not all(np.isfinite(value) for value in bbox_values):
            msg = f"Invalid gmdag bounds in {path}: non-finite bbox_min values detected"
            raise RuntimeError(msg)

        nodes = np.fromfile(handle, dtype=np.uint64, count=node_count)
        if nodes.shape[0] != node_count:
            msg = (
                f"Invalid gmdag node payload in {path}: expected {node_count} nodes, "
                f"found {nodes.shape[0]}"
            )
            raise RuntimeError(msg)
        trailing_bytes = handle.read(1)
        if trailing_bytes:
            msg = f"Invalid gmdag payload in {path}: trailing bytes detected"
            raise RuntimeError(msg)

    if validate_layout:
        _validate_gmdag_dag_layout(nodes)

    bbox_min = (float(bmin_x), float(bmin_y), float(bmin_z))
    extent = float(voxel_size) * int(resolution)
    bbox_max = (
        bbox_min[0] + extent,
        bbox_min[1] + extent,
        bbox_min[2] + extent,
    )
    return GmDagAsset(
        path=path,
        version=int(version),
        resolution=int(resolution),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        voxel_size=float(voxel_size),
        nodes=np.ascontiguousarray(nodes),
    )


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve Navi repository root from integration helper"
    raise RuntimeError(msg)


def _resolve_voxel_dag_executable() -> Path:
    repo_root = _repo_root()
    project_root = repo_root / "projects" / "voxel-dag"

    candidates = [
        project_root / "build-local" / "Release" / "voxel-dag.exe",
        project_root / "build-local" / "voxel-dag.exe",
        project_root / "build" / "Release" / "voxel-dag.exe",
        project_root / "build" / "voxel-dag.exe",
        project_root / "build" / "voxel-dag",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    msg = (
        "voxel-dag compiler executable not found. Build the internal project first: "
        "projects/voxel-dag"
    )
    raise RuntimeError(msg)


def compile_gmdag_world(
    *,
    source_path: Path,
    output_path: Path,
    resolution: int,
) -> GmDagCompileResult:
    """Compile *source_path* into a `.gmdag` cache via the internal compiler."""
    if resolution <= 0:
        msg = "resolution must be a positive integer"
        raise ValueError(msg)

    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not source_path.exists():
        msg = f"Source mesh does not exist: {source_path}"
        raise FileNotFoundError(msg)

    compiler = _resolve_voxel_dag_executable()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = (
        str(compiler),
        "--input",
        str(source_path),
        "--output",
        str(output_path),
        "--resolution",
        str(resolution),
    )
    compiler_dir = compiler.parent
    assimp_release_dir = compiler_dir.parent / "_deps" / "assimp-build" / "bin" / "Release"
    env = os.environ.copy()
    path_parts = [str(compiler_dir)]
    if assimp_release_dir.exists():
        path_parts.append(str(assimp_release_dir))
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(path_parts + ([existing_path] if existing_path else []))

    # The command is assembled from validated local paths plus an integer resolution.
    subprocess.run(command, check=True, env=env)  # noqa: S603
    return GmDagCompileResult(
        source_path=source_path,
        output_path=output_path,
        resolution=resolution,
        command=command,
    )
