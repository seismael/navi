"""navi-environment - Layer 1: The Environment."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__: list[str] = [
    "AbstractWorldGenerator",
    "DatasetAdapter",
    "DistanceMatrixBuilder",
    "DistancePruner",
    "EnvironmentConfig",
    "EnvironmentServer",
    "FileGenerator",
    "FrustumLoader",
    "HabitatAdapter",
    "LookAheadBuffer",
    "MazeGenerator",
    "MjxBackendInfo",
    "MjxEnvironment",
    "OcclusionCuller",
    "Open3DVoxelGenerator",
    "RaycastEngine",
    "SdfDagBackend",
    "SdfDagPerfSnapshot",
    "SimulatorBackend",
    "SlidingWindow",
    "SparseVoxelGrid",
    "VoxelBackend",
    "WedgeResult",
    "WorldCompileConfig",
    "WorldCompileResult",
    "WorldModelCompiler",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "AbstractWorldGenerator": ("navi_environment.generators.base", "AbstractWorldGenerator"),
    "DatasetAdapter": ("navi_environment.backends.adapter", "DatasetAdapter"),
    "DistanceMatrixBuilder": ("navi_environment.distance_matrix_v2", "DistanceMatrixBuilder"),
    "DistancePruner": ("navi_environment.pruning", "DistancePruner"),
    "EnvironmentConfig": ("navi_environment.config", "EnvironmentConfig"),
    "EnvironmentServer": ("navi_environment.server", "EnvironmentServer"),
    "FileGenerator": ("navi_environment.generators.file_loader", "FileGenerator"),
    "FrustumLoader": ("navi_environment.frustum", "FrustumLoader"),
    "HabitatAdapter": ("navi_environment.backends.habitat_adapter", "HabitatAdapter"),
    "LookAheadBuffer": ("navi_environment.lookahead", "LookAheadBuffer"),
    "MazeGenerator": ("navi_environment.generators.maze", "MazeGenerator"),
    "MjxBackendInfo": ("navi_environment.mjx_env", "MjxBackendInfo"),
    "MjxEnvironment": ("navi_environment.mjx_env", "MjxEnvironment"),
    "OcclusionCuller": ("navi_environment.pruning", "OcclusionCuller"),
    "Open3DVoxelGenerator": ("navi_environment.generators.open3d_voxel", "Open3DVoxelGenerator"),
    "RaycastEngine": ("navi_environment.raycast", "RaycastEngine"),
    "SdfDagBackend": ("navi_environment.backends.sdfdag_backend", "SdfDagBackend"),
    "SdfDagPerfSnapshot": ("navi_environment.backends.sdfdag_backend", "SdfDagPerfSnapshot"),
    "SimulatorBackend": ("navi_environment.backends.base", "SimulatorBackend"),
    "SlidingWindow": ("navi_environment.sliding_window", "SlidingWindow"),
    "SparseVoxelGrid": ("navi_environment.matrix", "SparseVoxelGrid"),
    "VoxelBackend": ("navi_environment.backends.voxel", "VoxelBackend"),
    "WedgeResult": ("navi_environment.sliding_window", "WedgeResult"),
    "WorldCompileConfig": ("navi_environment.transformers", "WorldCompileConfig"),
    "WorldCompileResult": ("navi_environment.transformers", "WorldCompileResult"),
    "WorldModelCompiler": ("navi_environment.transformers", "WorldModelCompiler"),
}


def __getattr__(name: str) -> Any:
    """Resolve exported symbols lazily to avoid import-time runtime coupling."""
    try:
        module_name, attr_name = _EXPORT_MAP[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    from navi_environment.backends.adapter import DatasetAdapter
    from navi_environment.backends.base import SimulatorBackend
    from navi_environment.backends.habitat_adapter import HabitatAdapter
    from navi_environment.backends.sdfdag_backend import SdfDagBackend, SdfDagPerfSnapshot
    from navi_environment.backends.voxel import VoxelBackend
    from navi_environment.config import EnvironmentConfig
    from navi_environment.distance_matrix_v2 import DistanceMatrixBuilder
    from navi_environment.frustum import FrustumLoader
    from navi_environment.generators.base import AbstractWorldGenerator
    from navi_environment.generators.file_loader import FileGenerator
    from navi_environment.generators.maze import MazeGenerator
    from navi_environment.generators.open3d_voxel import Open3DVoxelGenerator
    from navi_environment.lookahead import LookAheadBuffer
    from navi_environment.matrix import SparseVoxelGrid
    from navi_environment.mjx_env import MjxBackendInfo, MjxEnvironment
    from navi_environment.pruning import DistancePruner, OcclusionCuller
    from navi_environment.raycast import RaycastEngine
    from navi_environment.server import EnvironmentServer
    from navi_environment.sliding_window import SlidingWindow, WedgeResult
    from navi_environment.transformers import WorldCompileConfig, WorldCompileResult, WorldModelCompiler
