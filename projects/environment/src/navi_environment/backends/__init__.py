"""Pluggable simulator backends for the Environment."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__: list[str] = [
    "DatasetAdapter",
    "HabitatAdapter",
    "MeshSceneBackend",
    "SdfDagBackend",
    "SimulatorBackend",
    "VoxelBackend",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "DatasetAdapter": ("navi_environment.backends.adapter", "DatasetAdapter"),
    "HabitatAdapter": ("navi_environment.backends.habitat_adapter", "HabitatAdapter"),
    "MeshSceneBackend": ("navi_environment.backends.mesh_backend", "MeshSceneBackend"),
    "SdfDagBackend": ("navi_environment.backends.sdfdag_backend", "SdfDagBackend"),
    "SimulatorBackend": ("navi_environment.backends.base", "SimulatorBackend"),
    "VoxelBackend": ("navi_environment.backends.voxel", "VoxelBackend"),
}


def __getattr__(name: str) -> Any:
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
    from navi_environment.backends.mesh_backend import MeshSceneBackend
    from navi_environment.backends.sdfdag_backend import SdfDagBackend
    from navi_environment.backends.voxel import VoxelBackend
