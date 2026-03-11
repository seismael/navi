"""navi-environment - Layer 1: The Environment."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__: list[str] = [
    "DatasetAdapter",
    "EquirectangularDatasetAdapter",
    "EnvironmentConfig",
    "EnvironmentServer",
    "habitat_camera_transform_spec",
    "materialize_distance_matrix",
    "MjxBackendInfo",
    "MjxEnvironment",
    "SdfDagBackend",
    "SdfDagPerfSnapshot",
    "SimulatorBackend",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "DatasetAdapter": ("navi_environment.backends.adapter", "DatasetAdapter"),
    "EquirectangularDatasetAdapter": ("navi_environment.backends.adapter", "EquirectangularDatasetAdapter"),
    "habitat_camera_transform_spec": ("navi_environment.backends.adapter", "habitat_camera_transform_spec"),
    "materialize_distance_matrix": ("navi_environment.backends.adapter", "materialize_distance_matrix"),
    "EnvironmentConfig": ("navi_environment.config", "EnvironmentConfig"),
    "EnvironmentServer": ("navi_environment.server", "EnvironmentServer"),
    "MjxBackendInfo": ("navi_environment.mjx_env", "MjxBackendInfo"),
    "MjxEnvironment": ("navi_environment.mjx_env", "MjxEnvironment"),
    "SdfDagBackend": ("navi_environment.backends.sdfdag_backend", "SdfDagBackend"),
    "SdfDagPerfSnapshot": ("navi_environment.backends.sdfdag_backend", "SdfDagPerfSnapshot"),
    "SimulatorBackend": ("navi_environment.backends.base", "SimulatorBackend"),
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
    from navi_environment.backends.adapter import (
        DatasetAdapter,
        EquirectangularDatasetAdapter,
        habitat_camera_transform_spec,
        materialize_distance_matrix,
    )
    from navi_environment.backends.base import SimulatorBackend
    from navi_environment.backends.sdfdag_backend import SdfDagBackend, SdfDagPerfSnapshot
    from navi_environment.config import EnvironmentConfig
    from navi_environment.mjx_env import MjxBackendInfo, MjxEnvironment
    from navi_environment.server import EnvironmentServer
