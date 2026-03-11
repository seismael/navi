"""Pluggable simulator backends for the Environment."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__: list[str] = [
    "DatasetAdapter",
    "EquirectangularDatasetAdapter",
    "habitat_camera_transform_spec",
    "materialize_distance_matrix",
    "SdfDagBackend",
    "SimulatorBackend",
]

_EXPORT_MAP: dict[str, tuple[str, str]] = {
    "DatasetAdapter": ("navi_environment.backends.adapter", "DatasetAdapter"),
    "EquirectangularDatasetAdapter": ("navi_environment.backends.adapter", "EquirectangularDatasetAdapter"),
    "habitat_camera_transform_spec": ("navi_environment.backends.adapter", "habitat_camera_transform_spec"),
    "materialize_distance_matrix": ("navi_environment.backends.adapter", "materialize_distance_matrix"),
    "SdfDagBackend": ("navi_environment.backends.sdfdag_backend", "SdfDagBackend"),
    "SimulatorBackend": ("navi_environment.backends.base", "SimulatorBackend"),
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
    from navi_environment.backends.adapter import (
        DatasetAdapter,
        EquirectangularDatasetAdapter,
        habitat_camera_transform_spec,
        materialize_distance_matrix,
    )
    from navi_environment.backends.base import SimulatorBackend
    from navi_environment.backends.sdfdag_backend import SdfDagBackend
