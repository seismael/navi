"""Configuration for the Auditor service."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["AuditorConfig"]


@dataclass(frozen=True)
class AuditorConfig:
    """Auditor service configuration, loadable from TOML."""

    sub_addresses: tuple[str, ...] = ("tcp://localhost:5559", "tcp://localhost:5557")
    output_path: str = "session.zarr"
    pub_address: str = "tcp://*:5558"
