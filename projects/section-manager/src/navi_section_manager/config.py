"""Configuration for the Section Manager service."""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["SectionManagerConfig"]


@dataclass(frozen=True)
class SectionManagerConfig:
    """Section Manager service configuration, loadable from TOML."""

    pub_address: str = "tcp://*:5559"
    rep_address: str = "tcp://*:5560"
    action_sub_address: str = "tcp://localhost:5557"
    mode: str = "step"  # "step" or "async"
    world_source: str = "procedural"  # "procedural" or "file"
    world_file: str = ""
    generator: str = "arena"  # "arena", "city", "maze", "open3d", or "rooms"
    window_radius: int = 2
    lookahead_margin: int = 8
    seed: int = 42
    chunk_size: int = 16
    barrier_distance: float = 0.0
    collision_probe_radius: float = 1.5
    azimuth_bins: int = 256
    elevation_bins: int = 128
    max_distance: float = 30.0
