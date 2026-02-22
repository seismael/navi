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
    barrier_distance: float = 1.0
    collision_probe_radius: float = 1.5
    max_steps_per_episode: int = 1000
    azimuth_bins: int = 256
    elevation_bins: int = 128
    max_distance: float = 30.0

    # Multi-actor
    n_actors: int = 1

    # Skip overhead minimap computation (never used by policy)
    compute_overhead: bool = True

    # Backend selection — "voxel" (default) or "habitat"
    backend: str = "voxel"

    # Habitat-specific settings (ignored when backend != "habitat")
    habitat_scene: str = ""
    habitat_dataset_config: str = ""
    habitat_rgb_resolution: tuple[int, int] = (480, 640)
