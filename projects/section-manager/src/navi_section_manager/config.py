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
    max_steps_per_episode: int = 50_000
    azimuth_bins: int = 256
    elevation_bins: int = 128
    max_distance: float = 30.0

    # Physics timestep (seconds).  Default 0.02 = 50 Hz.
    physics_dt: float = 0.02

    # Command-hold: repeat the last action for N physics ticks per
    # decision.  1 = every tick requires a new decision (default).
    steps_per_decision: int = 1

    # Drone speed — scales normalised steering [-1, 1] from the policy
    # into physical velocities.  These are deployment parameters;
    # re-training is NOT required when changing speed.
    drone_speed: float = 50.0         # max forward speed (m/s, ~180 km/h)
    drone_climb_rate: float = 3.0     # max vertical rate (m/s)
    drone_strafe_speed: float = 5.0   # max lateral speed (m/s)
    drone_yaw_rate: float = 3.0       # max yaw rate (rad/s, ~172°/s)

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
