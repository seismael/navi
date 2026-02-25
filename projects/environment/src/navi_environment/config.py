"""Configuration for the Environment service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: list[str] = ["EnvironmentConfig"]

def find_root_env() -> Path:
    """Search upwards for the root .env file."""
    try:
        curr = Path(__file__).resolve().parent
        for _ in range(6):
            target = curr / ".env"
            if target.exists():
                return target
            curr = curr.parent
    except Exception:  # noqa: S110
        pass
    return Path(".env") # fallback

class EnvironmentConfig(BaseSettings):
    """Environment service configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Robust Fallback
    pub_address: str = Field(
        default="tcp://*:5559",
        validation_alias="NAVI_ENV_PUB_ADDRESS",
    )
    rep_address: str = Field(
        default="tcp://*:5560",
        validation_alias="NAVI_ENV_REP_ADDRESS",
    )
    action_sub_address: str = Field(
        default="tcp://localhost:5557",
        validation_alias="NAVI_ACTOR_PUB_ADDRESS",
    )

    mode: str = "step"
    world_source: str = "procedural"
    world_file: str = ""
    generator: str = "arena"
    window_radius: int = 2
    lookahead_margin: int = 8
    seed: int = 42
    chunk_size: int = 16
    barrier_distance: float = 1.0
    collision_probe_radius: float = 1.5
    max_steps_per_episode: int = 50_000
    azimuth_bins: int = Field(default=128, validation_alias="NAVI_AZIMUTH_BINS")
    elevation_bins: int = Field(default=24, validation_alias="NAVI_ELEVATION_BINS")
    max_distance: float = 30.0
    physics_dt: float = 0.02
    steps_per_decision: int = 1
    drone_max_speed: float = 10.0
    drone_climb_rate: float = 2.0
    drone_strafe_speed: float = 3.0
    drone_yaw_rate: float = 3.0
    n_actors: int = 1
    training_mode: bool = False
    compute_overhead: bool = False
    backend: str = "voxel"
    habitat_scene: str = ""
    habitat_dataset_config: str = ""
    habitat_rgb_resolution: tuple[int, int] = (480, 640)
    scene_pool: tuple[str, ...] = ()
