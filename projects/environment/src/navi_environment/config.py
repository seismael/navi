"""Configuration for the Environment service."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: list[str] = ["EnvironmentConfig"]


def derive_default_gmdag_resolution() -> int:
    """Return a compile profile aligned with the active observation contract."""
    try:
        azimuth_bins = int(os.environ.get("NAVI_AZIMUTH_BINS", "256"))
        elevation_bins = int(os.environ.get("NAVI_ELEVATION_BINS", "48"))
    except ValueError:
        azimuth_bins = 256
        elevation_bins = 48

    target = max(512, azimuth_bins * 2, elevation_bins * 8)
    resolution = 1
    while resolution < target:
        resolution <<= 1
    return resolution


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
    return Path(".env")  # fallback


def find_default_gmdag() -> Path:
    """Search upwards for the first canonical compiled corpus asset."""
    try:
        curr = Path(__file__).resolve().parent
        for _ in range(6):
            corpus_root = curr / "artifacts" / "gmdag" / "corpus"
            if corpus_root.exists():
                candidates = sorted(corpus_root.rglob("*.gmdag"))
                if candidates:
                    return candidates[0]
            curr = curr.parent
    except Exception:  # noqa: S110
        pass
    return Path()


class EnvironmentConfig(BaseSettings):
    """Environment service configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
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
    max_steps_per_episode: int = Field(default=2_000, validation_alias="NAVI_MAX_STEPS_PER_EPISODE", gt=0)
    azimuth_bins: int = Field(default=256, validation_alias="NAVI_AZIMUTH_BINS", gt=0)
    elevation_bins: int = Field(default=48, validation_alias="NAVI_ELEVATION_BINS", gt=0)
    max_distance: float = Field(default=30.0, gt=0.0)
    physics_dt: float = Field(default=0.02, gt=0.0)
    steps_per_decision: int = Field(default=1, gt=0)
    drone_max_speed: float = Field(default=10.0, gt=0.0)
    drone_climb_rate: float = Field(default=2.0, gt=0.0)
    drone_strafe_speed: float = Field(default=3.0, gt=0.0)
    drone_yaw_rate: float = Field(default=3.0, gt=0.0)
    n_actors: int = Field(default=1, gt=0)
    training_mode: bool = False
    compute_overhead: bool = False
    backend: str = "sdfdag"
    gmdag_file: str = str(find_default_gmdag())
    sdf_max_steps: int = Field(default=256, gt=0)
    gmdag_resolution: int = Field(
        default=derive_default_gmdag_resolution(),
        validation_alias="NAVI_GMDAG_RESOLUTION",
        gt=0,
    )
    scene_episodes_per_scene: int = Field(default=16, validation_alias="NAVI_SCENE_EPISODES_PER_SCENE", gt=0)
    obstacle_clearance_reward_scale: float = Field(
        default=0.6,
        validation_alias="NAVI_OBSTACLE_CLEARANCE_REWARD_SCALE",
    )
    obstacle_clearance_window: float = Field(
        default=1.5,
        validation_alias="NAVI_OBSTACLE_CLEARANCE_WINDOW",
        gt=0.0,
    )
    starvation_ratio_threshold: float = Field(
        default=0.8,
        validation_alias="NAVI_STARVATION_RATIO_THRESHOLD",
    )
    starvation_penalty_scale: float = Field(
        default=1.5,
        validation_alias="NAVI_STARVATION_PENALTY_SCALE",
    )
    proximity_distance_threshold: float = Field(
        default=1.0,
        validation_alias="NAVI_PROXIMITY_DISTANCE_THRESHOLD",
        gt=0.0,
    )
    proximity_penalty_scale: float = Field(
        default=0.8,
        validation_alias="NAVI_PROXIMITY_PENALTY_SCALE",
    )
    scene_pool: tuple[str, ...] = ()
