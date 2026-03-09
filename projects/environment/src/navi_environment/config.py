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
    max_steps_per_episode: int = 50_000
    azimuth_bins: int = Field(default=256, validation_alias="NAVI_AZIMUTH_BINS")
    elevation_bins: int = Field(default=48, validation_alias="NAVI_ELEVATION_BINS")
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
    backend: str = "sdfdag"
    gmdag_file: str = str(find_default_gmdag())
    sdf_max_steps: int = 256
    gmdag_resolution: int = Field(
        default=derive_default_gmdag_resolution(),
        validation_alias="NAVI_GMDAG_RESOLUTION",
    )
    scene_pool: tuple[str, ...] = ()
