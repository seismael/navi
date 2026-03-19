"""Configuration for the Auditor service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: list[str] = ["AuditorConfig"]


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


class AuditorConfig(BaseSettings):
    """Auditor service configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Robust Fallback: Defaults match the NAVI standard
    matrix_sub_address: str = Field(
        default="tcp://localhost:5559",
        validation_alias="NAVI_ENV_PUB_ADDRESS",
    )
    actor_sub_address: str = Field(
        default="tcp://localhost:5557",
        validation_alias="NAVI_ACTOR_PUB_ADDRESS",
    )
    actor_control_address: str = Field(
        default="tcp://localhost:5561",
        validation_alias="NAVI_ACTOR_CONTROL_ADDRESS",
    )
    step_endpoint: str = Field(
        default="tcp://localhost:5560",
        validation_alias="NAVI_ENV_REP_ADDRESS",
    )
    output_path: str = Field(
        default="session.zarr",
        validation_alias="NAVI_AUDITOR_OUTPUT",
    )
    pub_address: str = Field(
        default="tcp://*:5558",
        validation_alias="NAVI_AUDITOR_PUB_ADDRESS",
    )
    observation_max_distance_m: float = Field(
        default=30.0,
        validation_alias="NAVI_MAX_DISTANCE",
    )

    @property
    def sub_addresses(self) -> tuple[str, ...]:
        addresses: list[str] = []
        for addr in (self.matrix_sub_address, self.actor_sub_address):
            if addr and addr not in addresses:
                addresses.append(addr)
        return tuple(addresses)
