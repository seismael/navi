"""Configuration for the Auditor service."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: list[str] = ["AuditorConfig"]

_SettingsBase = cast(type[BaseModel], BaseSettings)

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

class AuditorConfig(_SettingsBase):
    """Auditor service configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
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

    @property
    def sub_addresses(self) -> tuple[str, ...]:
        return (self.matrix_sub_address, self.actor_sub_address)
