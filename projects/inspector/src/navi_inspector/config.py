"""Configuration for the Inspector service."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__: list[str] = ["InspectorConfig"]


def _find_root_env() -> Path:
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
    return Path(".env")


class InspectorConfig(BaseSettings):
    """Inspector configuration, loadable from environment or .env."""

    model_config = SettingsConfigDict(
        env_file=_find_root_env(),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    default_resolution: int = Field(
        default=128,
        validation_alias="NAVI_INSPECTOR_DEFAULT_RESOLUTION",
    )
    export_resolution: int = Field(
        default=512,
        validation_alias="NAVI_INSPECTOR_EXPORT_RESOLUTION",
    )
    cache_dir: Path = Field(
        default=Path("artifacts/inspector/cache"),
        validation_alias="NAVI_INSPECTOR_CACHE_DIR",
    )
