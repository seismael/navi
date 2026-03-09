from os import PathLike
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

class SettingsConfigDict(ConfigDict, total=False):
    env_file: str | PathLike[str]
    env_file_encoding: str

class BaseSettings(BaseModel):
    model_config: ClassVar[SettingsConfigDict]
    def __init__(self, **values: Any) -> None: ...
