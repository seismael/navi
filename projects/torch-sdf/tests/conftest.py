from __future__ import annotations

from pathlib import Path

from _pytest.tmpdir import TempPathFactory
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest tmp_path output inside the project test tree on Windows."""
    basetemp = Path(__file__).resolve().parent / ".pytest_tmp"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(basetemp)
    config._tmp_path_factory = TempPathFactory.from_config(config, _ispytest=True)
