from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temp output inside the contracts test tree on Windows."""
    basetemp = Path(__file__).resolve().parent / ".pytest_tmp"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(basetemp)
    config._tmp_path_factory = TempPathFactory.from_config(config, _ispytest=True)
