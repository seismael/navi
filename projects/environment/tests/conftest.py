from __future__ import annotations

from pathlib import Path

import pytest
from _pytest.tmpdir import TempPathFactory


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest tmp_path output inside the project test tree on Windows.

    Repo-root `uv run --project ... pytest` invocations resolve relative basetemp
    paths against the shell cwd, while this machine's system pytest temp root is
    not writable. Rebuild the temp-path factory against a project-local scratch
    directory so test runs are stable regardless of launch location.
    """
    basetemp = Path(__file__).resolve().parent / ".pytest_tmp"
    basetemp.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(basetemp)
    config._tmp_path_factory = TempPathFactory.from_config(config, _ispytest=True)
