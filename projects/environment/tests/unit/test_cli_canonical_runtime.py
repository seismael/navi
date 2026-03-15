"""Regression tests for canonical environment runtime defaults."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import typer
from pydantic import ValidationError
from typer.testing import CliRunner

from navi_environment.cli import _build_backend, app
from navi_environment.config import EnvironmentConfig

_RUNNER = CliRunner()


def test_environment_config_defaults_to_canonical_sdfdag() -> None:
    """Environment config should default to the canonical compiled runtime."""
    config = EnvironmentConfig()

    assert config.backend == "sdfdag"
    assert isinstance(config.gmdag_file, str)
    assert config.sdfdag_torch_compile is True


def test_build_backend_rejects_unknown_backend() -> None:
    """Unknown backend names must fail fast instead of falling back silently."""
    with pytest.raises(typer.Exit) as exc_info:
        _build_backend(EnvironmentConfig(backend="legacy"))

    assert exc_info.value.exit_code == 1


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_distance", 0.0),
        ("sdf_max_steps", 0),
        ("gmdag_resolution", 0),
    ],
)
def test_environment_config_rejects_nonpositive_runtime_parameters(field_name: str, value: float | int) -> None:
    with pytest.raises(ValidationError):
        EnvironmentConfig.model_validate({field_name: value})


def test_serve_defaults_to_canonical_sdfdag(monkeypatch: pytest.MonkeyPatch) -> None:
    """The serve command should launch on sdfdag without extra flags."""
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config: EnvironmentConfig, backend: object) -> None:
            captured["config"] = config
            captured["backend"] = backend

        def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    monkeypatch.setattr("navi_environment.cli._build_backend", lambda config: object())
    monkeypatch.setattr("navi_environment.cli.EnvironmentServer", FakeServer)

    result = _RUNNER.invoke(app, ["serve"])

    assert result.exit_code == 0
    config = cast("EnvironmentConfig", captured["config"])
    assert config.backend == "sdfdag"
    assert isinstance(Path(config.gmdag_file).name, str)
    assert captured["ran"] is True
