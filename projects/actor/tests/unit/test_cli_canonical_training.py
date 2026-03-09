"""Regression tests for canonical sdfdag training helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import typer
from typer.models import OptionInfo
from typer.testing import CliRunner

from navi_actor.cli import (
    _validate_sdfdag_training_scenes,
    app,
    train,
)
from navi_actor.config import ActorConfig

_RUNNER = CliRunner()


def _option_default(func: object, name: str) -> object:
    default = inspect.signature(func).parameters[name].default
    assert isinstance(default, OptionInfo)
    return default.default


def test_validate_sdfdag_training_scenes_accepts_gmdag_assets() -> None:
    """Canonical training scene pools should accept compiled assets only."""
    _validate_sdfdag_training_scenes([
        str(Path("artifacts/gmdag/sample_apartment.gmdag")),
        str(Path("artifacts/gmdag/arena.gmdag")),
    ])


@pytest.mark.parametrize(
    "scene_path",
    [
        "data/scenes/sample_apartment.glb",
        "data/scenes/sample_apartment.obj",
        "data/scenes/sample_episodes.json",
    ],
)
def test_validate_sdfdag_training_scenes_rejects_non_gmdag_assets(
    scene_path: str,
) -> None:
    """Raw manifests and meshes must be rejected for canonical training."""
    with pytest.raises(typer.Exit) as exc_info:
        _validate_sdfdag_training_scenes([scene_path])

    assert exc_info.value.exit_code == 1


def test_train_defaults_match_canonical_perf_profile() -> None:
    """Canonical training defaults should keep the canonical perf-safe knobs."""
    assert _option_default(train, "actors") == 4
    assert _option_default(train, "azimuth_bins") == 128
    assert _option_default(train, "elevation_bins") == 24
    assert _option_default(train, "minibatch_size") == 64
    assert _option_default(train, "bptt_len") == 8
    assert _option_default(train, "memory_capacity") == 100_000
    assert _option_default(train, "checkpoint_every") == 25_000
    assert _option_default(train, "enable_episodic_memory") is True
    assert _option_default(train, "enable_reward_shaping") is True
    assert _option_default(train, "emit_observation_stream") is True
    assert _option_default(train, "emit_training_telemetry") is True
    assert _option_default(train, "emit_perf_telemetry") is True


def test_cli_help_does_not_expose_second_canonical_training_mode() -> None:
    """Training cleanup should leave one canonical training command surface."""
    result = _RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "train" in result.output
    assert "train-sequential" not in result.output
    assert "train-ppo" not in result.output
    assert "train-unified" not in result.output


def test_train_help_does_not_expose_backend_selector() -> None:
    """Canonical training should not advertise alternate backend selection."""
    result = _RUNNER.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "--backend" not in result.output


def test_train_uses_single_canonical_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical training command should construct the single direct trainer."""
    scratch_dir = Path("tests/.tmp_local") / f"cli-canonical-{uuid4().hex}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scene_path = scratch_dir / "sample_scene.gmdag"
    scene_path.write_bytes(b"G" * 2048)

    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, config: object, *, gmdag_file: str, scene_pool: tuple[str, ...]) -> None:
            captured["config"] = config
            captured["gmdag_file"] = gmdag_file
            captured["scene_pool"] = scene_pool

        def load_training_state(self, checkpoint: str) -> None:
            captured["checkpoint"] = checkpoint

        def start(self) -> None:
            captured["started"] = True

        def train(
            self,
            *,
            total_steps: int,
            log_every: int,
            checkpoint_every: int,
            checkpoint_dir: str,
        ) -> object:
            captured["total_steps"] = total_steps
            captured["log_every"] = log_every
            captured["checkpoint_every"] = checkpoint_every
            captured["checkpoint_dir"] = checkpoint_dir
            return type(
                "Metrics",
                (),
                {
                    "total_steps": total_steps,
                    "episodes": 0,
                    "reward_ema": 0.0,
                    "policy_loss": 0.0,
                },
            )()

        def stop(self) -> None:
            captured["stopped"] = True

        def save_training_state(self, path: Path) -> None:
            captured["final_checkpoint"] = path

    monkeypatch.setattr(
        "navi_actor.cli._build_canonical_trainer",
        lambda config, *, gmdag_file, scene_pool: FakeTrainer(
            config,
            gmdag_file=gmdag_file,
            scene_pool=scene_pool,
        ),
    )
    monkeypatch.setattr("navi_actor.cli._validate_bindable", lambda address, label: None)

    result = _RUNNER.invoke(
        app,
        [
            "train",
            "--gmdag-file",
            str(scene_path),
            "--total-steps",
            "64",
            "--no-enable-episodic-memory",
            "--no-enable-reward-shaping",
            "--no-emit-observation-stream",
            "--no-emit-training-telemetry",
            "--no-emit-perf-telemetry",
            "--checkpoint-dir",
            str(scratch_dir / "ckpts"),
        ],
    )

    assert result.exit_code == 0
    assert captured["gmdag_file"] == str(scene_path)
    assert captured["scene_pool"] == (str(scene_path),)
    assert captured["started"] is True
    assert captured["stopped"] is True
    assert captured["total_steps"] == 64
    config = cast("ActorConfig", captured["config"])
    assert config.enable_episodic_memory is False
    assert config.enable_reward_shaping is False
    assert config.emit_observation_stream is False
    assert config.emit_training_telemetry is False
    assert config.emit_perf_telemetry is False
