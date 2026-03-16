"""Unit tests for the auditor CLI dashboard wiring."""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from navi_auditor import cli
from navi_auditor.config import AuditorConfig as RealAuditorConfig


@dataclass
class _StubConfig:
    matrix_sub_address: str = "tcp://env-pub:5559"
    actor_sub_address: str = "tcp://actor-pub:5557"
    actor_control_address: str = "tcp://actor-control:5561"
    step_endpoint: str = "tcp://env-rep:5560"
    output_path: str = "session.zarr"
    pub_address: str = "tcp://*:5558"
    observation_max_distance_m: float = 30.0


class _ViewerSpy:
    last_kwargs: dict[str, Any] | None = None
    run_calls: int = 0

    def __init__(self, **kwargs: Any) -> None:
        type(self).last_kwargs = kwargs

    def run(self) -> None:
        type(self).run_calls += 1


class _RecorderSpy:
    last_config: Any = None
    run_calls: int = 0

    def __init__(self, config: Any, backend: Any) -> None:
        type(self).last_config = config

    def run(self) -> None:
        type(self).run_calls += 1


runner = CliRunner()


def _reset_spy() -> None:
    _ViewerSpy.last_kwargs = None
    _ViewerSpy.run_calls = 0


def _reset_recorder_spy() -> None:
    _RecorderSpy.last_config = None
    _RecorderSpy.run_calls = 0


def _make_local_scratch_dir() -> Path:
    root = Path("tests/.pytest_cache_local") / "dataset_audit"
    root.mkdir(parents=True, exist_ok=True)
    scratch = root / uuid.uuid4().hex
    scratch.mkdir(parents=True, exist_ok=False)
    return scratch


def test_dashboard_passive_mode_disables_environment_sockets(monkeypatch: Any) -> None:
    _reset_spy()
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(cli, "MatrixViewer", _ViewerSpy)

    result = runner.invoke(cli.app, ["dashboard", "--passive"])

    assert result.exit_code == 0
    assert "Mode=passive actor-only" in result.stdout
    assert _ViewerSpy.run_calls == 1
    assert _ViewerSpy.last_kwargs == {
        "matrix_sub": "",
        "actor_sub": "tcp://actor-pub:5557",
        "actor_control_endpoint": "tcp://actor-control:5561",
        "step_endpoint": "",
        "actor_id": 0,
        "enable_actor_selector": True,
        "hz": 30.0,
        "linear_speed": 1.5,
        "yaw_rate": 1.5,
        "max_distance_m": 30.0,
        "scene_path": None,
    }


def test_dashboard_passive_mode_ignores_explicit_environment_endpoints(monkeypatch: Any) -> None:
    _reset_spy()
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(cli, "MatrixViewer", _ViewerSpy)

    result = runner.invoke(
        cli.app,
        [
            "dashboard",
            "--passive",
            "--matrix-sub",
            "tcp://override-env-pub:6000",
            "--step-endpoint",
            "tcp://override-env-rep:6001",
            "--actor-sub",
            "tcp://override-actor-pub:6002",
            "--enable-actor-selector",
            "--actor-id",
            "3",
        ],
    )

    assert result.exit_code == 0
    assert _ViewerSpy.run_calls == 1
    assert _ViewerSpy.last_kwargs is not None
    assert _ViewerSpy.last_kwargs["matrix_sub"] == ""
    assert _ViewerSpy.last_kwargs["step_endpoint"] == ""
    assert _ViewerSpy.last_kwargs["actor_sub"] == "tcp://override-actor-pub:6002"
    assert _ViewerSpy.last_kwargs["actor_control_endpoint"] == "tcp://actor-control:5561"
    assert _ViewerSpy.last_kwargs["enable_actor_selector"] is True
    assert _ViewerSpy.last_kwargs["actor_id"] == 3
    assert _ViewerSpy.last_kwargs["max_distance_m"] == 30.0


def test_dashboard_default_mode_uses_configured_environment_wiring(monkeypatch: Any) -> None:
    _reset_spy()
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(cli, "MatrixViewer", _ViewerSpy)

    result = runner.invoke(cli.app, ["dashboard"])

    assert result.exit_code == 0
    assert _ViewerSpy.run_calls == 1
    assert _ViewerSpy.last_kwargs is not None
    assert _ViewerSpy.last_kwargs["matrix_sub"] == "tcp://env-pub:5559"
    assert _ViewerSpy.last_kwargs["actor_sub"] == "tcp://actor-pub:5557"
    assert _ViewerSpy.last_kwargs["actor_control_endpoint"] == "tcp://actor-control:5561"
    assert _ViewerSpy.last_kwargs["step_endpoint"] == "tcp://env-rep:5560"
    assert _ViewerSpy.last_kwargs["enable_actor_selector"] is True
    assert _ViewerSpy.last_kwargs["actor_id"] == 0
    assert _ViewerSpy.last_kwargs["max_distance_m"] == 30.0


def test_dashboard_accepts_explicit_actor_control_endpoint(monkeypatch: Any) -> None:
    _reset_spy()
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(cli, "MatrixViewer", _ViewerSpy)

    result = runner.invoke(
        cli.app,
        [
            "dashboard",
            "--actor-control-endpoint",
            "tcp://override-actor-control:6003",
        ],
    )

    assert result.exit_code == 0
    assert _ViewerSpy.last_kwargs is not None
    assert _ViewerSpy.last_kwargs["actor_control_endpoint"] == "tcp://override-actor-control:6003"


def test_dashboard_attach_check_emits_json_summary(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(
        cli,
        "_wait_for_dashboard_attach",
        lambda actor_sub, timeout_seconds: {
            "profile": "dashboard-attach-check",
            "ok": True,
            "actor_sub": actor_sub,
            "topic": "telemetry_event_v2",
            "payload_bytes": 128,
            "elapsed_seconds": 0.25,
            "timeout_seconds": timeout_seconds,
        },
    )

    result = runner.invoke(cli.app, ["dashboard-attach-check", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["actor_sub"] == "tcp://actor-pub:5557"
    assert payload["topic"] == "telemetry_event_v2"


def test_dashboard_attach_check_exits_nonzero_on_timeout(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(
        cli,
        "_wait_for_dashboard_attach",
        lambda actor_sub, _timeout_seconds: {
            "profile": "dashboard-attach-check",
            "ok": False,
            "actor_sub": actor_sub,
            "topic": "",
            "payload_bytes": 0,
            "elapsed_seconds": 15.0,
            "issues": ["No dashboard-observable actor traffic arrived"],
        },
    )

    result = runner.invoke(cli.app, ["dashboard-attach-check"])

    assert result.exit_code == 1
    assert "ok=False" in result.output
    assert "issue=No dashboard-observable actor traffic arrived" in result.output


def test_dashboard_capture_frame_emits_json_summary(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(
        cli,
        "_capture_dashboard_frame",
        lambda **_kwargs: {
            "profile": "dashboard-capture-frame",
            "ok": True,
            "actor_sub": "tcp://actor-pub:5557",
            "actor_id": 0,
            "output_dir": "artifacts/dashboard-captures/20260315-000000",
            "valid_ratio": 0.82,
            "depth_min": 0.12,
            "depth_max": 0.98,
            "depth_mean": 0.44,
            "center_distance_m": 2.1,
            "elapsed_seconds": 0.5,
        },
    )

    result = runner.invoke(cli.app, ["dashboard-capture-frame", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["actor_id"] == 0


def test_dashboard_capture_frame_exits_nonzero_on_timeout(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(
        cli,
        "_capture_dashboard_frame",
        lambda **_kwargs: {
            "profile": "dashboard-capture-frame",
            "ok": False,
            "actor_sub": "tcp://actor-pub:5557",
            "actor_id": 0,
            "elapsed_seconds": 15.0,
            "issues": ["No matching distance_matrix_v2 frame arrived before timeout."],
        },
    )

    result = runner.invoke(cli.app, ["dashboard-capture-frame"])

    assert result.exit_code == 1
    assert "ok=False" in result.output
    assert "issue=No matching distance_matrix_v2 frame arrived before timeout." in result.output


def test_record_single_sub_uses_one_endpoint(monkeypatch: Any) -> None:
    _reset_recorder_spy()
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    monkeypatch.setattr(cli, "AuditorConfig", _StubConfig)
    monkeypatch.setattr(cli, "Recorder", _RecorderSpy)

    result = runner.invoke(cli.app, ["record", "--sub", "tcp://localhost:5557", "--out", "capture.zarr"])

    assert result.exit_code == 0
    assert _RecorderSpy.run_calls == 1
    assert _RecorderSpy.last_config.matrix_sub_address == "tcp://localhost:5557"
    assert _RecorderSpy.last_config.actor_sub_address == ""
    assert _RecorderSpy.last_config.output_path == "capture.zarr"


def test_real_auditor_config_accepts_field_name_overrides() -> None:
    config = RealAuditorConfig(
        matrix_sub_address="tcp://localhost:5557",
        actor_sub_address="",
        output_path="capture.zarr",
    )

    assert config.matrix_sub_address == "tcp://localhost:5557"
    assert config.actor_sub_address == ""
    assert config.output_path == "capture.zarr"


def test_dataset_audit_runs_environment_check_and_benchmark(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    scratch_dir = _make_local_scratch_dir()
    try:
        manifest_path = scratch_dir / "gmdag_manifest.json"
        gmdag_path = scratch_dir / "scene.gmdag"
        gmdag_path.write_bytes(b"GDAG")
        manifest_path.write_text(
            json.dumps({"scenes": [{"gmdag_path": str(gmdag_path)}]}),
            encoding="utf-8",
        )

        completed_processes = [
            subprocess.CompletedProcess(
                args=["uv", "check-sdfdag"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "profile": "check-sdfdag",
                        "ok": True,
                        "issues": [],
                        "corpus": {"manifest": str(manifest_path)},
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["uv", "bench-sdfdag"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "profile": "bench-sdfdag",
                        "gmdag_file": str(gmdag_path),
                        "actors": 1,
                        "steps": 8,
                        "measured_sps": 42.5,
                    }
                ),
                stderr="",
            ),
        ]
        seen_commands: list[list[str]] = []

        def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen_commands.append(command)
            return completed_processes.pop(0)

        monkeypatch.setattr(cli.subprocess, "run", _fake_run)

        result = runner.invoke(cli.app, ["dataset-audit", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["profile"] == "dataset-audit"
        assert payload["ok"] is True
        assert payload["check"]["profile"] == "check-sdfdag"
        assert payload["benchmark"]["profile"] == "bench-sdfdag"
        assert payload["benchmark"]["gmdag_file"] == str(gmdag_path)
        assert seen_commands[0][-4:] == ["check-sdfdag", "--expected-resolution", "512", "--json"]
        assert "bench-sdfdag" in seen_commands[1]
        assert any(argument.endswith("scene.gmdag") for argument in seen_commands[1])
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def test_dataset_audit_exits_after_failed_check(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    seen_commands: list[list[str]] = []

    def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=1,
            stdout=json.dumps(
                {
                    "profile": "check-sdfdag",
                    "ok": False,
                    "issues": ["Compiled manifest is missing"],
                    "corpus": None,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    result = runner.invoke(cli.app, ["dataset-audit", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["benchmark"] is None
    assert payload["issues"] == ["Compiled manifest is missing"]
    assert len(seen_commands) == 1


def test_dataset_audit_uses_explicit_gmdag_override_for_benchmark(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli, "setup_logging", lambda *_args: None)
    scratch_dir = _make_local_scratch_dir()
    try:
        gmdag_path = (scratch_dir / "override.gmdag").resolve()
        gmdag_path.write_bytes(b"GDAG")
        completed_processes = [
            subprocess.CompletedProcess(
                args=["uv", "check-sdfdag"],
                returncode=0,
                stdout=json.dumps({"profile": "check-sdfdag", "ok": True, "issues": [], "corpus": None}),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["uv", "bench-sdfdag"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "profile": "bench-sdfdag",
                        "gmdag_file": str(gmdag_path),
                        "actors": 1,
                        "steps": 8,
                        "measured_sps": 21.0,
                    }
                ),
                stderr="",
            ),
        ]
        seen_commands: list[list[str]] = []

        def _fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            seen_commands.append(command)
            return completed_processes.pop(0)

        monkeypatch.setattr(cli.subprocess, "run", _fake_run)

        result = runner.invoke(cli.app, ["dataset-audit", "--gmdag-file", str(gmdag_path), "--json"])

        assert result.exit_code == 0
        assert "--gmdag-file" in seen_commands[0]
        assert str(gmdag_path) in seen_commands[0]
        assert str(gmdag_path) in seen_commands[1]
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)
