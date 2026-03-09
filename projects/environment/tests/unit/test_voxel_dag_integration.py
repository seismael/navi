"""Tests for `.gmdag` integration helpers."""

from __future__ import annotations

import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from navi_environment.cli import _build_backend, app
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.voxel_dag import load_gmdag_asset, probe_sdfdag_runtime

_HEADER = struct.Struct("<4sIIffffI")
_RUNNER = CliRunner()


def test_load_gmdag_asset_reads_header_and_nodes(tmp_path: Path) -> None:
    asset_path = tmp_path / "mini_scene.gmdag"
    nodes = np.array([1, 2, 3], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        64,
        -2.0,
        -1.0,
        -3.0,
        0.25,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    asset = load_gmdag_asset(asset_path)

    assert asset.path == asset_path.resolve()
    assert asset.version == 1
    assert asset.resolution == 64
    assert asset.bbox_min == (-2.0, -1.0, -3.0)
    assert asset.bbox_max == (14.0, 15.0, 13.0)
    assert asset.voxel_size == 0.25
    assert np.array_equal(asset.nodes, nodes)


def test_load_gmdag_asset_rejects_bad_magic(tmp_path: Path) -> None:
    asset_path = tmp_path / "broken.gmdag"
    header = _HEADER.pack(
        b"NOPE",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        0,
    )
    asset_path.write_bytes(header)

    with pytest.raises(RuntimeError, match="Unsupported gmdag magic"):
        load_gmdag_asset(asset_path)


def test_probe_sdfdag_runtime_reports_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str) -> object:
        if name == "torch":
            raise ImportError("torch missing")
        if name == "torch_sdf":
            raise ImportError("torch_sdf missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("navi_environment.integration.voxel_dag.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "navi_environment.integration.voxel_dag._resolve_voxel_dag_executable",
        lambda: (_ for _ in ()).throw(RuntimeError("compiler missing")),
    )

    status = probe_sdfdag_runtime()

    assert status.compiler_ready is False
    assert status.torch_ready is False
    assert status.cuda_ready is False
    assert status.torch_sdf_ready is False
    assert any("compiler missing" in issue for issue in status.issues)
    assert any("torch import failed" in issue for issue in status.issues)
    assert any("torch_sdf import failed" in issue for issue in status.issues)


def test_probe_sdfdag_runtime_loads_asset_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset_path = tmp_path / "mini_scene.gmdag"
    nodes = np.array([7, 8], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        1.0,
        2.0,
        3.0,
        0.5,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    def fake_import_module(name: str) -> object:
        if name == "torch":
            return fake_torch
        if name == "torch_sdf":
            return object()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("navi_environment.integration.voxel_dag.importlib.import_module", fake_import_module)
    monkeypatch.setattr(
        "navi_environment.integration.voxel_dag._resolve_voxel_dag_executable",
        lambda: tmp_path / "voxel-dag.exe",
    )

    status = probe_sdfdag_runtime(asset_path)

    assert status.compiler_ready is True
    assert status.torch_ready is True
    assert status.cuda_ready is True
    assert status.torch_sdf_ready is True
    assert status.asset_loaded is True
    assert status.resolution == 32
    assert status.node_count == 2
    assert status.bbox_min == (1.0, 2.0, 3.0)
    assert status.bbox_max == (17.0, 18.0, 19.0)
    assert status.issues == ()


def test_check_sdfdag_cli_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    monkeypatch.setattr(
        "navi_environment.cli.probe_sdfdag_runtime",
        lambda _path=None: GmDagRuntimeStatus(
            compiler_ready=True,
            torch_ready=True,
            cuda_ready=True,
            torch_sdf_ready=True,
            asset_loaded=False,
            compiler_path=Path("C:/tools/voxel-dag.exe"),
            gmdag_path=None,
            resolution=None,
            node_count=None,
            bbox_min=None,
            bbox_max=None,
            issues=(),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 0
    assert "compiler_ready=True" in result.stdout
    assert "torch_sdf_ready=True" in result.stdout


def test_check_sdfdag_cli_exits_nonzero_on_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    monkeypatch.setattr(
        "navi_environment.cli.probe_sdfdag_runtime",
        lambda _path=None: GmDagRuntimeStatus(
            compiler_ready=False,
            torch_ready=False,
            cuda_ready=False,
            torch_sdf_ready=False,
            asset_loaded=False,
            compiler_path=None,
            gmdag_path=None,
            resolution=None,
            node_count=None,
            bbox_min=None,
            bbox_max=None,
            issues=("torch import failed: missing",),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 1
    assert "issue=torch import failed: missing" in result.output


def test_build_backend_sdfdag_exits_on_preflight_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    monkeypatch.setattr(
        "navi_environment.cli.probe_sdfdag_runtime",
        lambda _path=None: GmDagRuntimeStatus(
            compiler_ready=False,
            torch_ready=False,
            cuda_ready=False,
            torch_sdf_ready=False,
            asset_loaded=False,
            compiler_path=None,
            gmdag_path=None,
            resolution=None,
            node_count=None,
            bbox_min=None,
            bbox_max=None,
            issues=("torch import failed: missing",),
        ),
    )

    with pytest.raises(typer.Exit) as exc_info:
        _build_backend(EnvironmentConfig(backend="sdfdag", gmdag_file="scene.gmdag"))

    assert exc_info.value.exit_code == 1


def test_bench_sdfdag_cli_reports_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, int]] = []
            self.batch_calls = 0

        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            self.reset_calls.append((episode_id, actor_id))
            return object()

        def batch_step(self, actions: tuple[object, ...], step_id: int) -> tuple[tuple[object, ...], tuple[object, ...]]:
            self.batch_calls += 1
            return tuple(object() for _ in actions), tuple(object() for _ in actions)

        def perf_snapshot(self) -> object:
            return SimpleNamespace(
                sps=63.5,
                last_batch_step_ms=14.2,
                ema_batch_step_ms=13.8,
                avg_batch_step_ms=14.0,
                avg_actor_step_ms=3.5,
                total_batches=9,
                total_actor_steps=36,
            )

        def close(self) -> None:
            return None

    backend = FakeBackend()
    monkeypatch.setattr("navi_environment.cli._build_backend", lambda _config: backend)
    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr("navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values))

    result = _RUNNER.invoke(
        app,
        [
            "bench-sdfdag",
            "--gmdag-file",
            "scene.gmdag",
            "--actors",
            "4",
            "--steps",
            "8",
            "--warmup-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert backend.reset_calls == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert backend.batch_calls == 9
    assert "measured_sps=64.00" in result.stdout
    assert "rolling_sps=63.50" in result.stdout
