"""Tests for `.gmdag` integration helpers."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from navi_environment.cli import _build_backend, app
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.voxel_dag import load_gmdag_asset, probe_sdfdag_runtime

_HEADER = struct.Struct("<4sIIffffI")
_RUNNER = CliRunner()


def _leaf_node(distance_bits: int = 1, semantic: int = 0) -> np.uint64:
    return np.uint64((1 << 63) | ((semantic & 0xFFFF) << 16) | distance_bits)


def test_load_gmdag_asset_reads_header_and_nodes(tmp_path: Path) -> None:
    asset_path = tmp_path / "mini_scene.gmdag"
    nodes = np.array([_leaf_node(distance_bits=1, semantic=7)], dtype=np.uint64)
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


def test_load_gmdag_asset_rejects_unsupported_version(tmp_path: Path) -> None:
    asset_path = tmp_path / "wrong_version.gmdag"
    nodes = np.array([1], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        2,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="Unsupported gmdag version"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_nonpositive_resolution(tmp_path: Path) -> None:
    asset_path = tmp_path / "bad_resolution.gmdag"
    nodes = np.array([1], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        0,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="Invalid gmdag resolution"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_nonpositive_voxel_size(tmp_path: Path) -> None:
    asset_path = tmp_path / "bad_voxel_size.gmdag"
    nodes = np.array([1], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        0.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="Invalid gmdag voxel size"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_empty_node_payload(tmp_path: Path) -> None:
    asset_path = tmp_path / "empty_nodes.gmdag"
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        0,
    )
    asset_path.write_bytes(header)

    with pytest.raises(RuntimeError, match="Invalid gmdag node count"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_trailing_bytes(tmp_path: Path) -> None:
    asset_path = tmp_path / "trailing_bytes.gmdag"
    nodes = np.array([1, 2], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes() + b"x")

    with pytest.raises(RuntimeError, match="trailing bytes detected"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_nonfinite_bbox_values(tmp_path: Path) -> None:
    asset_path = tmp_path / "bad_bbox.gmdag"
    nodes = np.array([(1 << 63) | 1], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        float("nan"),
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="non-finite bbox_min values"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_empty_internal_child_mask(tmp_path: Path) -> None:
    asset_path = tmp_path / "empty_mask.gmdag"
    nodes = np.array([0], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="empty child mask"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_out_of_range_child_reference(tmp_path: Path) -> None:
    asset_path = tmp_path / "bad_child_ref.gmdag"
    internal_word = np.uint64((1 << 55) | 1)
    nodes = np.array([internal_word, 99], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="child reference out of range"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_rejects_child_pointer_table_past_payload(tmp_path: Path) -> None:
    asset_path = tmp_path / "bad_child_table.gmdag"
    internal_word = np.uint64((1 << 55) | 5)
    nodes = np.array([internal_word], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    with pytest.raises(RuntimeError, match="child pointer table starts beyond payload"):
        load_gmdag_asset(asset_path)


def test_load_gmdag_asset_can_skip_deep_layout_validation_for_runtime_loads(
    tmp_path: Path,
) -> None:
    asset_path = tmp_path / "runtime_only_bad_layout.gmdag"
    internal_word = np.uint64((1 << 55) | 5)
    nodes = np.array([internal_word], dtype=np.uint64)
    header = _HEADER.pack(
        b"GDAG",
        1,
        32,
        0.0,
        0.0,
        0.0,
        1.0,
        nodes.shape[0],
    )
    asset_path.write_bytes(header + nodes.tobytes())

    asset = load_gmdag_asset(asset_path, validate_layout=False)

    assert asset.path == asset_path.resolve()
    assert asset.resolution == 32
    assert np.array_equal(asset.nodes, nodes)


def test_probe_sdfdag_runtime_reports_missing_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str) -> object:
        if name == "torch":
            raise ImportError("torch missing")
        if name == "torch_sdf":
            raise ImportError("torch_sdf missing")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(
        "navi_environment.integration.voxel_dag.importlib.import_module", fake_import_module
    )
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
    nodes = np.array([_leaf_node(distance_bits=2, semantic=8)], dtype=np.uint64)
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

    monkeypatch.setattr(
        "navi_environment.integration.voxel_dag.importlib.import_module", fake_import_module
    )
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
    assert status.node_count == 1
    assert status.bbox_min == (1.0, 2.0, 3.0)
    assert status.bbox_max == (17.0, 18.0, 19.0)
    assert status.issues == ()


def test_check_sdfdag_cli_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=1,
            compiled_resolutions=(512,),
            issues=(),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 0
    assert "compiler_ready=True" in result.stdout
    assert "torch_sdf_ready=True" in result.stdout


def test_check_sdfdag_cli_can_emit_json_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=2,
            compiled_resolutions=(512,),
            issues=(),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "check-sdfdag"
    assert payload["runtime"]["compiler_ready"] is True
    assert payload["corpus"]["scene_count"] == 2
    assert payload["issues"] == []
    assert payload["ok"] is True


def test_check_sdfdag_cli_exits_nonzero_on_issues(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=0,
            compiled_resolutions=(),
            issues=(),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 1
    assert "issue=torch import failed: missing" in result.output


def test_check_sdfdag_cli_validates_promoted_corpus_when_no_asset_is_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=2,
            compiled_resolutions=(512,),
            issues=(),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 0
    assert "corpus - root=C:\\repo\\artifacts\\gmdag\\corpus" in result.stdout
    assert "compiled_resolutions=[512]" in result.stdout


def test_check_sdfdag_cli_exits_nonzero_on_corpus_validation_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=1,
            compiled_resolutions=(256,),
            issues=(
                "Compiled asset resolution mismatch for C:/repo/artifacts/gmdag/corpus/scene.gmdag: expected 512, got 256",
            ),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag"])

    assert result.exit_code == 1
    assert "resolution mismatch" in result.output


def test_check_sdfdag_cli_json_failure_remains_parseable(monkeypatch: pytest.MonkeyPatch) -> None:
    from navi_environment.integration.corpus import CompiledCorpusValidation
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
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
    monkeypatch.setattr(
        "navi_environment.cli.validate_compiled_scene_corpus",
        lambda *_args, **_kwargs: CompiledCorpusValidation(
            gmdag_root=Path("C:/repo/artifacts/gmdag/corpus"),
            manifest_path=Path("C:/repo/artifacts/gmdag/corpus/gmdag_manifest.json"),
            manifest_present=True,
            scene_count=1,
            compiled_resolutions=(256,),
            issues=(
                "Compiled asset resolution mismatch for C:/repo/artifacts/gmdag/corpus/scene.gmdag: expected 512, got 256",
            ),
        ),
    )

    result = _RUNNER.invoke(app, ["check-sdfdag", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any("resolution mismatch" in issue for issue in payload["issues"])


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


def test_build_backend_only_uses_dependency_preflight_before_runtime_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from navi_environment.integration.voxel_dag import GmDagRuntimeStatus

    captured: dict[str, object] = {}

    def _fake_probe(
        path: Path | None = None, *, validate_asset_layout: bool = True
    ) -> GmDagRuntimeStatus:
        captured["probe_path"] = path
        captured["validate_asset_layout"] = validate_asset_layout
        return GmDagRuntimeStatus(
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
        )

    class FakeBackend:
        def __init__(self, config: EnvironmentConfig) -> None:
            captured["config"] = config

    monkeypatch.setattr("navi_environment.cli.probe_sdfdag_runtime", _fake_probe)
    monkeypatch.setattr("navi_environment.backends.sdfdag_backend.SdfDagBackend", FakeBackend)

    backend = _build_backend(EnvironmentConfig(backend="sdfdag", gmdag_file="scene.gmdag"))

    assert isinstance(backend, FakeBackend)
    assert captured["probe_path"] is None
    assert captured["validate_asset_layout"] is True


def test_bench_sdfdag_cli_reports_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeBackend:
        def __init__(self) -> None:
            self.reset_calls: list[tuple[int, int]] = []
            self.batch_calls = 0

        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            self.reset_calls.append((episode_id, actor_id))
            return object()

        def batch_step(
            self, actions: tuple[object, ...], step_id: int
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
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
    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)

    def _build_backend_with_capture(config: EnvironmentConfig) -> FakeBackend:
        captured["config"] = config
        return backend

    monkeypatch.setattr("navi_environment.cli._build_backend", _build_backend_with_capture)
    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr(
        "navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values)
    )

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
    assert cast("EnvironmentConfig", captured["config"]).sdfdag_torch_compile is True
    assert backend.reset_calls == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert backend.batch_calls == 9
    assert "measured_sps=64.00" in result.stdout
    assert "rolling_sps=63.50" in result.stdout


def test_bench_sdfdag_cli_can_disable_torch_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, EnvironmentConfig] = {}

    class FakeBackend:
        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            return object()

        def batch_step(
            self, actions: tuple[object, ...], step_id: int
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
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

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    fake_backend = FakeBackend()

    def _build_backend_with_capture(config: EnvironmentConfig) -> FakeBackend:
        captured["config"] = config
        return fake_backend

    monkeypatch.setattr("navi_environment.cli._build_backend", _build_backend_with_capture)
    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr(
        "navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values)
    )

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
            "--no-torch-compile",
        ],
    )

    assert result.exit_code == 0
    assert captured["config"].sdfdag_torch_compile is False


def test_bench_sdfdag_cli_can_emit_json_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBackend:
        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            return object()

        def batch_step(
            self, actions: tuple[object, ...], step_id: int
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
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

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    monkeypatch.setattr("navi_environment.cli._build_backend", lambda _config: FakeBackend())
    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr(
        "navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values)
    )

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
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "bench-sdfdag"
    assert payload["measured_sps"] == 64.0
    assert payload["expected_total_batches"] == 9
    assert payload["expected_total_actor_steps"] == 36
    assert payload["torch_compile_active"] == 0


def test_bench_sdfdag_cli_can_emit_median_summary_for_repeated_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def __init__(
            self, avg_batch_ms: float, sps: float, total_batches: int, total_actor_steps: int
        ) -> None:
            self._avg_batch_ms = avg_batch_ms
            self._sps = sps
            self._total_batches = total_batches
            self._total_actor_steps = total_actor_steps

        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            return object()

        def batch_step(
            self, actions: tuple[object, ...], step_id: int
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
            return tuple(object() for _ in actions), tuple(object() for _ in actions)

        def perf_snapshot(self) -> object:
            return SimpleNamespace(
                sps=self._sps,
                last_batch_step_ms=self._avg_batch_ms - 0.2,
                ema_batch_step_ms=self._avg_batch_ms - 0.1,
                avg_batch_step_ms=self._avg_batch_ms,
                avg_actor_step_ms=self._avg_batch_ms / 4.0,
                total_batches=self._total_batches,
                total_actor_steps=self._total_actor_steps,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    backends = iter(
        (
            FakeBackend(avg_batch_ms=14.0, sps=63.5, total_batches=9, total_actor_steps=36),
            FakeBackend(avg_batch_ms=10.0, sps=79.5, total_batches=9, total_actor_steps=36),
            FakeBackend(avg_batch_ms=12.0, sps=71.5, total_batches=9, total_actor_steps=36),
        )
    )
    monkeypatch.setattr("navi_environment.cli._build_backend", lambda _config: next(backends))
    perf_counter_values = iter([100.0, 100.5, 200.0, 200.4, 300.0, 300.45])
    monkeypatch.setattr(
        "navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values)
    )

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
            "--repeats",
            "3",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "bench-sdfdag"
    assert payload["aggregation"] == "median"
    assert payload["repeats"] == 3
    assert payload["torch_compile_active"] == 0
    assert payload["measured_sps"] == pytest.approx(71.11111111111111)
    assert payload["measured_sps_mean"] == pytest.approx(71.70370370370371)
    assert payload["measured_sps_min"] == pytest.approx(64.0)
    assert payload["measured_sps_max"] == pytest.approx(80.0)
    assert payload["avg_batch_ms"] == 12.0
    assert len(payload["per_run"]) == 3


def test_bench_sdfdag_cli_exits_on_inconsistent_perf_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBackend:
        def reset(self, episode_id: int, *, actor_id: int = 0) -> object:
            return object()

        def batch_step(
            self, actions: tuple[object, ...], step_id: int
        ) -> tuple[tuple[object, ...], tuple[object, ...]]:
            return tuple(object() for _ in actions), tuple(object() for _ in actions)

        def perf_snapshot(self) -> object:
            return SimpleNamespace(
                sps=63.5,
                last_batch_step_ms=14.2,
                ema_batch_step_ms=13.8,
                avg_batch_step_ms=14.0,
                avg_actor_step_ms=3.5,
                total_batches=2,
                total_actor_steps=8,
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    monkeypatch.setattr("navi_environment.cli._build_backend", lambda _config: FakeBackend())
    perf_counter_values = iter([100.0, 100.5])
    monkeypatch.setattr(
        "navi_environment.cli.time.perf_counter", lambda: next(perf_counter_values)
    )

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

    assert result.exit_code == 1
    assert "perf snapshot is inconsistent" in result.output
