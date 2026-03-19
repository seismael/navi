"""End-to-end validation for the promoted canonical downloaded corpus."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from navi_contracts import Action
from navi_environment.backends.sdfdag_backend import SdfDagBackend
from navi_environment.cli import app
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.corpus import validate_compiled_scene_corpus
from navi_environment.integration.voxel_dag import probe_sdfdag_runtime

_RUNNER = CliRunner()


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve repo root for live corpus validation"
    raise RuntimeError(msg)


def test_live_corpus_uses_downloaded_datasets_only() -> None:
    repo_root = _repo_root()
    live_root = repo_root / "artifacts" / "gmdag" / "corpus"
    manifest_path = live_root / "gmdag_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live compiled corpus is not present in this workspace")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scenes = manifest.get("scenes", [])
    assert scenes, "Live compiled corpus must contain at least one scene"
    assert set(manifest.get("compiled_resolutions", [])) == {512}

    allowed_datasets = {"habitat_test_scenes", "hssd", "replicacad"}
    forbidden_tokens = (
        "sample",
        "generated",
        "synthetic",
        "procedural",
        "corpus-refresh",
        "downloads",
    )

    for scene in scenes:
        dataset = str(scene.get("dataset", ""))
        scene_name = str(scene.get("scene_name", ""))
        source_path_raw = str(scene.get("source_path", ""))
        source_path = source_path_raw.lower()
        gmdag_path = Path(str(scene.get("gmdag_path", ""))).expanduser()
        source_asset_path = Path(source_path_raw).expanduser()
        if not gmdag_path.is_absolute():
            gmdag_path = (manifest_path.parent / gmdag_path).resolve()
        else:
            gmdag_path = gmdag_path.resolve()
        if not source_asset_path.is_absolute():
            source_asset_path = (manifest_path.parent / source_asset_path).resolve()
        else:
            source_asset_path = source_asset_path.resolve()

        assert dataset in allowed_datasets
        assert scene_name
        assert not any(token in source_path for token in forbidden_tokens)
        assert source_asset_path.suffix.lower() == ".gmdag"
        assert source_asset_path.exists()
        assert live_root == source_asset_path or live_root in source_asset_path.parents
        assert gmdag_path.suffix.lower() == ".gmdag"
        assert gmdag_path.exists()
        assert live_root == gmdag_path or live_root in gmdag_path.parents
        assert source_asset_path == gmdag_path


def test_live_compiled_corpus_validator_accepts_promoted_manifest() -> None:
    repo_root = _repo_root()
    live_root = repo_root / "artifacts" / "gmdag" / "corpus"
    manifest_path = live_root / "gmdag_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live compiled corpus is not present in this workspace")

    validation = validate_compiled_scene_corpus(
        live_root, manifest_path=manifest_path, expected_resolution=512
    )

    assert validation.manifest_present is True
    assert validation.scene_count > 0
    assert validation.compiled_resolutions == (512,)
    assert validation.issues == ()


def test_live_check_sdfdag_cli_emits_parseable_json_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    live_root = repo_root / "artifacts" / "gmdag" / "corpus"
    manifest_path = live_root / "gmdag_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live compiled corpus is not present in this workspace")

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    result = _RUNNER.invoke(app, ["check-sdfdag", "--json"])

    if result.exit_code != 0:
        payload = json.loads(result.stdout)
        pytest.skip("Live sdfdag runtime unavailable: " + "; ".join(payload.get("issues", [])))

    payload = json.loads(result.stdout)
    assert payload["profile"] == "check-sdfdag"
    assert payload["ok"] is True
    assert payload["runtime"]["compiler_ready"] is True
    assert payload["runtime"]["torch_ready"] is True
    assert payload["runtime"]["torch_sdf_ready"] is True
    assert payload["corpus"]["manifest_present"] is True
    assert payload["corpus"]["scene_count"] > 0
    assert payload["corpus"]["compiled_resolutions"] == [512]
    assert payload["issues"] == []


def test_live_bench_sdfdag_cli_emits_parseable_json_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = _repo_root()
    live_root = repo_root / "artifacts" / "gmdag" / "corpus"
    manifest_path = live_root / "gmdag_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live compiled corpus is not present in this workspace")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scenes = manifest.get("scenes", [])
    if not scenes:
        pytest.skip("Live compiled corpus manifest has no scenes")

    gmdag_path = Path(str(scenes[0].get("gmdag_path", ""))).expanduser()
    if not gmdag_path.is_absolute():
        gmdag_path = (manifest_path.parent / gmdag_path).resolve()
    else:
        gmdag_path = gmdag_path.resolve()
    if not gmdag_path.exists():
        pytest.skip("Resolved live .gmdag asset is missing")

    monkeypatch.setattr("navi_environment.cli.setup_logging", lambda *_args: None)
    result = _RUNNER.invoke(
        app,
        [
            "bench-sdfdag",
            "--gmdag-file",
            str(gmdag_path),
            "--actors",
            "1",
            "--steps",
            "1",
            "--warmup-steps",
            "0",
            "--azimuth-bins",
            "32",
            "--elevation-bins",
            "8",
            "--json",
        ],
    )

    if result.exit_code != 0:
        pytest.skip("Live sdfdag benchmark unavailable: " + result.output.strip())

    payload = json.loads(result.stdout)
    assert payload["profile"] == "bench-sdfdag"
    assert Path(payload["gmdag_file"]).resolve() == gmdag_path
    assert payload["actors"] == 1
    assert payload["steps"] == 1
    assert payload["warmup_steps"] == 0
    assert payload["expected_total_batches"] == 1
    assert payload["expected_total_actor_steps"] == 1
    assert payload["total_batches"] >= 1
    assert payload["total_actor_steps"] >= 1
    assert payload["measured_sps"] >= 0.0


def test_live_sdfdag_step_uses_fixed_horizon_saturation_contract() -> None:
    repo_root = _repo_root()
    live_root = repo_root / "artifacts" / "gmdag" / "corpus"
    manifest_path = live_root / "gmdag_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Live compiled corpus is not present in this workspace")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scenes = manifest.get("scenes", [])
    if not scenes:
        pytest.skip("Live compiled corpus manifest has no scenes")

    gmdag_path = Path(str(scenes[0].get("gmdag_path", ""))).expanduser()
    if not gmdag_path.is_absolute():
        gmdag_path = (manifest_path.parent / gmdag_path).resolve()
    else:
        gmdag_path = gmdag_path.resolve()
    if not gmdag_path.exists():
        pytest.skip("Resolved live .gmdag asset is missing")

    runtime_status = probe_sdfdag_runtime(gmdag_path)
    if runtime_status.issues:
        pytest.skip("Live sdfdag runtime unavailable: " + "; ".join(runtime_status.issues))

    backend = SdfDagBackend(
        EnvironmentConfig(
            backend="sdfdag",
            gmdag_file=str(gmdag_path),
            n_actors=1,
            max_distance=0.5,
            azimuth_bins=64,
            elevation_bins=16,
            training_mode=True,
        )
    )
    try:
        backend.reset(episode_id=1, actor_id=0)
        action = Action(
            env_ids=np.array([0], dtype=np.int32),
            linear_velocity=np.zeros((1, 3), dtype=np.float32),
            angular_velocity=np.zeros((1, 3), dtype=np.float32),
            policy_id="test-fixed-horizon",
            step_id=1,
            timestamp=0.0,
        )

        observation, result = backend.step(action, step_id=1, actor_id=0)

        invalid_mask = np.logical_not(observation.valid_mask[0])
        assert result.env_id == 0
        assert observation.matrix_shape == observation.depth[0].shape
        assert observation.valid_mask[0].shape == observation.depth[0].shape
        assert invalid_mask.any(), (
            "Expected at least one ray to saturate beyond the fixed configured horizon"
        )
        assert np.allclose(observation.depth[0][invalid_mask], 1.0)
        assert np.all(observation.depth[0] >= 0.0)
        assert np.all(observation.depth[0] <= 1.0)
    finally:
        backend.close()
