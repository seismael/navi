"""Regression tests for canonical training corpus preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from navi_environment.integration.corpus import prepare_training_scene_corpus


def test_prepare_training_scene_corpus_discovers_and_compiles_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canonical corpus prep should discover sources, compile them, and emit manifests."""
    source_root = tmp_path / "sources"
    gmdag_root = tmp_path / "compiled"
    dataset_root = source_root / "replicacad"
    dataset_root.mkdir(parents=True, exist_ok=True)

    scene_one = dataset_root / "stage_one.glb"
    scene_two = dataset_root / "stage_two.glb"
    scene_one.write_bytes(b"A" * 2048)
    scene_two.write_bytes(b"B" * 4096)

    compile_calls: list[tuple[Path, Path, int]] = []

    def fake_compile_gmdag_world(*, source_path: Path, output_path: Path, resolution: int) -> None:
        compile_calls.append((source_path, output_path, resolution))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"G" * 1024)

    monkeypatch.setattr(
        "navi_environment.integration.corpus.compile_gmdag_world",
        fake_compile_gmdag_world,
    )

    corpus = prepare_training_scene_corpus(
        source_root=source_root,
        gmdag_root=gmdag_root,
        resolution=1536,
        min_scene_bytes=1000,
    )

    compiled_paths = tuple(entry.compiled_path for entry in corpus.scene_entries)
    assert len(corpus.scene_entries) == 2
    assert compile_calls == [
        (scene_one.resolve(), compiled_paths[0], 1536),
        (scene_two.resolve(), compiled_paths[1], 1536),
    ]
    assert compiled_paths[0].name == "stage_one.gmdag"
    assert compiled_paths[1].name == "stage_two.gmdag"
    assert all(path.exists() for path in compiled_paths)

    source_manifest = json.loads(corpus.source_manifest_path.read_text(encoding="utf-8"))
    compiled_manifest = json.loads(corpus.compiled_manifest_path.read_text(encoding="utf-8"))
    assert source_manifest["scene_count"] == 2
    assert compiled_manifest["scene_count"] == 2
    assert source_manifest["scenes"][0]["dataset"] == "replicacad"
    assert compiled_manifest["scenes"][0]["gmdag_path"].endswith("stage_one.gmdag")


def test_prepare_training_scene_corpus_reuses_existing_compiled_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fresh compiled outputs should be reused when force refresh is disabled."""
    source_root = tmp_path / "sources"
    gmdag_root = tmp_path / "compiled"
    source_root.mkdir(parents=True, exist_ok=True)
    gmdag_root.mkdir(parents=True, exist_ok=True)

    scene_path = source_root / "solo.glb"
    scene_path.write_bytes(b"S" * 2048)
    compiled_path = gmdag_root / "solo.gmdag"
    compiled_path.write_bytes(b"G" * 1024)

    compile_calls: list[tuple[Path, Path, int]] = []

    def fake_compile_gmdag_world(*, source_path: Path, output_path: Path, resolution: int) -> None:
        compile_calls.append((source_path, output_path, resolution))

    monkeypatch.setattr(
        "navi_environment.integration.corpus.compile_gmdag_world",
        fake_compile_gmdag_world,
    )

    corpus = prepare_training_scene_corpus(
        source_root=source_root,
        gmdag_root=gmdag_root,
        resolution=1024,
        min_scene_bytes=1000,
    )

    assert len(corpus.scene_entries) == 1
    assert corpus.scene_entries[0].compiled_path == compiled_path.resolve()
    assert compile_calls == []


def test_prepare_training_scene_corpus_supports_compiled_only_reuse(
    tmp_path: Path,
) -> None:
    """Canonical training should be able to run from the compiled corpus alone."""
    gmdag_root = tmp_path / "compiled"
    compiled_path = gmdag_root / "replicacad" / "stage.gmdag"
    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_path.write_bytes(b"G" * 1024)

    corpus = prepare_training_scene_corpus(gmdag_root=gmdag_root)

    assert len(corpus.scene_entries) == 1
    assert corpus.scene_entries[0].compiled_path == compiled_path.resolve()
    assert corpus.scene_entries[0].scene_name == "stage"


def test_prepare_training_scene_corpus_falls_back_when_compiled_manifest_is_stale(
    tmp_path: Path,
) -> None:
    """Canonical training should scan the live corpus when manifest paths are stale."""
    gmdag_root = tmp_path / "compiled"
    compiled_path = gmdag_root / "replicacad" / "stage.gmdag"
    compiled_path.parent.mkdir(parents=True, exist_ok=True)
    compiled_path.write_bytes(b"G" * 1024)

    stale_manifest = {
        "gmdag_root": str(tmp_path / "scratch" / "compiled"),
        "scene_count": 1,
        "scenes": [
            {
                "source_path": str(tmp_path / "scratch" / "downloads" / "replicacad" / "stage.glb"),
                "gmdag_path": str(tmp_path / "scratch" / "compiled" / "replicacad" / "stage.gmdag"),
                "dataset": "replicacad",
                "scene_name": "stage",
                "resolution": 512,
            }
        ],
    }
    (gmdag_root / "gmdag_manifest.json").write_text(json.dumps(stale_manifest), encoding="utf-8")

    corpus = prepare_training_scene_corpus(gmdag_root=gmdag_root)

    assert len(corpus.scene_entries) == 1
    assert corpus.scene_entries[0].compiled_path == compiled_path.resolve()
    assert corpus.scene_entries[0].scene_name == "stage"


def test_prepare_training_scene_corpus_recompiles_resolution_mismatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Force refresh should rebuild compiled assets with the wrong stored resolution."""
    source_root = tmp_path / "sources"
    gmdag_root = tmp_path / "compiled"
    source_root.mkdir(parents=True, exist_ok=True)
    gmdag_root.mkdir(parents=True, exist_ok=True)

    scene_path = source_root / "solo.glb"
    scene_path.write_bytes(b"S" * 2048)
    compiled_path = gmdag_root / "solo.gmdag"
    compiled_path.write_bytes(b"G" * 1024)

    compile_calls: list[tuple[Path, Path, int]] = []

    def fake_compile_gmdag_world(*, source_path: Path, output_path: Path, resolution: int) -> None:
        compile_calls.append((source_path, output_path, resolution))

    class _FakeAsset:
        def __init__(self, resolution: int) -> None:
            self.resolution = resolution

    monkeypatch.setattr(
        "navi_environment.integration.corpus.compile_gmdag_world",
        fake_compile_gmdag_world,
    )
    monkeypatch.setattr(
        "navi_environment.integration.corpus.load_gmdag_asset",
        lambda _path: _FakeAsset(2048),
    )

    prepare_training_scene_corpus(
        source_root=source_root,
        gmdag_root=gmdag_root,
        resolution=512,
        min_scene_bytes=1000,
        force_recompile=True,
    )

    assert compile_calls == [(scene_path.resolve(), compiled_path.resolve(), 512)]
