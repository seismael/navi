"""Canonical dataset discovery and `.gmdag` corpus preparation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

from navi_environment.integration.voxel_dag import compile_gmdag_world

__all__ = [
    "CompiledSceneEntry",
    "PreparedSceneCorpus",
    "SceneSourceEntry",
    "discover_scene_sources",
    "find_default_gmdag_corpus_root",
    "find_default_scene_root",
    "prepare_training_scene_corpus",
    "resolve_scene_query",
]

_SUPPORTED_SOURCE_SUFFIXES = {".glb", ".obj", ".ply", ".stl"}
_DEFAULT_SOURCE_MANIFEST = "scene_manifest_all.json"
_DEFAULT_COMPILED_MANIFEST = "gmdag_manifest.json"


@dataclass(frozen=True, slots=True)
class SceneSourceEntry:
    """Source or compiled scene discovered from a manifest or corpus scan."""

    path: Path
    dataset: str
    scene_name: str


@dataclass(frozen=True, slots=True)
class CompiledSceneEntry:
    """Compiled training-ready scene entry."""

    source_path: Path
    compiled_path: Path
    dataset: str
    scene_name: str


@dataclass(frozen=True, slots=True)
class PreparedSceneCorpus:
    """Prepared canonical training corpus with compiled scene assets."""

    source_root: Path
    gmdag_root: Path
    source_manifest_path: Path
    compiled_manifest_path: Path
    scene_entries: tuple[CompiledSceneEntry, ...]


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists() and (parent / "projects").exists():
            return parent
    msg = "Could not resolve Navi repository root for corpus preparation"
    raise RuntimeError(msg)


def find_default_scene_root() -> Path:
    """Return the canonical dataset scene root."""
    return _repo_root() / "data" / "scenes"


def find_default_gmdag_corpus_root() -> Path:
    """Return the canonical compiled corpus root."""
    return _repo_root() / "artifacts" / "gmdag" / "corpus"


def _default_source_manifest_path(source_root: Path) -> Path:
    return source_root / _DEFAULT_SOURCE_MANIFEST


def _default_compiled_manifest_path(gmdag_root: Path) -> Path:
    return gmdag_root / _DEFAULT_COMPILED_MANIFEST


def _infer_dataset(path: Path, source_root: Path) -> str:
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return "external"
    if len(relative.parts) > 1:
        return relative.parts[0]
    return "root"


def _canonical_scene_name(path: Path) -> str:
    return path.stem


def _resolve_manifest_scene_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (manifest_path.parent / path).resolve()


def _load_manifest_entries(manifest_path: Path) -> list[SceneSourceEntry]:
    with manifest_path.open(encoding="utf-8-sig") as handle:
        manifest_data: Any = json.load(handle)

    raw_entries: list[SceneSourceEntry] = []
    if isinstance(manifest_data, list):
        for item in manifest_data:
            if isinstance(item, str):
                path = _resolve_manifest_scene_path(item, manifest_path)
                raw_entries.append(
                    SceneSourceEntry(path=path, dataset="manifest", scene_name=_canonical_scene_name(path))
                )
        return raw_entries

    if isinstance(manifest_data, dict) and "episodes" in manifest_data:
        seen: set[Path] = set()
        for episode in manifest_data["episodes"]:
            if not isinstance(episode, dict):
                continue
            scene_id = episode.get("scene_id") or episode.get("scene")
            if not isinstance(scene_id, str):
                continue
            path = _resolve_manifest_scene_path(scene_id, manifest_path)
            if path in seen:
                continue
            seen.add(path)
            raw_entries.append(
                SceneSourceEntry(path=path, dataset="manifest", scene_name=_canonical_scene_name(path))
            )
        return raw_entries

    if isinstance(manifest_data, dict) and "scenes" in manifest_data:
        for item in manifest_data["scenes"]:
            if isinstance(item, str):
                path = _resolve_manifest_scene_path(item, manifest_path)
                raw_entries.append(
                    SceneSourceEntry(path=path, dataset="manifest", scene_name=_canonical_scene_name(path))
                )
                continue
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path") or item.get("scene")
            if not isinstance(raw_path, str):
                continue
            path = _resolve_manifest_scene_path(raw_path, manifest_path)
            dataset = item.get("dataset") if isinstance(item.get("dataset"), str) else "manifest"
            raw_entries.append(
                SceneSourceEntry(path=path, dataset=dataset, scene_name=_canonical_scene_name(path))
            )
        return raw_entries

    msg = f"Unsupported scene manifest structure: {manifest_path}"
    raise RuntimeError(msg)


def discover_scene_sources(
    source_root: Path | None = None,
    *,
    manifest_path: Path | None = None,
    min_scene_bytes: int = 1000,
) -> list[SceneSourceEntry]:
    """Discover canonical scene sources from a manifest or corpus root."""
    if manifest_path is not None:
        entries = _load_manifest_entries(manifest_path.resolve())
    else:
        resolved_root = (source_root or find_default_scene_root()).resolve()
        entries = []
        for path in sorted(resolved_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SUPPORTED_SOURCE_SUFFIXES:
                continue
            if path.stat().st_size < min_scene_bytes:
                continue
            entries.append(
                SceneSourceEntry(
                    path=path.resolve(),
                    dataset=_infer_dataset(path.resolve(), resolved_root),
                    scene_name=_canonical_scene_name(path),
                )
            )

    filtered: list[SceneSourceEntry] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        if not entry.path.exists():
            continue
        if entry.path.suffix.lower() in _SUPPORTED_SOURCE_SUFFIXES and entry.path.stat().st_size < min_scene_bytes:
            continue
        if entry.path in seen_paths:
            continue
        seen_paths.add(entry.path)
        filtered.append(entry)

    if not filtered:
        root_display = str(manifest_path or source_root or find_default_scene_root())
        msg = f"No valid scene files found under {root_display}"
        raise RuntimeError(msg)

    return filtered


def resolve_scene_query(scene_query: str, entries: list[SceneSourceEntry]) -> SceneSourceEntry:
    """Resolve a single explicit scene query to one unique entry."""
    query = scene_query.strip()
    if not query:
        msg = "Scene query must be non-empty"
        raise RuntimeError(msg)

    query_path = Path(query).expanduser()
    if query_path.exists():
        resolved = query_path.resolve()
        for entry in entries:
            if entry.path == resolved:
                return entry
        msg = f"Scene exists but is not part of the discovered corpus: {resolved}"
        raise RuntimeError(msg)

    normalized = query.replace("\\", "/").lower()
    matches = [
        entry
        for entry in entries
        if entry.scene_name.lower() == normalized
        or entry.path.name.lower() == normalized
        or entry.path.as_posix().lower().endswith(normalized)
    ]
    if not matches:
        msg = f"No discovered scene matched query: {scene_query}"
        raise RuntimeError(msg)
    if len(matches) > 1:
        options = ", ".join(entry.path.as_posix() for entry in matches[:5])
        msg = f"Scene query is ambiguous: {scene_query}. Matches: {options}"
        raise RuntimeError(msg)
    return matches[0]


def _compiled_output_path(source_path: Path, *, source_root: Path, gmdag_root: Path) -> Path:
    if source_path.suffix.lower() == ".gmdag":
        return source_path.resolve()
    try:
        relative = source_path.resolve().relative_to(source_root.resolve())
        return (gmdag_root / relative).with_suffix(".gmdag")
    except ValueError:
        digest = hashlib.sha1(source_path.resolve().as_posix().encode("utf-8")).hexdigest()[:12]
        return gmdag_root / f"external_{source_path.stem}_{digest}.gmdag"


def _write_source_manifest(entries: list[SceneSourceEntry], manifest_path: Path, *, source_root: Path) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "scene_count": len(entries),
        "scenes": [
            {
                "path": entry.path.as_posix(),
                "dataset": entry.dataset,
                "scene_name": entry.scene_name,
            }
            for entry in entries
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_compiled_manifest(entries: list[CompiledSceneEntry], manifest_path: Path, *, source_root: Path, gmdag_root: Path) -> None:
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "gmdag_root": str(gmdag_root),
        "scene_count": len(entries),
        "scenes": [
            {
                "source_path": entry.source_path.as_posix(),
                "gmdag_path": entry.compiled_path.as_posix(),
                "dataset": entry.dataset,
                "scene_name": entry.scene_name,
            }
            for entry in entries
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_training_scene_corpus(
    *,
    scene: str = "",
    manifest_path: Path | None = None,
    gmdag_file: Path | None = None,
    source_root: Path | None = None,
    gmdag_root: Path | None = None,
    resolution: int = 2048,
    min_scene_bytes: int = 1000,
    force_recompile: bool = False,
) -> PreparedSceneCorpus:
    """Prepare the canonical compiled training corpus.

    Priority order:
    1. explicit compiled `.gmdag` file
    2. explicit scene query
    3. explicit manifest
    4. full corpus discovery under the canonical scene root
    """
    resolved_source_root = (source_root or find_default_scene_root()).resolve()
    resolved_gmdag_root = (gmdag_root or find_default_gmdag_corpus_root()).resolve()
    resolved_source_manifest = _default_source_manifest_path(resolved_source_root)
    resolved_compiled_manifest = _default_compiled_manifest_path(resolved_gmdag_root)

    if gmdag_file is not None:
        compiled = gmdag_file.expanduser().resolve()
        if not compiled.exists():
            msg = f"Compiled gmdag asset does not exist: {compiled}"
            raise RuntimeError(msg)
        if compiled.suffix.lower() != ".gmdag":
            msg = f"Canonical training requires compiled .gmdag assets: {compiled}"
            raise RuntimeError(msg)
        entry = CompiledSceneEntry(
            source_path=compiled,
            compiled_path=compiled,
            dataset="compiled",
            scene_name=compiled.stem,
        )
        return PreparedSceneCorpus(
            source_root=resolved_source_root,
            gmdag_root=resolved_gmdag_root,
            source_manifest_path=resolved_source_manifest,
            compiled_manifest_path=resolved_compiled_manifest,
            scene_entries=(entry,),
        )

    discovered = discover_scene_sources(
        resolved_source_root,
        manifest_path=manifest_path.resolve() if manifest_path is not None else None,
        min_scene_bytes=min_scene_bytes,
    )

    if scene:
        discovered = [resolve_scene_query(scene, discovered)]

    _write_source_manifest(discovered, resolved_source_manifest, source_root=resolved_source_root)

    compiled_entries: list[CompiledSceneEntry] = []
    for entry in discovered:
        compiled_path = _compiled_output_path(
            entry.path,
            source_root=resolved_source_root,
            gmdag_root=resolved_gmdag_root,
        )
        if entry.path.suffix.lower() != ".gmdag":
            needs_compile = force_recompile or not compiled_path.exists()
            if compiled_path.exists() and compiled_path.stat().st_mtime < entry.path.stat().st_mtime:
                needs_compile = True
            if needs_compile:
                compiled_path.parent.mkdir(parents=True, exist_ok=True)
                compile_gmdag_world(
                    source_path=entry.path,
                    output_path=compiled_path,
                    resolution=resolution,
                )
        compiled_entries.append(
            CompiledSceneEntry(
                source_path=entry.path,
                compiled_path=compiled_path,
                dataset=entry.dataset,
                scene_name=entry.scene_name,
            )
        )

    _write_compiled_manifest(
        compiled_entries,
        resolved_compiled_manifest,
        source_root=resolved_source_root,
        gmdag_root=resolved_gmdag_root,
    )

    return PreparedSceneCorpus(
        source_root=resolved_source_root,
        gmdag_root=resolved_gmdag_root,
        source_manifest_path=resolved_source_manifest,
        compiled_manifest_path=resolved_compiled_manifest,
        scene_entries=tuple(compiled_entries),
    )
