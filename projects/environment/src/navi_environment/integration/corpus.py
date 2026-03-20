"""Canonical dataset discovery and `.gmdag` corpus preparation helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from navi_environment.integration.voxel_dag import compile_gmdag_world, load_gmdag_asset

__all__ = [
    "CompiledCorpusValidation",
    "CompiledSceneEntry",
    "PreparedSceneCorpus",
    "SceneSourceEntry",
    "discover_compiled_scene_entries",
    "discover_scene_sources",
    "find_default_gmdag_corpus_root",
    "find_default_scene_root",
    "prepare_training_scene_corpus",
    "resolve_compiled_scene_query",
    "resolve_scene_query",
    "validate_compiled_scene_corpus",
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


@dataclass(frozen=True, slots=True)
class CompiledCorpusValidation:
    """Validation status for the promoted compiled corpus and manifest."""

    gmdag_root: Path
    manifest_path: Path
    manifest_present: bool
    scene_count: int
    compiled_resolutions: tuple[int, ...]
    issues: tuple[str, ...]


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
                    SceneSourceEntry(
                        path=path, dataset="manifest", scene_name=_canonical_scene_name(path)
                    )
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
                SceneSourceEntry(
                    path=path, dataset="manifest", scene_name=_canonical_scene_name(path)
                )
            )
        return raw_entries

    if isinstance(manifest_data, dict) and "scenes" in manifest_data:
        for item in manifest_data["scenes"]:
            if isinstance(item, str):
                path = _resolve_manifest_scene_path(item, manifest_path)
                raw_entries.append(
                    SceneSourceEntry(
                        path=path, dataset="manifest", scene_name=_canonical_scene_name(path)
                    )
                )
                continue
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path") or item.get("scene")
            if not isinstance(raw_path, str):
                continue
            path = _resolve_manifest_scene_path(raw_path, manifest_path)
            raw_dataset = item.get("dataset")
            dataset = raw_dataset if isinstance(raw_dataset, str) else "manifest"
            raw_entries.append(
                SceneSourceEntry(
                    path=path, dataset=dataset, scene_name=_canonical_scene_name(path)
                )
            )
        return raw_entries

    msg = f"Unsupported scene manifest structure: {manifest_path}"
    raise RuntimeError(msg)


def _load_compiled_manifest_entries(manifest_path: Path) -> list[CompiledSceneEntry]:
    with manifest_path.open(encoding="utf-8-sig") as handle:
        manifest_data: Any = json.load(handle)

    scenes: Any
    if isinstance(manifest_data, list):
        scenes = manifest_data
    elif isinstance(manifest_data, dict):
        scenes = manifest_data.get("scenes", [])
    else:
        scenes = []

    entries: list[CompiledSceneEntry] = []
    for item in scenes:
        if isinstance(item, str):
            compiled_path = _resolve_manifest_scene_path(item, manifest_path)
            if compiled_path.suffix.lower() != ".gmdag":
                continue
            entries.append(
                CompiledSceneEntry(
                    source_path=compiled_path,
                    compiled_path=compiled_path,
                    dataset="compiled",
                    scene_name=compiled_path.stem,
                )
            )
            continue
        if not isinstance(item, dict):
            continue
        raw_compiled_path = item.get("gmdag_path") or item.get("path") or item.get("scene")
        if not isinstance(raw_compiled_path, str):
            continue
        compiled_path = _resolve_manifest_scene_path(raw_compiled_path, manifest_path)
        if compiled_path.suffix.lower() != ".gmdag":
            continue
        raw_source_path = (
            item.get("source_path")
            if isinstance(item.get("source_path"), str)
            else raw_compiled_path
        )
        if not isinstance(raw_source_path, str):
            continue
        source_path = _resolve_manifest_scene_path(raw_source_path, manifest_path)
        raw_dataset = item.get("dataset")
        dataset = raw_dataset if isinstance(raw_dataset, str) else "compiled"
        raw_scene_name = item.get("scene_name")
        scene_name = raw_scene_name if isinstance(raw_scene_name, str) else compiled_path.stem
        entries.append(
            CompiledSceneEntry(
                source_path=source_path,
                compiled_path=compiled_path,
                dataset=dataset,
                scene_name=scene_name,
            )
        )

    if not entries:
        msg = f"Unsupported compiled manifest structure: {manifest_path}"
        raise RuntimeError(msg)
    return entries


def _scan_compiled_scene_entries(gmdag_root: Path) -> list[CompiledSceneEntry]:
    """Discover compiled scenes directly from the filesystem.

    This keeps canonical training usable even if a compiled manifest exists but
    contains stale paths from a previous staging location.
    """
    resolved_root = gmdag_root.resolve()
    return [
        CompiledSceneEntry(
            source_path=path.resolve(),
            compiled_path=path.resolve(),
            dataset=_infer_dataset(path.resolve(), resolved_root),
            scene_name=path.stem,
        )
        for path in sorted(resolved_root.rglob("*.gmdag"))
        if path.is_file()
    ]


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
        if (
            entry.path.suffix.lower() in _SUPPORTED_SOURCE_SUFFIXES
            and entry.path.stat().st_size < min_scene_bytes
        ):
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


def discover_compiled_scene_entries(
    gmdag_root: Path | None = None,
    *,
    manifest_path: Path | None = None,
) -> list[CompiledSceneEntry]:
    """Discover compiled `.gmdag` entries from a manifest or corpus root."""
    entries: list[CompiledSceneEntry] = []
    resolved_root = (gmdag_root or find_default_gmdag_corpus_root()).resolve()
    if manifest_path is not None:
        entries = _load_compiled_manifest_entries(manifest_path.resolve())
    else:
        manifest_candidate = _default_compiled_manifest_path(resolved_root)
        if manifest_candidate.exists():
            try:
                entries = _load_compiled_manifest_entries(manifest_candidate)
            except RuntimeError:
                entries = []
        if not entries:
            entries = _scan_compiled_scene_entries(resolved_root)

    filtered: list[CompiledSceneEntry] = []
    seen_paths: set[Path] = set()
    for entry in entries:
        if not entry.compiled_path.exists():
            continue
        if entry.compiled_path in seen_paths:
            continue
        seen_paths.add(entry.compiled_path)
        filtered.append(entry)

    if not filtered:
        fallback_entries = _scan_compiled_scene_entries(resolved_root)
        if fallback_entries:
            return fallback_entries
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


def resolve_compiled_scene_query(
    scene_query: str,
    entries: list[CompiledSceneEntry],
) -> CompiledSceneEntry:
    """Resolve a scene query against a compiled corpus."""
    query = scene_query.strip()
    if not query:
        msg = "Scene query must be non-empty"
        raise RuntimeError(msg)

    query_path = Path(query).expanduser()
    if query_path.exists():
        resolved = query_path.resolve()
        for entry in entries:
            if entry.compiled_path == resolved or entry.source_path == resolved:
                return entry
        msg = f"Scene exists but is not part of the compiled corpus: {resolved}"
        raise RuntimeError(msg)

    normalized = query.replace("\\", "/").lower()
    matches = [
        entry
        for entry in entries
        if entry.scene_name.lower() == normalized
        or entry.compiled_path.name.lower() == normalized
        or entry.source_path.name.lower() == normalized
        or entry.compiled_path.as_posix().lower().endswith(normalized)
        or entry.source_path.as_posix().lower().endswith(normalized)
    ]
    if not matches:
        msg = f"No compiled scene matched query: {scene_query}"
        raise RuntimeError(msg)
    if len(matches) > 1:
        options = ", ".join(entry.compiled_path.as_posix() for entry in matches[:5])
        msg = f"Compiled scene query is ambiguous: {scene_query}. Matches: {options}"
        raise RuntimeError(msg)
    return matches[0]


def _compiled_output_path(source_path: Path, *, source_root: Path, gmdag_root: Path) -> Path:
    if source_path.suffix.lower() == ".gmdag":
        return source_path.resolve()
    try:
        relative = source_path.resolve().relative_to(source_root.resolve())
        return (gmdag_root / relative).with_suffix(".gmdag")
    except ValueError:
        digest = hashlib.sha256(source_path.resolve().as_posix().encode("utf-8")).hexdigest()[:12]
        return gmdag_root / f"external_{source_path.stem}_{digest}.gmdag"


def _write_source_manifest(
    entries: list[SceneSourceEntry], manifest_path: Path, *, source_root: Path
) -> None:
    payload = {
        "generated": datetime.now(UTC).isoformat(),
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


def _write_compiled_manifest(
    entries: list[CompiledSceneEntry],
    manifest_path: Path,
    *,
    source_root: Path,
    gmdag_root: Path,
    requested_resolution: int,
) -> None:
    scene_payload: list[dict[str, Any]] = []
    compiled_resolutions: set[int] = set()
    for entry in entries:
        resolution: int | None = None
        if entry.compiled_path.exists():
            try:
                resolution = int(load_gmdag_asset(entry.compiled_path).resolution)
            except (FileNotFoundError, RuntimeError):
                resolution = None
        if resolution is not None:
            compiled_resolutions.add(resolution)
        scene_payload.append(
            {
                "source_path": entry.source_path.as_posix(),
                "gmdag_path": entry.compiled_path.as_posix(),
                "dataset": entry.dataset,
                "scene_name": entry.scene_name,
                "resolution": resolution,
            }
        )

    payload = {
        "generated": datetime.now(UTC).isoformat(),
        "source_root": str(source_root),
        "gmdag_root": str(gmdag_root),
        "scene_count": len(entries),
        "requested_resolution": requested_resolution,
        "compiled_resolutions": sorted(compiled_resolutions),
        "scenes": scene_payload,
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
    resolution: int = 512,
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
    resolved_manifest_path = manifest_path.resolve() if manifest_path is not None else None

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

    if not force_recompile:
        discovered_compiled_entries = discover_compiled_scene_entries(
            resolved_gmdag_root,
            manifest_path=resolved_manifest_path,
        )
        if discovered_compiled_entries:
            if not resolved_compiled_manifest.exists():
                _write_compiled_manifest(
                    discovered_compiled_entries,
                    resolved_compiled_manifest,
                    source_root=resolved_source_root,
                    gmdag_root=resolved_gmdag_root,
                    requested_resolution=resolution,
                )
            if scene:
                discovered_compiled_entries = [
                    resolve_compiled_scene_query(scene, discovered_compiled_entries)
                ]
            return PreparedSceneCorpus(
                source_root=resolved_source_root,
                gmdag_root=resolved_gmdag_root,
                source_manifest_path=resolved_source_manifest,
                compiled_manifest_path=resolved_compiled_manifest,
                scene_entries=tuple(discovered_compiled_entries),
            )

    discovered: list[SceneSourceEntry] = discover_scene_sources(
        resolved_source_root,
        manifest_path=resolved_manifest_path,
        min_scene_bytes=min_scene_bytes,
    )

    if scene:
        discovered = [resolve_scene_query(scene, discovered)]

    _write_source_manifest(discovered, resolved_source_manifest, source_root=resolved_source_root)

    compiled_entries: list[CompiledSceneEntry] = []
    for source_entry in discovered:
        compiled_path = _compiled_output_path(
            source_entry.path,
            source_root=resolved_source_root,
            gmdag_root=resolved_gmdag_root,
        )
        if source_entry.path.suffix.lower() != ".gmdag":
            needs_compile = force_recompile or not compiled_path.exists()
            if (
                compiled_path.exists()
                and compiled_path.stat().st_mtime < source_entry.path.stat().st_mtime
            ):
                needs_compile = True
            if compiled_path.exists():
                try:
                    compiled_resolution = int(load_gmdag_asset(compiled_path).resolution)
                except (FileNotFoundError, RuntimeError):
                    needs_compile = True
                else:
                    if compiled_resolution != resolution:
                        needs_compile = True
            if needs_compile:
                compiled_path.parent.mkdir(parents=True, exist_ok=True)
                compile_gmdag_world(
                    source_path=source_entry.path,
                    output_path=compiled_path,
                    resolution=resolution,
                )
        compiled_entries.append(
            CompiledSceneEntry(
                source_path=source_entry.path,
                compiled_path=compiled_path,
                dataset=source_entry.dataset,
                scene_name=source_entry.scene_name,
            )
        )

    _write_compiled_manifest(
        compiled_entries,
        resolved_compiled_manifest,
        source_root=resolved_source_root,
        gmdag_root=resolved_gmdag_root,
        requested_resolution=resolution,
    )

    return PreparedSceneCorpus(
        source_root=resolved_source_root,
        gmdag_root=resolved_gmdag_root,
        source_manifest_path=resolved_source_manifest,
        compiled_manifest_path=resolved_compiled_manifest,
        scene_entries=tuple(compiled_entries),
    )


def validate_compiled_scene_corpus(
    gmdag_root: Path | None = None,
    *,
    manifest_path: Path | None = None,
    expected_resolution: int | None = 512,
) -> CompiledCorpusValidation:
    """Validate the promoted compiled corpus manifest against live `.gmdag` assets."""
    resolved_root = (gmdag_root or find_default_gmdag_corpus_root()).resolve()
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else _default_compiled_manifest_path(resolved_root)
    )

    issues: list[str] = []
    live_entries = _scan_compiled_scene_entries(resolved_root)
    live_paths = {entry.compiled_path.resolve() for entry in live_entries}
    compiled_resolutions: set[int] = set()

    if not live_entries:
        issues.append(f"No compiled .gmdag assets found under {resolved_root}")

    manifest_present = resolved_manifest.exists()
    if not manifest_present:
        issues.append(f"Compiled manifest is missing: {resolved_manifest}")
        return CompiledCorpusValidation(
            gmdag_root=resolved_root,
            manifest_path=resolved_manifest,
            manifest_present=False,
            scene_count=len(live_entries),
            compiled_resolutions=(),
            issues=tuple(issues),
        )

    try:
        manifest_payload = json.loads(resolved_manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"Invalid compiled manifest JSON in {resolved_manifest}: {exc}")
        return CompiledCorpusValidation(
            gmdag_root=resolved_root,
            manifest_path=resolved_manifest,
            manifest_present=True,
            scene_count=len(live_entries),
            compiled_resolutions=(),
            issues=tuple(issues),
        )

    try:
        manifest_entries = _load_compiled_manifest_entries(resolved_manifest)
    except RuntimeError as exc:
        issues.append(str(exc))
        return CompiledCorpusValidation(
            gmdag_root=resolved_root,
            manifest_path=resolved_manifest,
            manifest_present=True,
            scene_count=len(live_entries),
            compiled_resolutions=(),
            issues=tuple(issues),
        )

    manifest_scene_count = manifest_payload.get("scene_count")
    if manifest_scene_count is not None and int(manifest_scene_count) != len(manifest_entries):
        issues.append(
            "Compiled manifest scene_count mismatch: "
            f"declares {manifest_scene_count}, enumerates {len(manifest_entries)} scene entries"
        )

    manifest_root = manifest_payload.get("gmdag_root")
    if isinstance(manifest_root, str) and manifest_root:
        resolved_manifest_root = Path(manifest_root).expanduser().resolve()
        if resolved_manifest_root != resolved_root:
            issues.append(
                "Compiled manifest gmdag_root mismatch: "
                f"declares {resolved_manifest_root}, expected {resolved_root}"
            )

    requested_resolution = manifest_payload.get("requested_resolution")
    if (
        expected_resolution is not None
        and requested_resolution is not None
        and int(requested_resolution) != int(expected_resolution)
    ):
        issues.append(
            "Compiled manifest requested_resolution mismatch: "
            f"declares {requested_resolution}, expected {expected_resolution}"
        )

    manifest_paths: set[Path] = set()
    for entry in manifest_entries:
        compiled_path = entry.compiled_path.resolve()
        if compiled_path in manifest_paths:
            issues.append(f"Compiled manifest contains duplicate asset entry: {compiled_path}")
            continue
        manifest_paths.add(compiled_path)

        if compiled_path.suffix.lower() != ".gmdag":
            issues.append(f"Compiled manifest entry is not a .gmdag asset: {compiled_path}")
            continue
        if not compiled_path.exists():
            issues.append(f"Compiled manifest points at a missing asset: {compiled_path}")
            continue
        if resolved_root != compiled_path and resolved_root not in compiled_path.parents:
            issues.append(
                f"Compiled manifest points outside the promoted corpus root {resolved_root}: {compiled_path}"
            )
            continue

        try:
            asset = load_gmdag_asset(compiled_path)
        except (FileNotFoundError, RuntimeError) as exc:
            issues.append(str(exc))
            continue

        compiled_resolutions.add(int(asset.resolution))
        if expected_resolution is not None and int(asset.resolution) != int(expected_resolution):
            issues.append(
                f"Compiled asset resolution mismatch for {compiled_path}: expected {expected_resolution}, got {asset.resolution}"
            )

    missing_from_manifest = sorted(live_paths - manifest_paths)
    for path in missing_from_manifest:
        issues.append(f"Promoted compiled asset is missing from manifest: {path}")

    stale_manifest_paths = sorted(manifest_paths - live_paths)
    for path in stale_manifest_paths:
        issues.append(f"Compiled manifest references a non-promoted asset: {path}")

    manifest_compiled_resolutions = manifest_payload.get("compiled_resolutions")
    if manifest_compiled_resolutions is not None:
        manifest_resolution_tuple = tuple(
            sorted(int(value) for value in manifest_compiled_resolutions)
        )
        live_resolution_tuple = tuple(sorted(compiled_resolutions))
        if manifest_resolution_tuple != live_resolution_tuple:
            issues.append(
                "Compiled manifest compiled_resolutions mismatch: "
                f"declares {list(manifest_resolution_tuple)}, live corpus uses {list(live_resolution_tuple)}"
            )

    return CompiledCorpusValidation(
        gmdag_root=resolved_root,
        manifest_path=resolved_manifest,
        manifest_present=True,
        scene_count=len(live_entries),
        compiled_resolutions=tuple(sorted(compiled_resolutions)),
        issues=tuple(issues),
    )
