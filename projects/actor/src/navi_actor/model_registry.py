"""Centralized model registry for tracking promoted checkpoints.

The registry lives at ``artifacts/models/registry.json`` alongside
versioned checkpoint copies (``vNNN.pt``) and a ``latest.pt`` pointer.
All training sources (RL, BC, nightly, manual) promote their best
checkpoints into this single surface so model progression is unified.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)

_DEFAULT_MODELS_DIR = Path("artifacts/models")


@dataclass
class ModelEntry:
    """Metadata for a promoted model in the registry."""

    id: str
    path: str
    source_run: str
    source_checkpoint: str
    parent_model: str | None
    training_source: str
    step_id: int
    episode_count: int
    reward_ema: float
    temporal_core: str
    corpus_summary: str
    promoted_at: str
    notes: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Registry:
    """Top-level registry structure."""

    format_version: int = 1
    latest: str = ""
    models: list[ModelEntry] = field(default_factory=list)


class ModelRegistry:
    """Manage promoted models and the ``registry.json`` catalog.

    Parameters
    ----------
    models_dir:
        Directory where promoted models and ``registry.json`` live.
        Defaults to ``artifacts/models``.
    """

    def __init__(self, models_dir: Path | str = _DEFAULT_MODELS_DIR) -> None:
        self._models_dir = Path(models_dir)
        self._registry_path = self._models_dir / "registry.json"

    # ── Public API ────────────────────────────────────────────────

    def promote(
        self,
        checkpoint_path: str | Path,
        *,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> ModelEntry:
        """Promote a checkpoint into the registry.

        Copies the checkpoint to ``artifacts/models/vNNN.pt``, updates
        ``latest.pt``, and writes the registry catalog.

        Returns the new :class:`ModelEntry`.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint to extract metadata
        data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        if not isinstance(data, dict) or data.get("version") != 3:
            raise RuntimeError(
                "Cannot promote: checkpoint must be v3 canonical format"
            )

        registry = self._load_registry()

        # Determine next version number
        existing_ids = [m.id for m in registry.models]
        next_num = 1
        for mid in existing_ids:
            if mid.startswith("v") and mid[1:].isdigit():
                next_num = max(next_num, int(mid[1:]) + 1)
        version_id = f"v{next_num:03d}"

        # Copy checkpoint to models directory
        self._models_dir.mkdir(parents=True, exist_ok=True)
        dest_path = self._models_dir / f"{version_id}.pt"
        shutil.copy2(checkpoint_path, dest_path)

        # Update latest.pt
        latest_path = self._models_dir / "latest.pt"
        shutil.copy2(checkpoint_path, latest_path)

        # Determine parent model from checkpoint lineage
        parent_checkpoint = data["parent_checkpoint"]
        parent_model = self._find_model_by_source_checkpoint(
            registry, parent_checkpoint
        )

        entry = ModelEntry(
            id=version_id,
            path=str(dest_path),
            source_run=data["run_id"],
            source_checkpoint=str(checkpoint_path),
            parent_model=parent_model,
            training_source=data["training_source"],
            step_id=int(data["step_id"]),
            episode_count=int(data["episode_count"]),
            reward_ema=float(data["reward_ema"]),
            temporal_core=data["temporal_core"],
            corpus_summary=data["corpus_summary"],
            promoted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            notes=notes,
            tags=tags or [],
        )

        registry.models.append(entry)
        registry.latest = str(latest_path)
        self._save_registry(registry)

        _LOGGER.info(
            "Promoted %s → %s (step=%d, reward_ema=%.4f, source=%s)",
            checkpoint_path,
            version_id,
            entry.step_id,
            entry.reward_ema,
            entry.training_source,
        )
        return entry

    def get_latest(self) -> Path | None:
        """Return the path to ``latest.pt``, or ``None`` if no models promoted."""
        latest = self._models_dir / "latest.pt"
        if latest.exists():
            return latest
        return None

    def get_latest_entry(self) -> ModelEntry | None:
        """Return the :class:`ModelEntry` for the latest promoted model."""
        registry = self._load_registry()
        if not registry.models:
            return None
        return registry.models[-1]

    def list_models(self) -> list[ModelEntry]:
        """Return all promoted models, oldest first."""
        registry = self._load_registry()
        return list(registry.models)

    def get_model(self, version_id: str) -> ModelEntry | None:
        """Look up a model by version ID (e.g. ``"v001"``)."""
        registry = self._load_registry()
        for m in registry.models:
            if m.id == version_id:
                return m
        return None

    def get_model_path(self, version_id: str) -> Path | None:
        """Return the file path for a given version ID."""
        entry = self.get_model(version_id)
        if entry is None:
            return None
        p = Path(entry.path)
        return p if p.exists() else None

    # ── Internal ──────────────────────────────────────────────────

    def _load_registry(self) -> Registry:
        if not self._registry_path.exists():
            return Registry()
        try:
            raw = json.loads(self._registry_path.read_text(encoding="utf-8"))
            models = [ModelEntry(**m) for m in raw.get("models", [])]
            return Registry(
                format_version=raw.get("format_version", 1),
                latest=raw.get("latest", ""),
                models=models,
            )
        except (json.JSONDecodeError, TypeError, KeyError):
            _LOGGER.warning("Corrupt registry at %s — starting fresh", self._registry_path)
            return Registry()

    def _save_registry(self, registry: Registry) -> None:
        self._models_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format_version": registry.format_version,
            "latest": registry.latest,
            "models": [asdict(m) for m in registry.models],
        }
        self._registry_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    def _find_model_by_source_checkpoint(
        self, registry: Registry, checkpoint_path: str | None
    ) -> str | None:
        if not checkpoint_path:
            return None
        for m in registry.models:
            if m.source_checkpoint == checkpoint_path or m.path == checkpoint_path:
                return m.id
        return None
