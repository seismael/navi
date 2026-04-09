"""Tests for the model registry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from navi_actor.model_registry import ModelRegistry


def _make_v3_checkpoint(path: Path, *, step_id: int = 100, reward_ema: float = -0.5) -> None:
    """Write a minimal v3 checkpoint for testing."""
    state = {
        "version": 3,
        "run_id": "test-run",
        "step_id": step_id,
        "episode_count": step_id // 10,
        "reward_ema": reward_ema,
        "wall_time_hours": 0.1,
        "parent_checkpoint": None,
        "training_source": "rl",
        "temporal_core": "mamba2",
        "corpus_summary": "5 scenes",
        "created_at": "2026-04-09T12:00:00Z",
        "policy_state_dict": {"w": torch.randn(4, 4)},
        "rnd_state_dict": {"w": torch.randn(4, 4)},
        "reward_shaper_step": step_id,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def test_promote_creates_versioned_copy_and_latest() -> None:
    """Promoting a checkpoint should create vNNN.pt, latest.pt, and registry.json."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        ckpt = Path(tmpdir) / "source" / "checkpoint.pt"
        _make_v3_checkpoint(ckpt, step_id=50000, reward_ema=-0.85)

        registry = ModelRegistry(models_dir)
        entry = registry.promote(ckpt, notes="first model", tags=["test"])

        assert entry.id == "v001"
        assert entry.step_id == 50000
        assert entry.reward_ema == -0.85
        assert entry.training_source == "rl"
        assert entry.notes == "first model"
        assert "test" in entry.tags

        # Files should exist
        assert (models_dir / "v001.pt").exists()
        assert (models_dir / "latest.pt").exists()
        assert (models_dir / "registry.json").exists()


def test_promote_increments_version_numbers() -> None:
    """Each promotion should get a sequential version number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(models_dir)

        for i in range(3):
            ckpt = Path(tmpdir) / f"ckpt_{i}.pt"
            _make_v3_checkpoint(ckpt, step_id=(i + 1) * 10000, reward_ema=-1.0 + i * 0.1)
            entry = registry.promote(ckpt)
            assert entry.id == f"v{i + 1:03d}"

        entries = registry.list_models()
        assert len(entries) == 3
        assert [e.id for e in entries] == ["v001", "v002", "v003"]


def test_get_latest_returns_most_recent() -> None:
    """get_latest should return the latest.pt path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(models_dir)

        assert registry.get_latest() is None

        ckpt = Path(tmpdir) / "ckpt.pt"
        _make_v3_checkpoint(ckpt)
        registry.promote(ckpt)

        latest = registry.get_latest()
        assert latest is not None
        assert latest.name == "latest.pt"
        assert latest.exists()


def test_get_latest_entry_returns_metadata() -> None:
    """get_latest_entry should return the ModelEntry for the latest model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(models_dir)

        assert registry.get_latest_entry() is None

        ckpt = Path(tmpdir) / "ckpt.pt"
        _make_v3_checkpoint(ckpt, step_id=25000, reward_ema=-0.9)
        registry.promote(ckpt, notes="test entry")

        entry = registry.get_latest_entry()
        assert entry is not None
        assert entry.step_id == 25000
        assert entry.reward_ema == -0.9
        assert entry.notes == "test entry"


def test_get_model_by_version() -> None:
    """Should be able to look up a model by version ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(models_dir)

        ckpt = Path(tmpdir) / "ckpt.pt"
        _make_v3_checkpoint(ckpt)
        registry.promote(ckpt)

        assert registry.get_model("v001") is not None
        assert registry.get_model("v999") is None


def test_promote_v2_checkpoint_is_rejected() -> None:
    """Promoting a v2 checkpoint must fail — only v3 is accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        ckpt = Path(tmpdir) / "v2_ckpt.pt"
        state = {
            "version": 2,
            "run_id": "legacy-run",
            "policy_state_dict": {"w": torch.randn(4, 4)},
            "rnd_state_dict": {"w": torch.randn(4, 4)},
            "reward_shaper_step": 500,
        }
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt)

        registry = ModelRegistry(models_dir)
        import pytest
        with pytest.raises(RuntimeError, match="v3 canonical format"):
            registry.promote(ckpt)


def test_registry_survives_reload() -> None:
    """Registry should be loadable from disk after being written."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"

        ckpt = Path(tmpdir) / "ckpt.pt"
        _make_v3_checkpoint(ckpt, step_id=99999)

        registry1 = ModelRegistry(models_dir)
        registry1.promote(ckpt, notes="persist test")

        # Create new instance (simulates restart)
        registry2 = ModelRegistry(models_dir)
        entries = registry2.list_models()
        assert len(entries) == 1
        assert entries[0].step_id == 99999
        assert entries[0].notes == "persist test"


def test_parent_lineage_tracked() -> None:
    """When a promoted model's source checkpoint matches a prior model, the parent should be recorded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = Path(tmpdir) / "models"
        registry = ModelRegistry(models_dir)

        # First model
        ckpt1 = Path(tmpdir) / "ckpt1.pt"
        _make_v3_checkpoint(ckpt1, step_id=10000)
        entry1 = registry.promote(ckpt1)

        # Second model with parent_checkpoint pointing to promoted model's path
        state = torch.load(ckpt1, weights_only=False, map_location="cpu")
        state["step_id"] = 50000
        state["parent_checkpoint"] = entry1.path  # points to v001.pt
        ckpt2 = Path(tmpdir) / "ckpt2.pt"
        torch.save(state, ckpt2)

        entry2 = registry.promote(ckpt2)
        assert entry2.parent_model == "v001"
