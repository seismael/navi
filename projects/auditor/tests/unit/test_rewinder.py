"""Tests for the Rewinder."""

from __future__ import annotations

from navi_auditor.config import AuditorConfig
from navi_auditor.rewinder import Rewinder
from navi_auditor.storage.zarr_backend import ZarrBackend


class TestRewinder:
    """Tests for Rewinder."""

    def test_creation(self) -> None:
        config = AuditorConfig()
        backend = ZarrBackend()
        rewinder = Rewinder(config=config, backend=backend)
        assert rewinder is not None
