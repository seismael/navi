"""Tests for the Recorder node."""

from __future__ import annotations

from navi_auditor.config import AuditorConfig
from navi_auditor.recorder import Recorder
from navi_auditor.storage.zarr_backend import ZarrBackend


class TestRecorder:
    """Tests for Recorder."""

    def test_creation(self) -> None:
        config = AuditorConfig()
        backend = ZarrBackend()
        recorder = Recorder(config=config, backend=backend)
        assert recorder is not None
