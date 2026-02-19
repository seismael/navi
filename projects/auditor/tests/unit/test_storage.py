"""Tests for storage backend."""

from __future__ import annotations

import tempfile
from pathlib import Path

from navi_auditor.storage.base import AbstractStorageBackend
from navi_auditor.storage.zarr_backend import ZarrBackend


class TestZarrBackend:
    """Tests for ZarrBackend."""

    def test_implements_abstract(self) -> None:
        assert isinstance(ZarrBackend(), AbstractStorageBackend)

    def test_write_and_read(self) -> None:
        with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
            path = str(Path(tmp_dir) / "test.zarr")

            backend = ZarrBackend()
            backend.open(path, mode="w")
            backend.write("test_topic", b"hello world", 1.0)
            backend.write("test_topic", b"second message", 2.0)
            assert len(backend) == 2
            backend.close()

            backend2 = ZarrBackend()
            backend2.open(path, mode="r")
            messages = backend2.read_all()
            assert len(messages) == 2
            assert messages[0][0] == "test_topic"
            assert messages[0][2] == 1.0
            backend2.close()
