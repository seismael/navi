"""Tests for the standalone .gmdag reader."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from navi_inspector.gmdag_io import GmdagAsset, gmdag_info, load_gmdag

_HEADER = struct.Struct("<4sIIffffI")


def _write_gmdag(
    path: Path,
    *,
    magic: bytes = b"GDAG",
    version: int = 1,
    resolution: int = 8,
    bmin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    voxel_size: float = 0.125,
    nodes: np.ndarray | None = None,
    trailing: bytes = b"",
) -> Path:
    """Helper: write a minimal valid (or intentionally broken) .gmdag file."""
    if nodes is None:
        # One leaf node: bit 63 set, fp16 distance = 0.5
        leaf = np.uint64((1 << 63) | int(np.float16(0.5).view(np.uint16)))
        nodes = np.array([leaf], dtype=np.uint64)

    header = _HEADER.pack(
        magic, version, resolution, bmin[0], bmin[1], bmin[2], voxel_size, len(nodes)
    )
    path.write_bytes(header + nodes.tobytes() + trailing)
    return path


class TestLoadGmdag:
    """load_gmdag() happy path and validation gates."""

    def test_loads_valid_file(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "test.gmdag")
        asset = load_gmdag(fp)
        assert isinstance(asset, GmdagAsset)
        assert asset.version == 1
        assert asset.resolution == 8
        assert asset.voxel_size == pytest.approx(0.125)
        assert asset.bbox_min == (0.0, 0.0, 0.0)
        assert asset.bbox_max[0] == pytest.approx(1.0)  # 8 * 0.125
        assert len(asset.nodes) == 1
        assert asset.nodes.dtype == np.uint64

    def test_rejects_bad_magic(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", magic=b"XXXX")
        with pytest.raises(RuntimeError, match="magic"):
            load_gmdag(fp)

    def test_rejects_bad_version(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", version=99)
        with pytest.raises(RuntimeError, match="version"):
            load_gmdag(fp)

    def test_rejects_zero_resolution(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", resolution=0)
        with pytest.raises(RuntimeError, match="resolution"):
            load_gmdag(fp)

    def test_rejects_nan_voxel_size(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", voxel_size=float("nan"))
        with pytest.raises(RuntimeError, match="voxel size"):
            load_gmdag(fp)

    def test_rejects_negative_voxel_size(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", voxel_size=-1.0)
        with pytest.raises(RuntimeError, match="voxel size"):
            load_gmdag(fp)

    def test_rejects_trailing_bytes(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", trailing=b"\x00")
        with pytest.raises(RuntimeError, match="[Tt]railing"):
            load_gmdag(fp)

    def test_rejects_truncated_nodes(self, tmp_path: Path) -> None:
        # Write header claiming 10 nodes but only provide 1
        nodes = np.array([np.uint64(1 << 63)], dtype=np.uint64)
        fp = tmp_path / "trunc.gmdag"
        header = _HEADER.pack(b"GDAG", 1, 8, 0.0, 0.0, 0.0, 0.125, 10)
        fp.write_bytes(header + nodes.tobytes())
        with pytest.raises(RuntimeError, match="[Tt]runcated|expected"):
            load_gmdag(fp)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gmdag(tmp_path / "nonexistent.gmdag")

    def test_rejects_inf_bbox(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "bad.gmdag", bmin=(float("inf"), 0.0, 0.0))
        with pytest.raises(RuntimeError, match="finite|bbox"):
            load_gmdag(fp)


class TestGmdagInfo:
    """gmdag_info() metadata-only read."""

    def test_returns_metadata(self, tmp_path: Path) -> None:
        fp = _write_gmdag(tmp_path / "test.gmdag", resolution=16, voxel_size=0.25)
        meta = gmdag_info(fp)
        assert meta["version"] == 1
        assert meta["resolution"] == 16
        assert meta["voxel_size"] == pytest.approx(0.25)
        assert meta["node_count"] == 1
        assert meta["file_size_bytes"] == fp.stat().st_size
        assert "file_size_mb" in meta

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            gmdag_info(tmp_path / "missing.gmdag")
