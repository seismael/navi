"""Phase 3 — .gmdag Binary Format Round-Trip Tests.

Proves that write → read preserves all header fields and node data
exactly, with no information loss at the voxel-dag ↔ torch-sdf boundary.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]
# Add voxel-dag to path for compiler imports (torch-sdf venv lacks navi_contracts)
_VOXEL_DAG_SRC = str(_REPO / "projects" / "voxel-dag")
if _VOXEL_DAG_SRC not in sys.path:
    sys.path.insert(0, _VOXEL_DAG_SRC)


def _load_oracle_box() -> Any:
    helper_path = _REPO / "projects" / "contracts" / "src" / "navi_contracts" / "testing" / "oracle_box.py"
    spec = importlib.util.spec_from_file_location("navi_test_oracle_box", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load oracle_box from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_oracle = _load_oracle_box()
write_unit_box_obj: Callable[..., None] = _oracle.write_unit_box_obj

from voxel_dag.compiler import (  # noqa: E402
    BinaryGmdagWriter,
    EikonalSdfComputer,
    MeshIngestor,
    SvoDagCompressor,
)

_HEADER_STRUCT = struct.Struct("<4sIIffffI")


# ── Helpers ──────────────────────────────────────────────────────────

def _compile_unit_box_gmdag(
    tmp_path: Path, resolution: int = 32
) -> tuple[Path, np.ndarray, int, np.ndarray, float]:
    """Compile unit box to .gmdag.  Returns (path, dag_nodes, resolution, cube_min, voxel_size)."""
    obj = tmp_path / "unit_box.obj"
    write_unit_box_obj(obj)
    verts, idxs, bmin, bmax = MeshIngestor.load_obj(str(obj))
    grid, h, cube_min = EikonalSdfComputer().compute(
        verts, idxs, bmin, bmax, resolution, padding=0.0
    )
    semantic = np.ones_like(grid, dtype=np.int32)
    dag = SvoDagCompressor().compress(grid, resolution, cell_size=h, semantic_grid=semantic)
    out = tmp_path / "unit_box.gmdag"
    BinaryGmdagWriter().write(out, dag, resolution, cube_min, h)
    return out, dag, resolution, cube_min, h


def _load_gmdag_asset(path: Path) -> dict:
    """Minimal header + payload reader independent of environment code."""
    raw = path.read_bytes()
    magic, version, resolution, bx, by, bz, voxel_size, node_count = _HEADER_STRUCT.unpack(
        raw[: _HEADER_STRUCT.size]
    )
    payload = np.frombuffer(raw[_HEADER_STRUCT.size :], dtype=np.uint64)
    return {
        "magic": magic,
        "version": version,
        "resolution": resolution,
        "bbox_min": (bx, by, bz),
        "voxel_size": voxel_size,
        "node_count": node_count,
        "nodes": payload,
    }


# ── Tests ────────────────────────────────────────────────────────────

class TestHeaderRoundtrip:
    """Write-read preserves every header field exactly."""

    def test_magic_bytes(self, tmp_path: Path) -> None:
        path, *_ = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        assert loaded["magic"] == b"GDAG"

    def test_version(self, tmp_path: Path) -> None:
        path, *_ = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        assert loaded["version"] == 1

    def test_resolution_preserved(self, tmp_path: Path) -> None:
        path, dag, res, cube_min, h = _compile_unit_box_gmdag(tmp_path, resolution=32)
        loaded = _load_gmdag_asset(path)
        assert loaded["resolution"] == 32

    def test_bbox_min_preserved(self, tmp_path: Path) -> None:
        path, dag, res, cube_min, h = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        for i, (expected, actual) in enumerate(
            zip(cube_min.tolist(), loaded["bbox_min"], strict=True)
        ):
            assert abs(expected - actual) < 1e-7, (
                f"bbox_min[{i}]: expected {expected}, got {actual}"
            )

    def test_voxel_size_preserved(self, tmp_path: Path) -> None:
        path, dag, res, cube_min, h = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        assert abs(loaded["voxel_size"] - h) < 1e-7, (
            f"voxel_size: expected {h}, got {loaded['voxel_size']}"
        )

    def test_node_count_matches_dag_length(self, tmp_path: Path) -> None:
        path, dag, *_ = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        assert loaded["node_count"] == len(dag)


class TestNodeArrayRoundtrip:
    """Every node word survives the write-read cycle bit-identically."""

    def test_all_nodes_identical(self, tmp_path: Path) -> None:
        path, dag, *_ = _compile_unit_box_gmdag(tmp_path)
        loaded = _load_gmdag_asset(path)
        np.testing.assert_array_equal(
            loaded["nodes"],
            dag,
            err_msg="Node array mismatch after gmdag round-trip",
        )


class TestBboxMaxDerivation:
    """Verify that bbox_max = bbox_min + voxel_size × resolution matches
    the compiler's intended padded cubic domain."""

    def test_bbox_max_correct(self, tmp_path: Path) -> None:
        path, dag, res, cube_min, h = _compile_unit_box_gmdag(tmp_path, resolution=32)
        loaded = _load_gmdag_asset(path)
        extent = loaded["voxel_size"] * loaded["resolution"]
        for i in range(3):
            derived_max = loaded["bbox_min"][i] + extent
            expected_max = float(cube_min[i]) + h * 32
            assert abs(derived_max - expected_max) < 1e-6, (
                f"bbox_max[{i}]: derived {derived_max} vs expected {expected_max}"
            )

    def test_no_trailing_bytes(self, tmp_path: Path) -> None:
        """File must contain exactly header + node_count × 8 bytes."""
        path, dag, *_ = _compile_unit_box_gmdag(tmp_path)
        file_size = path.stat().st_size
        expected = _HEADER_STRUCT.size + len(dag) * 8
        assert file_size == expected, (
            f"File size {file_size} != expected {expected} "
            f"(header {_HEADER_STRUCT.size} + {len(dag)} × 8)"
        )


class TestGmdagRoundtripViaEnvironmentLoader:
    """If the environment loader is importable, validate it reads the same data."""

    def test_environment_loader_matches(self, tmp_path: Path) -> None:
        try:
            from navi_environment.integration.voxel_dag import load_gmdag_asset
        except ImportError:
            pytest.skip("navi_environment not importable")

        path, dag, res, cube_min, h = _compile_unit_box_gmdag(tmp_path, resolution=32)
        asset = load_gmdag_asset(path, validate_layout=True)

        assert asset.version == 1
        assert asset.resolution == 32
        assert abs(asset.voxel_size - h) < 1e-7
        for i in range(3):
            assert abs(asset.bbox_min[i] - float(cube_min[i])) < 1e-7
        np.testing.assert_array_equal(asset.nodes, dag)
