"""Tests for offline world compiler."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from navi_section_manager.transformers import PlyCompileConfig, PlyWorldCompiler


def test_compile_ascii_ply_to_sparse_world() -> None:
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        tmp_path = Path(tmp_dir)
        source = tmp_path / "mini_world.ply"
        source.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 4",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                    "0 1 0",
                    "0 0 1",
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "compiled_world.zarr"
        compiler = PlyWorldCompiler()
        result = compiler.compile(
            source_path=source,
            output_path=output,
            config=PlyCompileConfig(chunk_size=8, voxel_size=1.0, semantic_id=6),
        )

        assert result.occupied_voxels == 4
        assert result.chunk_count >= 1

        store = zarr.open_group(str(output), mode="r")
        assert store.attrs["representation"] == "sparse_chunks"
        assert int(store.attrs["chunk_size"]) == 8
        chunk_index = np.asarray(store["chunk_index"][:], dtype=np.int32)
        assert chunk_index.shape[1] == 3
        assert chunk_index.shape[0] >= 1


def test_compile_obj_to_sparse_world() -> None:
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        tmp_path = Path(tmp_dir)
        source = tmp_path / "mini_world.obj"
        source.write_text(
            "\n".join(
                [
                    "v 0 0 0",
                    "v 1 0 0",
                    "v 0 1 0",
                    "v 0 0 1",
                    "f 1 2 3",
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "compiled_obj.zarr"
        compiler = PlyWorldCompiler()
        result = compiler.compile(
            source_path=source,
            output_path=output,
            config=PlyCompileConfig(
                chunk_size=8,
                voxel_size=1.0,
                semantic_id=6,
                source_format="obj",
            ),
        )

        assert result.occupied_voxels == 4
        store = zarr.open_group(str(output), mode="r")
        assert store.attrs["representation"] == "sparse_chunks"


def test_compile_ascii_stl_to_sparse_world() -> None:
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        tmp_path = Path(tmp_dir)
        source = tmp_path / "mini_world.stl"
        source.write_text(
            "\n".join(
                [
                    "solid test",
                    "  facet normal 0 0 1",
                    "    outer loop",
                    "      vertex 0 0 0",
                    "      vertex 1 0 0",
                    "      vertex 0 1 0",
                    "    endloop",
                    "  endfacet",
                    "endsolid test",
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "compiled_stl.zarr"
        compiler = PlyWorldCompiler()
        result = compiler.compile(
            source_path=source,
            output_path=output,
            config=PlyCompileConfig(
                chunk_size=8,
                voxel_size=1.0,
                semantic_id=6,
                source_format="stl",
            ),
        )

        assert result.occupied_voxels == 3
        store = zarr.open_group(str(output), mode="r")
        assert store.attrs["representation"] == "sparse_chunks"
