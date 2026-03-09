"""Tests for FileGenerator sparse chunk world loading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from navi_environment.generators.file_loader import FileGenerator
from navi_environment.transformers import WorldCompileConfig, WorldModelCompiler


def test_file_generator_reads_sparse_chunks() -> None:
    with tempfile.TemporaryDirectory(dir=str(Path.cwd())) as tmp_dir:
        tmp_path = Path(tmp_dir)
        source = tmp_path / "shape.ply"
        source.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 2",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "1 0 0",
                ]
            ),
            encoding="utf-8",
        )

        output = tmp_path / "world.zarr"
        compiler = WorldModelCompiler()
        compiler.compile(
            source_path=source,
            output_path=output,
            config=WorldCompileConfig(chunk_size=8, voxel_size=1.0, semantic_id=4),
        )

        generator = FileGenerator(path=output, chunk_size=8)
        chunk = generator.generate_chunk(0, 0, 0)

        assert chunk.shape == (8, 8, 8, 2)
        occupied = chunk[..., 0] > 0
        assert int(np.count_nonzero(occupied)) == 2
        assert np.all(chunk[..., 1][occupied] == 4.0)
