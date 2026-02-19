"""Typer CLI for the Section Manager service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer

from navi_section_manager.config import SectionManagerConfig
from navi_section_manager.generators.arena import ArenaGenerator
from navi_section_manager.generators.city import CityGenerator
from navi_section_manager.generators.file_loader import FileGenerator
from navi_section_manager.generators.maze import MazeGenerator
from navi_section_manager.generators.open3d_voxel import Open3DVoxelGenerator
from navi_section_manager.generators.rooms import RoomsGenerator
from navi_section_manager.server import SectionManagerServer
from navi_section_manager.transformers import WorldCompileConfig, WorldModelCompiler

if TYPE_CHECKING:
    from navi_section_manager.generators.base import AbstractWorldGenerator

__all__: list[str] = ["app"]

app = typer.Typer(
    name="navi-section-manager",
    help="Layer 2: The Engine — Section Manager service",
)


def _build_generator(config: SectionManagerConfig) -> AbstractWorldGenerator:
    """Instantiate the correct world generator from config."""
    if config.world_source == "file":
        if not config.world_file:
            typer.echo("Error: --world-file is required when --world-source=file", err=True)
            raise typer.Exit(code=1)
        return FileGenerator(path=config.world_file, chunk_size=config.chunk_size)

    # Procedural generators
    if config.generator == "arena":
        return ArenaGenerator(
            seed=config.seed,
            chunk_size=config.chunk_size,
        )
    if config.generator == "city":
        return CityGenerator(
            seed=config.seed,
            chunk_size=config.chunk_size,
        )
    if config.generator == "maze":
        return MazeGenerator(
            seed=config.seed,
            chunk_size=config.chunk_size,
            complexity=0.5,
        )
    if config.generator == "open3d":
        return Open3DVoxelGenerator(
            seed=config.seed,
            chunk_size=config.chunk_size,
        )
    if config.generator == "rooms":
        return RoomsGenerator(
            seed=config.seed,
            chunk_size=config.chunk_size,
        )

    typer.echo(f"Unknown generator: {config.generator}", err=True)
    raise typer.Exit(code=1)


@app.command()
def serve(
    pub: str = typer.Option("tcp://*:5559", help="ZMQ PUB bind address (DistanceMatrix v2)"),
    rep: str = typer.Option("tcp://*:5560", help="ZMQ REP bind address (StepRequest/StepResult)"),
    action_sub: str = typer.Option(
        "tcp://localhost:5557",
        help="ZMQ SUB address for Action (async mode)",
    ),
    mode: str = typer.Option("step", help="Mode: step (REQ/REP) or async (PUB/SUB)"),
    world_source: str = typer.Option("procedural", help="World source: procedural or file"),
    world_file: str = typer.Option("", help="Path to world file (.zarr)"),
    generator: str = typer.Option("arena", help="Generator type: arena, city, maze, open3d, rooms"),
    window_radius: int = typer.Option(2, help="Window radius in chunk units"),
    lookahead_margin: int = typer.Option(8, help="Look-ahead margin in chunks"),
    seed: int = typer.Option(42, help="Random seed for procedural generation"),
    chunk_size: int = typer.Option(16, help="Chunk side length in voxels"),
    barrier_distance: float = typer.Option(0.75, help="Minimum standoff to occupied voxels"),
    collision_probe_radius: float = typer.Option(
        1.5, help="Collision probe radius around candidate pose"
    ),
    azimuth_bins: int = typer.Option(256, help="Distance-matrix azimuth bins"),
    elevation_bins: int = typer.Option(128, help="Distance-matrix elevation bins"),
    max_distance: float = typer.Option(30.0, help="Distance normalization range"),
) -> None:
    """Start the Section Manager service."""
    config = SectionManagerConfig(
        pub_address=pub,
        rep_address=rep,
        action_sub_address=action_sub,
        mode=mode,
        world_source=world_source,
        world_file=world_file,
        generator=generator,
        window_radius=window_radius,
        lookahead_margin=lookahead_margin,
        seed=seed,
        chunk_size=chunk_size,
        barrier_distance=barrier_distance,
        collision_probe_radius=collision_probe_radius,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        max_distance=max_distance,
    )

    gen = _build_generator(config)
    server = SectionManagerServer(config=config, generator=gen)

    typer.echo(
        f"Section Manager starting — mode={config.mode}, "
        f"pub={config.pub_address}, rep={config.rep_address}, "
        f"generator={config.generator}, seed={config.seed}",
    )
    server.run()

@app.command("compile-world")
def compile_world(
    source: str = typer.Option(..., "--source", help="Input source model path (.ply/.obj/.stl)"),
    output: str = typer.Option(..., "--output", help="Output .zarr world store path"),
    source_format: str = typer.Option(
        "auto",
        "--source-format",
        help="Source format: auto, ply, obj, stl",
    ),
    chunk_size: int = typer.Option(16, help="Output chunk side length in voxels"),
    voxel_size: float = typer.Option(1.0, help="Voxel size for quantization"),
    semantic_id: int = typer.Option(6, help="Semantic ID to assign to occupied voxels"),
) -> None:
    """Compile a source world model into canonical sparse voxel chunks."""
    compiler = WorldModelCompiler()
    result = compiler.compile(
        source_path=source,
        output_path=output,
        config=WorldCompileConfig(
            chunk_size=chunk_size,
            voxel_size=voxel_size,
            semantic_id=semantic_id,
            source_format=source_format,
        ),
    )
    typer.echo(
        "Compiled world — "
        f"voxels={result.occupied_voxels}, "
        f"chunks={result.chunk_count}, "
        f"spawn={result.spawn_position}, "
        f"out={result.output_path}",
    )


if __name__ == "__main__":
    app()
