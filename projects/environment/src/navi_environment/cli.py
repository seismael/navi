"""Typer CLI for the Environment service."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import typer

from navi_contracts import Action, setup_logging
from navi_environment.config import EnvironmentConfig
from navi_environment.generators.arena import ArenaGenerator
from navi_environment.generators.city import CityGenerator
from navi_environment.generators.file_loader import FileGenerator
from navi_environment.generators.maze import MazeGenerator
from navi_environment.generators.open3d_voxel import Open3DVoxelGenerator
from navi_environment.generators.rooms import RoomsGenerator
from navi_environment.integration.corpus import prepare_training_scene_corpus
from navi_environment.integration.voxel_dag import compile_gmdag_world, probe_sdfdag_runtime
from navi_environment.server import EnvironmentServer
from navi_environment.transformers import WorldCompileConfig, WorldModelCompiler

if TYPE_CHECKING:
    from navi_environment.backends.base import SimulatorBackend
    from navi_environment.generators.base import AbstractWorldGenerator

__all__: list[str] = ["app"]

app = typer.Typer(
    name="navi-environment",
    help="Layer 1: The Environment",
)

_DIAGNOSTIC_BACKENDS = {"voxel", "habitat", "mesh"}


def _default_gmdag_option() -> str:
    """Expose the canonical sample asset on user-facing launch surfaces."""
    return EnvironmentConfig().gmdag_file


def _build_generator(config: EnvironmentConfig) -> AbstractWorldGenerator:
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


def _build_backend(config: EnvironmentConfig) -> SimulatorBackend:
    """Create the appropriate SimulatorBackend based on config.backend."""
    if config.backend == "sdfdag":
        from navi_environment.backends.sdfdag_backend import SdfDagBackend

        if not config.gmdag_file:
            typer.echo("Error: --gmdag-file is required when --backend=sdfdag", err=True)
            raise typer.Exit(code=1)

        status = probe_sdfdag_runtime(Path(config.gmdag_file))
        if status.issues:
            for issue in status.issues:
                typer.echo(f"Error: {issue}", err=True)
            raise typer.Exit(code=1)

        return SdfDagBackend(config)

    if config.backend == "habitat":
        try:
            from navi_environment.backends.habitat_backend import HabitatBackend
        except ImportError as exc:
            typer.echo(
                "Error: habitat-sim is not installed. "
                "Install it with: conda install -c aihabitat habitat-sim",
                err=True,
            )
            raise typer.Exit(code=1) from exc

        if not config.habitat_scene:
            typer.echo("Error: --habitat-scene is required when --backend=habitat", err=True)
            raise typer.Exit(code=1)

        return HabitatBackend(config)

    if config.backend == "mesh":
        from navi_environment.backends.mesh_backend import MeshSceneBackend

        if not config.habitat_scene:
            typer.echo("Error: --habitat-scene is required when --backend=mesh", err=True)
            raise typer.Exit(code=1)

        return MeshSceneBackend(config)

    if config.backend == "voxel":
        from navi_environment.backends.voxel import VoxelBackend

        generator = _build_generator(config)
        return VoxelBackend(config, generator)

    typer.echo(
        "Error: unsupported backend. Canonical runtime requires backend=sdfdag; "
        "voxel, mesh, and habitat are diagnostic-only and must be requested explicitly.",
        err=True,
    )
    raise typer.Exit(code=1)


def _benchmark_actions(*, actor_count: int, step_id: int) -> tuple[Action, ...]:
    actions: list[Action] = []
    timestamp = time.time()
    for actor_id in range(actor_count):
        yaw_direction = -0.15 if actor_id % 2 else 0.15
        actions.append(
            Action(
                env_ids=np.array([actor_id], dtype=np.int32),
                linear_velocity=np.array([[1.5, 0.0, 0.0]], dtype=np.float32),
                angular_velocity=np.array([[0.0, 0.0, yaw_direction]], dtype=np.float32),
                policy_id="bench-sdfdag",
                step_id=step_id,
                timestamp=timestamp,
            ),
        )
    return tuple(actions)


@app.command()
def serve(
    pub: str = typer.Option(None, help="ZMQ PUB bind address (DistanceMatrix v2)"),
    rep: str = typer.Option(None, help="ZMQ REP bind address (StepRequest/StepResult)"),
    action_sub: str = typer.Option(
        None,
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
    elevation_bins: int = typer.Option(48, help="Distance-matrix elevation bins"),
    max_distance: float = typer.Option(30.0, help="Distance normalization range"),
    backend: str = typer.Option(
        "sdfdag",
        help="Simulator backend: sdfdag (canonical) or explicit diagnostic backend voxel|mesh|habitat",
    ),
    gmdag_file: str = typer.Option(
        _default_gmdag_option(),
        help="Path to compiled .gmdag world cache for the canonical sdfdag backend",
    ),
    sdf_max_steps: int = typer.Option(256, help="Maximum sphere-tracing iterations per ray for sdfdag backend"),
    habitat_scene: str = typer.Option("", help="Habitat scene file (.glb)"),
    habitat_dataset_config: str = typer.Option(
        "", help="Habitat dataset config (.json) for PointNav episodes"
    ),
    habitat_rgb_height: int = typer.Option(480, help="Habitat RGB camera height"),
    habitat_rgb_width: int = typer.Option(640, help="Habitat RGB camera width"),
    actors: int = typer.Option(1, help="Number of actors sharing the scene"),
) -> None:
    """Start the Environment service."""
    setup_logging("navi_environment")
    default_config = EnvironmentConfig()

    if backend in _DIAGNOSTIC_BACKENDS:
        typer.echo(
            f"Warning: backend={backend} is diagnostic-only. Canonical production runtime uses backend=sdfdag.",
            err=True,
        )

    config = EnvironmentConfig(
        pub_address=pub or default_config.pub_address,
        rep_address=rep or default_config.rep_address,
        action_sub_address=action_sub or default_config.action_sub_address,
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
        n_actors=actors,
        backend=backend,
        gmdag_file=gmdag_file,
        sdf_max_steps=sdf_max_steps,
        habitat_scene=habitat_scene,
        habitat_dataset_config=habitat_dataset_config,
        habitat_rgb_resolution=(habitat_rgb_height, habitat_rgb_width),
    )

    sim_backend = _build_backend(config)
    server = EnvironmentServer(config=config, backend=sim_backend)

    typer.echo(
        f"Environment starting — backend={config.backend}, mode={config.mode}, "
        f"actors={config.n_actors}, "
        f"pub={config.pub_address}, rep={config.rep_address}, "
        f"generator={config.generator}, seed={config.seed}",
    )
    server.run()


def serve_shortcut() -> None:
    """Shortcut for 'navi-environment serve' command."""
    app(["serve"])


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
    setup_logging("navi_environment_compiler")
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


@app.command("compile-gmdag")
def compile_gmdag(
    source: str = typer.Option(..., "--source", help="Input source model path (.glb/.obj/.ply/.stl)"),
    output: str = typer.Option(..., "--output", help="Output .gmdag world cache path"),
    resolution: int = typer.Option(2048, help="Compiler voxel resolution for DAG generation"),
) -> None:
    """Compile a source model into a `.gmdag` cache via the internal voxel-dag project."""
    setup_logging("navi_environment_gmdag_compiler")
    result = compile_gmdag_world(
        source_path=Path(source),
        output_path=Path(output),
        resolution=resolution,
    )
    typer.echo(
        "Compiled gmdag — "
        f"source={result.source_path}, "
        f"output={result.output_path}, "
        f"resolution={result.resolution}, "
        f"command={' '.join(result.command)}",
    )


@app.command("prepare-corpus")
def prepare_corpus(
    scene: str = typer.Option("", help="Explicit scene name or path override"),
    manifest: str = typer.Option("", help="Manifest of source or compiled scenes"),
    corpus_root: str = typer.Option("", help="Root directory for source-scene discovery"),
    gmdag_root: str = typer.Option("", help="Root directory for compiled `.gmdag` outputs"),
    resolution: int = typer.Option(2048, help="Compiler voxel resolution for corpus preparation"),
    min_scene_bytes: int = typer.Option(1000, help="Ignore tiny invalid scene files"),
    force_recompile: bool = typer.Option(False, help="Overwrite compiled `.gmdag` outputs during refresh"),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON summary instead of text"),
) -> None:
    """Discover, compile, and summarize the canonical training corpus."""
    setup_logging("navi_environment_prepare_corpus")

    prepared = prepare_training_scene_corpus(
        scene=scene,
        manifest_path=Path(manifest) if manifest else None,
        source_root=Path(corpus_root) if corpus_root else None,
        gmdag_root=Path(gmdag_root) if gmdag_root else None,
        resolution=resolution,
        min_scene_bytes=min_scene_bytes,
        force_recompile=force_recompile,
    )

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "source_root": str(prepared.source_root),
                    "gmdag_root": str(prepared.gmdag_root),
                    "source_manifest": str(prepared.source_manifest_path),
                    "compiled_manifest": str(prepared.compiled_manifest_path),
                    "scene_count": len(prepared.scene_entries),
                    "scenes": [
                        {
                            "scene_name": entry.scene_name,
                            "dataset": entry.dataset,
                            "source_path": entry.source_path.as_posix(),
                            "gmdag_path": entry.compiled_path.as_posix(),
                        }
                        for entry in prepared.scene_entries
                    ],
                },
                indent=2,
            )
        )
        return

    typer.echo(
        "Prepared corpus - "
        f"source_root={prepared.source_root}, "
        f"gmdag_root={prepared.gmdag_root}, "
        f"scenes={len(prepared.scene_entries)}"
    )
    typer.echo(f"source_manifest={prepared.source_manifest_path}")
    typer.echo(f"compiled_manifest={prepared.compiled_manifest_path}")
    for entry in prepared.scene_entries[:10]:
        typer.echo(
            f"scene={entry.scene_name} dataset={entry.dataset} gmdag={entry.compiled_path}"
        )
    if len(prepared.scene_entries) > 10:
        typer.echo(f"... and {len(prepared.scene_entries) - 10} more")


@app.command("check-sdfdag")
def check_sdfdag(
    gmdag_file: str = typer.Option("", help=".gmdag asset to validate alongside runtime readiness"),
) -> None:
    """Validate the canonical SDF/DAG compiler/runtime path and asset metadata."""
    setup_logging("navi_environment_sdfdag_check")
    status = probe_sdfdag_runtime(Path(gmdag_file) if gmdag_file else None)

    typer.echo(
        "SDF/DAG runtime - "
        f"compiler_ready={status.compiler_ready}, "
        f"torch_ready={status.torch_ready}, "
        f"cuda_ready={status.cuda_ready}, "
        f"torch_sdf_ready={status.torch_sdf_ready}, "
        f"asset_loaded={status.asset_loaded}",
    )
    if status.compiler_path is not None:
        typer.echo(f"compiler={status.compiler_path}")
    if status.gmdag_path is not None:
        typer.echo(f"gmdag={status.gmdag_path}")
    if status.asset_loaded:
        typer.echo(
            "asset - "
            f"resolution={status.resolution}, "
            f"nodes={status.node_count}, "
            f"bbox_min={status.bbox_min}, "
            f"bbox_max={status.bbox_max}",
        )
    if status.issues:
        for issue in status.issues:
            typer.echo(f"issue={issue}", err=True)
        raise typer.Exit(code=1)


@app.command("bench-sdfdag")
def bench_sdfdag(
    gmdag_file: str = typer.Option(..., help="Path to compiled .gmdag world cache"),
    actors: int = typer.Option(4, min=1, help="Number of actors to benchmark in one batch"),
    steps: int = typer.Option(200, min=1, help="Measured batch steps after warmup"),
    warmup_steps: int = typer.Option(25, min=0, help="Unmeasured warmup batch steps"),
    azimuth_bins: int = typer.Option(256, help="Distance-matrix azimuth bins"),
    elevation_bins: int = typer.Option(48, help="Distance-matrix elevation bins"),
    max_distance: float = typer.Option(30.0, help="Distance normalization range"),
    sdf_max_steps: int = typer.Option(256, help="Maximum sphere-tracing iterations per ray"),
) -> None:
    """Benchmark the canonical batched SDF/DAG environment path."""
    setup_logging("navi_environment_sdfdag_bench")
    default_config = EnvironmentConfig()
    config = EnvironmentConfig(
        pub_address=default_config.pub_address,
        rep_address=default_config.rep_address,
        action_sub_address=default_config.action_sub_address,
        mode="step",
        training_mode=True,
        backend="sdfdag",
        gmdag_file=gmdag_file,
        sdf_max_steps=sdf_max_steps,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        max_distance=max_distance,
        n_actors=actors,
    )
    backend = _build_backend(config)
    try:
        for actor_id in range(actors):
            backend.reset(episode_id=0, actor_id=actor_id)

        for step_id in range(warmup_steps):
            backend.batch_step(_benchmark_actions(actor_count=actors, step_id=step_id), step_id)

        started_at = time.perf_counter()
        for step_id in range(warmup_steps, warmup_steps + steps):
            backend.batch_step(_benchmark_actions(actor_count=actors, step_id=step_id), step_id)
        elapsed = time.perf_counter() - started_at
        snapshot = backend.perf_snapshot()
    finally:
        backend.close()

    if snapshot is None:
        typer.echo("Error: selected backend does not expose perf snapshots", err=True)
        raise typer.Exit(code=1)

    total_actor_steps = steps * actors
    measured_sps = total_actor_steps / elapsed if elapsed > 0.0 else 0.0
    typer.echo(
        "bench-sdfdag - "
        f"actors={actors}, "
        f"steps={steps}, "
        f"warmup_steps={warmup_steps}, "
        f"measured_sps={measured_sps:.2f}, "
        f"rolling_sps={snapshot.sps:.2f}, "
        f"last_batch_ms={snapshot.last_batch_step_ms:.2f}, "
        f"ema_batch_ms={snapshot.ema_batch_step_ms:.2f}, "
        f"avg_batch_ms={snapshot.avg_batch_step_ms:.2f}, "
        f"avg_actor_ms={snapshot.avg_actor_step_ms:.2f}, "
        f"total_batches={snapshot.total_batches}, "
        f"total_actor_steps={snapshot.total_actor_steps}"
    )


if __name__ == "__main__":
    app()
