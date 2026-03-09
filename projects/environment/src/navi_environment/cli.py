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
from navi_environment.integration.corpus import prepare_training_scene_corpus
from navi_environment.integration.voxel_dag import compile_gmdag_world, probe_sdfdag_runtime
from navi_environment.server import EnvironmentServer

if TYPE_CHECKING:
    from navi_environment.backends.base import SimulatorBackend

__all__: list[str] = ["app"]

app = typer.Typer(
    name="navi-environment",
    help="Layer 1: The Environment",
)

def _default_gmdag_option() -> str:
    """Expose the first discovered compiled corpus asset when available."""
    return EnvironmentConfig().gmdag_file


def _build_backend(config: EnvironmentConfig) -> SimulatorBackend:
    """Create the canonical SDF/DAG simulator backend."""
    if config.backend != "sdfdag":
        typer.echo("Error: unsupported backend. Canonical runtime requires backend=sdfdag.", err=True)
        raise typer.Exit(code=1)

    from navi_environment.backends.sdfdag_backend import SdfDagBackend

    if not config.gmdag_file and not config.scene_pool:
        typer.echo(
            "Error: no compiled .gmdag asset was resolved. "
            "Run `scripts/refresh-scene-corpus.ps1` or pass --gmdag-file explicitly.",
            err=True,
        )
        raise typer.Exit(code=1)

    status = probe_sdfdag_runtime(Path(config.gmdag_file) if config.gmdag_file else None)
    if status.issues:
        for issue in status.issues:
            typer.echo(f"Error: {issue}", err=True)
        raise typer.Exit(code=1)

    return SdfDagBackend(config)


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
    azimuth_bins: int = typer.Option(256, help="Distance-matrix azimuth bins"),
    elevation_bins: int = typer.Option(48, help="Distance-matrix elevation bins"),
    max_distance: float = typer.Option(30.0, help="Distance normalization range"),
    backend: str = typer.Option("sdfdag", hidden=True),
    gmdag_file: str = typer.Option(
        _default_gmdag_option(),
        help="Path to compiled .gmdag world cache for the canonical runtime",
    ),
    sdf_max_steps: int = typer.Option(256, help="Maximum sphere-tracing iterations per ray for sdfdag backend"),
    actors: int = typer.Option(1, help="Number of actors sharing the scene"),
) -> None:
    """Start the Environment service."""
    setup_logging("navi_environment")
    default_config = EnvironmentConfig()

    config = EnvironmentConfig(
        pub_address=pub or default_config.pub_address,
        rep_address=rep or default_config.rep_address,
        action_sub_address=action_sub or default_config.action_sub_address,
        mode=mode,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        max_distance=max_distance,
        n_actors=actors,
        backend=backend,
        gmdag_file=gmdag_file,
        sdf_max_steps=sdf_max_steps,
    )

    sim_backend = _build_backend(config)
    server = EnvironmentServer(config=config, backend=sim_backend)

    typer.echo(
        f"Environment starting — backend={config.backend}, mode={config.mode}, "
        f"actors={config.n_actors}, "
        f"pub={config.pub_address}, rep={config.rep_address}, "
        f"gmdag={config.gmdag_file}",
    )
    server.run()


def serve_shortcut() -> None:
    """Shortcut for 'navi-environment serve' command."""
    app(["serve"])


@app.command("compile-gmdag")
def compile_gmdag(
    source: str = typer.Option(..., "--source", help="Input source model path (.glb/.obj/.ply/.stl)"),
    output: str = typer.Option(..., "--output", help="Output .gmdag world cache path"),
    resolution: int = typer.Option(512, help="Compiler voxel resolution for DAG generation"),
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
    resolution: int = typer.Option(512, help="Compiler voxel resolution for corpus preparation"),
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
