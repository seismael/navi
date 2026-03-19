"""Typer CLI for the Environment service."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from navi_contracts import setup_logging
from navi_environment.config import EnvironmentConfig
from navi_environment.integration.corpus import (
    prepare_training_scene_corpus,
    validate_compiled_scene_corpus,
)
from navi_environment.integration.voxel_dag import compile_gmdag_world, probe_sdfdag_runtime
from navi_environment.server import EnvironmentServer

if TYPE_CHECKING:
    import torch

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
        typer.echo(
            "Error: unsupported backend. Canonical runtime requires backend=sdfdag.", err=True
        )
        raise typer.Exit(code=1)

    from navi_environment.backends.sdfdag_backend import SdfDagBackend

    if not config.gmdag_file and not config.scene_pool:
        typer.echo(
            "Error: no compiled .gmdag asset was resolved. "
            "Run `scripts/refresh-scene-corpus.ps1` or pass --gmdag-file explicitly.",
            err=True,
        )
        raise typer.Exit(code=1)

    status = probe_sdfdag_runtime()
    if status.issues:
        for issue in status.issues:
            typer.echo(f"Error: {issue}", err=True)
        raise typer.Exit(code=1)

    try:
        return SdfDagBackend(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _benchmark_actions_tensor(
    *, actor_count: int, device: torch.device | str | int
) -> torch.Tensor:
    """Build tensor-native benchmark actions: (actors, 4) = [fwd, vert, lat, yaw]."""
    import torch

    actions = torch.zeros(actor_count, 4, device=device, dtype=torch.float32)
    actions[:, 0] = 1.5  # forward
    actions[::2, 3] = 0.15  # even actors yaw right
    actions[1::2, 3] = -0.15  # odd actors yaw left
    return actions


def _run_bench_iteration(
    *,
    config: EnvironmentConfig,
    actors: int,
    steps: int,
    warmup_steps: int,
) -> dict[str, float | int]:
    import torch

    backend = _build_backend(config)
    device = torch.device("cuda")
    try:
        for actor_id in range(actors):
            if hasattr(backend, "reset_tensor"):
                backend.reset_tensor(episode_id=0, actor_id=actor_id)
            else:
                backend.reset(episode_id=0, actor_id=actor_id)

        actions = _benchmark_actions_tensor(actor_count=actors, device=device)
        for step_id in range(warmup_steps):
            if hasattr(backend, "batch_step_tensor_actions"):
                backend.batch_step_tensor_actions(actions, step_id)
            else:
                backend.batch_step(tuple(object() for _ in range(actors)), step_id)

        started_at = time.perf_counter()
        for step_id in range(warmup_steps, warmup_steps + steps):
            if hasattr(backend, "batch_step_tensor_actions"):
                backend.batch_step_tensor_actions(actions, step_id)
            else:
                backend.batch_step(tuple(object() for _ in range(actors)), step_id)
        elapsed = time.perf_counter() - started_at
        snapshot = backend.perf_snapshot()
    finally:
        backend.close()

    if snapshot is None:
        typer.echo("Error: selected backend does not expose perf snapshots", err=True)
        raise typer.Exit(code=1)

    expected_total_batches = warmup_steps + steps
    expected_total_actor_steps = expected_total_batches * actors
    if (
        snapshot.total_batches < expected_total_batches
        or snapshot.total_actor_steps < expected_total_actor_steps
    ):
        typer.echo(
            "Error: backend perf snapshot is inconsistent with the requested benchmark profile. "
            f"Expected at least batches={expected_total_batches}, actor_steps={expected_total_actor_steps}; "
            f"got batches={snapshot.total_batches}, actor_steps={snapshot.total_actor_steps}.",
            err=True,
        )
        raise typer.Exit(code=1)

    total_actor_steps = steps * actors
    measured_sps = total_actor_steps / elapsed if elapsed > 0.0 else 0.0
    torch_compile_active = False
    if hasattr(backend, "torch_compile_enabled"):
        torch_compile_active = bool(backend.torch_compile_enabled())
    return {
        "measured_sps": measured_sps,
        "rolling_sps": snapshot.sps,
        "last_batch_ms": snapshot.last_batch_step_ms,
        "ema_batch_ms": snapshot.ema_batch_step_ms,
        "avg_batch_ms": snapshot.avg_batch_step_ms,
        "avg_actor_ms": snapshot.avg_actor_step_ms,
        "total_batches": snapshot.total_batches,
        "total_actor_steps": snapshot.total_actor_steps,
        "expected_total_batches": expected_total_batches,
        "expected_total_actor_steps": expected_total_actor_steps,
        "torch_compile_active": int(torch_compile_active),
    }


def _aggregate_bench_runs(
    runs: list[dict[str, float | int]],
) -> dict[str, float | int | str | list[dict[str, float | int]]]:
    if not runs:
        raise RuntimeError("bench-sdfdag requires at least one run")

    compile_values = {
        int(run["torch_compile_active"]) for run in runs if "torch_compile_active" in run
    }
    compile_summary: dict[str, float | int] = {}
    if len(compile_values) == 1:
        compile_summary["torch_compile_active"] = next(iter(compile_values))

    if len(runs) == 1:
        return {
            **runs[0],
            **compile_summary,
            "aggregation": "single",
            "repeats": 1,
            "per_run": [dict(runs[0], run_index=1)],
        }

    measured_sps_values = [float(run["measured_sps"]) for run in runs]
    rolling_sps_values = [float(run["rolling_sps"]) for run in runs]
    last_batch_ms_values = [float(run["last_batch_ms"]) for run in runs]
    ema_batch_ms_values = [float(run["ema_batch_ms"]) for run in runs]
    avg_batch_ms_values = [float(run["avg_batch_ms"]) for run in runs]
    avg_actor_ms_values = [float(run["avg_actor_ms"]) for run in runs]
    total_batches_values = [int(run["total_batches"]) for run in runs]
    total_actor_steps_values = [int(run["total_actor_steps"]) for run in runs]
    expected_total_batches_values = [int(run["expected_total_batches"]) for run in runs]
    expected_total_actor_steps_values = [int(run["expected_total_actor_steps"]) for run in runs]

    return {
        **compile_summary,
        "aggregation": "median",
        "repeats": len(runs),
        "measured_sps": statistics.median(measured_sps_values),
        "measured_sps_mean": statistics.fmean(measured_sps_values),
        "measured_sps_min": min(measured_sps_values),
        "measured_sps_max": max(measured_sps_values),
        "rolling_sps": statistics.median(rolling_sps_values),
        "last_batch_ms": statistics.median(last_batch_ms_values),
        "ema_batch_ms": statistics.median(ema_batch_ms_values),
        "avg_batch_ms": statistics.median(avg_batch_ms_values),
        "avg_actor_ms": statistics.median(avg_actor_ms_values),
        "total_batches": int(statistics.median(total_batches_values)),
        "total_actor_steps": int(statistics.median(total_actor_steps_values)),
        "expected_total_batches": int(statistics.median(expected_total_batches_values)),
        "expected_total_actor_steps": int(statistics.median(expected_total_actor_steps_values)),
        "per_run": [dict(run, run_index=index + 1) for index, run in enumerate(runs)],
    }


def _require_summary_float(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"Benchmark summary key '{key}' must be numeric, got {type(value).__name__}"
    raise RuntimeError(msg)


def _require_summary_int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    msg = f"Benchmark summary key '{key}' must be numeric, got {type(value).__name__}"
    raise RuntimeError(msg)


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
    sdf_max_steps: int = typer.Option(
        256, help="Maximum sphere-tracing iterations per ray for sdfdag backend"
    ),
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
        f"Environment starting - backend={config.backend}, mode={config.mode}, "
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
    source: str = typer.Option(
        ..., "--source", help="Input source model path (.glb/.obj/.ply/.stl)"
    ),
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
        "Compiled gmdag - "
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
    force_recompile: bool = typer.Option(
        False, help="Overwrite compiled `.gmdag` outputs during refresh"
    ),
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
        typer.echo(f"scene={entry.scene_name} dataset={entry.dataset} gmdag={entry.compiled_path}")
    if len(prepared.scene_entries) > 10:
        typer.echo(f"... and {len(prepared.scene_entries) - 10} more")


@app.command("check-sdfdag")
def check_sdfdag(
    gmdag_file: str = typer.Option(
        "", help=".gmdag asset to validate alongside runtime readiness"
    ),
    gmdag_root: str = typer.Option(
        "", help="Compiled corpus root to validate when checking the promoted corpus"
    ),
    manifest: str = typer.Option(
        "", help="Compiled corpus manifest to validate against live promoted assets"
    ),
    expected_resolution: int = typer.Option(
        512, help="Expected canonical compiled resolution for corpus validation"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a JSON validation summary instead of text"
    ),
) -> None:
    """Validate the canonical SDF/DAG compiler/runtime path and asset metadata."""
    setup_logging("navi_environment_sdfdag_check")
    status = probe_sdfdag_runtime(Path(gmdag_file) if gmdag_file else None)

    corpus_validation = None
    if not gmdag_file:
        corpus_validation = validate_compiled_scene_corpus(
            Path(gmdag_root) if gmdag_root else None,
            manifest_path=Path(manifest) if manifest else None,
            expected_resolution=expected_resolution,
        )

    issues = list(status.issues)
    if corpus_validation is not None:
        issues.extend(corpus_validation.issues)

    if json_output:
        summary = {
            "profile": "check-sdfdag",
            "gmdag_file": str(Path(gmdag_file).expanduser()) if gmdag_file else "",
            "expected_resolution": expected_resolution,
            "runtime": {
                "compiler_ready": status.compiler_ready,
                "torch_ready": status.torch_ready,
                "cuda_ready": status.cuda_ready,
                "torch_sdf_ready": status.torch_sdf_ready,
                "asset_loaded": status.asset_loaded,
                "compiler_path": str(status.compiler_path)
                if status.compiler_path is not None
                else None,
                "gmdag_path": str(status.gmdag_path) if status.gmdag_path is not None else None,
                "resolution": status.resolution,
                "node_count": status.node_count,
                "bbox_min": list(status.bbox_min) if status.bbox_min is not None else None,
                "bbox_max": list(status.bbox_max) if status.bbox_max is not None else None,
            },
            "corpus": None
            if corpus_validation is None
            else {
                "root": str(corpus_validation.gmdag_root),
                "manifest": str(corpus_validation.manifest_path),
                "manifest_present": corpus_validation.manifest_present,
                "scene_count": corpus_validation.scene_count,
                "compiled_resolutions": list(corpus_validation.compiled_resolutions),
            },
            "issues": issues,
            "ok": not issues,
        }
        typer.echo(json.dumps(summary, indent=2))
        if issues:
            raise typer.Exit(code=1)
        return

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
    if corpus_validation is not None:
        typer.echo(
            "corpus - "
            f"root={corpus_validation.gmdag_root}, "
            f"manifest={corpus_validation.manifest_path}, "
            f"manifest_present={corpus_validation.manifest_present}, "
            f"scenes={corpus_validation.scene_count}, "
            f"compiled_resolutions={list(corpus_validation.compiled_resolutions)}",
        )
    if issues:
        for issue in issues:
            typer.echo(f"issue={issue}", err=True)
        raise typer.Exit(code=1)


@app.command("bench-sdfdag")
def bench_sdfdag(
    gmdag_file: str = typer.Option(..., help="Path to compiled .gmdag world cache"),
    actors: int = typer.Option(4, min=1, help="Number of actors to benchmark in one batch"),
    steps: int = typer.Option(200, min=1, help="Measured batch steps after warmup"),
    warmup_steps: int = typer.Option(25, min=0, help="Unmeasured warmup batch steps"),
    repeats: int = typer.Option(
        1, min=1, help="Number of independent benchmark runs to aggregate by median"
    ),
    azimuth_bins: int = typer.Option(256, help="Distance-matrix azimuth bins"),
    elevation_bins: int = typer.Option(48, help="Distance-matrix elevation bins"),
    max_distance: float = typer.Option(30.0, help="Distance normalization range"),
    sdf_max_steps: int = typer.Option(256, help="Maximum sphere-tracing iterations per ray"),
    torch_compile: bool = typer.Option(
        True,
        "--torch-compile/--no-torch-compile",
        help="Compile tensor-only sdfdag helper graphs with torch.compile",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit a JSON benchmark summary instead of text"
    ),
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
        sdfdag_torch_compile=torch_compile,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        max_distance=max_distance,
        n_actors=actors,
    )
    runs = [
        _run_bench_iteration(
            config=config,
            actors=actors,
            steps=steps,
            warmup_steps=warmup_steps,
        )
        for _ in range(repeats)
    ]
    aggregated = _aggregate_bench_runs(runs)
    summary = {
        "profile": "bench-sdfdag",
        "gmdag_file": str(Path(gmdag_file).expanduser()),
        "actors": actors,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "repeats": repeats,
        "azimuth_bins": azimuth_bins,
        "elevation_bins": elevation_bins,
        "max_distance": max_distance,
        "sdf_max_steps": sdf_max_steps,
        "torch_compile": torch_compile,
        **aggregated,
    }
    if json_output:
        typer.echo(json.dumps(summary, indent=2))
        return

    measured_sps = _require_summary_float(summary, "measured_sps")
    rolling_sps = _require_summary_float(summary, "rolling_sps")
    last_batch_ms = _require_summary_float(summary, "last_batch_ms")
    ema_batch_ms = _require_summary_float(summary, "ema_batch_ms")
    avg_batch_ms = _require_summary_float(summary, "avg_batch_ms")
    avg_actor_ms = _require_summary_float(summary, "avg_actor_ms")
    total_batches = _require_summary_int(summary, "total_batches")
    total_actor_steps = _require_summary_int(summary, "total_actor_steps")

    typer.echo(
        "bench-sdfdag - "
        f"actors={actors}, "
        f"steps={steps}, "
        f"warmup_steps={warmup_steps}, "
        f"repeats={repeats}, "
        f"aggregation={summary['aggregation']}, "
        f"measured_sps={measured_sps:.2f}, "
        f"rolling_sps={rolling_sps:.2f}, "
        f"last_batch_ms={last_batch_ms:.2f}, "
        f"ema_batch_ms={ema_batch_ms:.2f}, "
        f"avg_batch_ms={avg_batch_ms:.2f}, "
        f"avg_actor_ms={avg_actor_ms:.2f}, "
        f"total_batches={total_batches}, "
        f"total_actor_steps={total_actor_steps}"
    )
    if repeats > 1:
        measured_sps_min = _require_summary_float(summary, "measured_sps_min")
        measured_sps_max = _require_summary_float(summary, "measured_sps_max")
        measured_sps_mean = _require_summary_float(summary, "measured_sps_mean")
        typer.echo(
            "bench-sdfdag-runs - "
            f"min_sps={measured_sps_min:.2f}, "
            f"max_sps={measured_sps_max:.2f}, "
            f"mean_sps={measured_sps_mean:.2f}"
        )


if __name__ == "__main__":
    app()
