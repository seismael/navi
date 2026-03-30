"""Typer CLI for the Actor service."""

from __future__ import annotations

import logging
import random as _random
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
import typer

from navi_actor.config import SUPPORTED_TEMPORAL_CORES, ActorConfig, TemporalCoreName
from navi_actor.server import ActorServer
from navi_contracts import (
    JsonlMetricsSink,
    build_phase_metrics_payload,
    get_or_create_run_context,
    setup_logging,
    write_process_manifest,
)
from navi_environment.integration.corpus import PreparedSceneCorpus, prepare_training_scene_corpus

if TYPE_CHECKING:
    from navi_actor.training.ppo_trainer import PpoTrainer

__all__: list[str] = ["app"]

_LOG = logging.getLogger(__name__)

app = typer.Typer(name="navi-actor", help="Brain Layer — Sacred Cognitive Engine")


def _emit_command_phase_metric(
    sink: JsonlMetricsSink | None,
    *,
    operation: str,
    started_at: float,
    include_resources: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    if sink is None:
        return
    sink.emit(
        "command_phase",
        build_phase_metrics_payload(
            operation,
            started_at=started_at,
            cuda_device="cuda" if metadata and metadata.get("cuda_expected") else None,
            include_resources=include_resources,
            metadata=metadata,
        ),
    )


def _configure_torch_training_runtime() -> None:
    """Enable stable backend autotuning for fixed-shape canonical training kernels."""
    import torch

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True


def _validate_sdfdag_training_scenes(scenes: list[str]) -> None:
    """Ensure canonical training scenes are compiled `.gmdag` assets."""
    non_gmdag = [scene for scene in scenes if Path(scene).suffix.lower() != ".gmdag"]
    if non_gmdag:
        typer.echo(
            "Canonical train requires compiled .gmdag assets only. "
            "Compile meshes first with `navi-environment compile-gmdag`.",
            err=True,
        )
        raise typer.Exit(1)


def _validate_bindable(address: str, label: str) -> None:
    """Fail fast when a required local ZMQ bind address is already occupied."""
    import zmq

    zmq_mod = cast("Any", zmq)
    ctx = zmq_mod.Context()
    sock = ctx.socket(zmq_mod.PUB)
    try:
        sock.setsockopt(zmq_mod.LINGER, 0)
        sock.bind(address)
    except zmq_mod.ZMQError as err:
        typer.echo(
            f"ERROR: {label} address is already in use: {address}\n"
            "Stop stale navi processes and retry.",
            err=True,
        )
        raise typer.Exit(1) from err
    finally:
        sock.close()
        ctx.term()


def _load_training_scenes(
    *,
    scene: str,
    manifest: str,
    gmdag_file: str,
    corpus_root: str,
    gmdag_root: str,
    compile_resolution: int,
    min_scene_bytes: int,
    force_corpus_refresh: bool,
) -> PreparedSceneCorpus:
    """Resolve and validate the canonical scene pool for sdfdag training."""
    try:
        corpus = prepare_training_scene_corpus(
            scene=scene,
            manifest_path=Path(manifest) if manifest else None,
            gmdag_file=Path(gmdag_file) if gmdag_file else None,
            source_root=Path(corpus_root) if corpus_root else None,
            gmdag_root=Path(gmdag_root) if gmdag_root else None,
            resolution=compile_resolution,
            min_scene_bytes=min_scene_bytes,
            force_recompile=force_corpus_refresh,
        )
    except RuntimeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc

    _validate_sdfdag_training_scenes([str(entry.compiled_path) for entry in corpus.scene_entries])
    return corpus


def _build_canonical_trainer(
    config: ActorConfig,
    *,
    gmdag_file: str,
    scene_pool: tuple[str, ...],
) -> PpoTrainer:
    """Create the single canonical direct-backend trainer.

    Direct environment integration stays at the CLI boundary so the actor
    package keeps one trainer implementation without reintroducing service-mode
    architecture branches.
    """
    from navi_actor.training.ppo_trainer import PpoTrainer
    from navi_environment.backends.sdfdag_backend import SdfDagBackend
    from navi_environment.config import EnvironmentConfig

    env_config = EnvironmentConfig(
        mode="step",
        training_mode=True,
        n_actors=config.n_actors,
        backend="sdfdag",
        gmdag_file=gmdag_file,
        scene_pool=scene_pool,
        azimuth_bins=config.azimuth_bins,
        elevation_bins=config.elevation_bins,
        max_distance=30.0,
        compute_overhead=False,
    )
    runtime = SdfDagBackend(env_config)

    return PpoTrainer(
        config,
        runtime=runtime,
        gmdag_file=gmdag_file,
        scene_pool=scene_pool,
    )


def _select_bootstrap_scene(scenes: list[str]) -> str:
    """Choose a deterministic first scene for trainer startup.

    Canonical training still uses the full scene pool, but startup should not
    depend on a random first asset when one compiled scene can initialize more
    predictably than another on the active machine.
    """

    if not scenes:
        raise RuntimeError("Canonical training requires at least one compiled scene")

    def _scene_rank(scene_path: str) -> tuple[int, str]:
        path = Path(scene_path)
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 2**63 - 1
        return (size, path.name.lower())

    return min(scenes, key=_scene_rank)


def _resolve_temporal_core(temporal_core: str, default_config: ActorConfig) -> TemporalCoreName:
    """Validate a requested temporal core against the supported selector set."""
    resolved = temporal_core or default_config.temporal_core
    if resolved not in SUPPORTED_TEMPORAL_CORES:
        supported = ", ".join(SUPPORTED_TEMPORAL_CORES)
        typer.echo(
            f"Unsupported temporal core '{resolved}'. Expected one of: {supported}.",
            err=True,
        )
        raise typer.Exit(1)
    return cast("TemporalCoreName", resolved)


@app.command()  # type: ignore[untyped-decorator]
def serve(
    # Typer command decorators are intentionally untyped.
    sub: str = typer.Option(None, help="ZMQ SUB address (DistanceMatrix v2)"),
    pub: str = typer.Option(None, help="ZMQ PUB address (Action v2)"),
    mode: str = typer.Option("async", help="Mode: async (inference) or step (training)"),
    step_endpoint: str = typer.Option(
        None,
        help="ZMQ REP bind address for StepResult (step mode)",
    ),
    policy_checkpoint: str = typer.Option(
        "",
        help="Checkpoint path (.pt) to load pre-trained weights",
    ),
    temporal_core: str = typer.Option(
        "",
        help="Temporal core selector: mamba2 (default), gru, or mambapy.",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(48, help="Expected distance-matrix elevation resolution"),
) -> None:
    """Start the Actor service (always uses CognitiveMambaPolicy)."""
    setup_logging("navi_actor")

    default_config = ActorConfig()
    resolved_temporal_core = _resolve_temporal_core(temporal_core, default_config)

    config = ActorConfig(
        sub_address=sub or default_config.sub_address,
        pub_address=pub or default_config.pub_address,
        mode=mode,
        step_endpoint=step_endpoint or default_config.step_endpoint,
        temporal_core=resolved_temporal_core,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
    )

    from navi_actor.cognitive_policy import CognitiveMambaPolicy

    if policy_checkpoint:
        runtime_policy = CognitiveMambaPolicy.load_checkpoint(
            policy_checkpoint,
            embedding_dim=config.embedding_dim,
            temporal_core=config.temporal_core,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        )
    else:
        runtime_policy = CognitiveMambaPolicy(
            embedding_dim=config.embedding_dim,
            temporal_core=config.temporal_core,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            max_forward=config.max_forward,
            max_vertical=config.max_vertical,
            max_lateral=config.max_lateral,
            max_yaw=config.max_yaw,
        )

    server = ActorServer(config=config, policy=runtime_policy)
    typer.echo(
        f"Actor starting — mode={config.mode}, sub={config.sub_address}, "
        f"pub={config.pub_address}, policy=cognitive, temporal={config.temporal_core}"
    )
    server.run()


@app.command("train")  # type: ignore[untyped-decorator]
def train(
    scene: str = typer.Option("", help="Explicit scene name or path override"),
    manifest: str = typer.Option("", help="Path to scene manifest JSON"),
    corpus_root: str = typer.Option("", help="Root directory for canonical scene discovery"),
    gmdag_root: str = typer.Option("", help="Root directory for compiled corpus outputs"),
    actors: int = typer.Option(4, help="Number of parallel environments"),
    total_steps: int = typer.Option(
        0, help="Total environment steps (0 = continuous until stopped)"
    ),
    min_scene_bytes: int = typer.Option(1000, help="Ignore small scene files"),
    shuffle: bool = typer.Option(True, help="Shuffle scene pool"),
    gmdag_file: str = typer.Option(
        "", help="Single compiled .gmdag cache for canonical sdfdag training"
    ),
    compile_resolution: int = typer.Option(
        512, help="Compiler voxel resolution for source-scene corpus preparation"
    ),
    force_corpus_refresh: bool = typer.Option(
        False, help="Force overwrite/recompile of the prepared training corpus"
    ),
    temporal_core: str = typer.Option(
        "",
        help="Temporal core selector: mamba2 (default), gru, or mambapy.",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected azimuth resolution"),
    elevation_bins: int = typer.Option(48, help="Expected elevation resolution"),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension"),
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    learning_rate_final: float = typer.Option(3e-5, help="Final annealed learning rate"),
    rollout_length: int = typer.Option(256, help="Steps per rollout"),
    ppo_epochs: int = typer.Option(2, help="PPO epochs per update"),
    minibatch_size: int = typer.Option(64, help="Minibatch size"),
    bptt_len: int = typer.Option(8, help="BPTT sequence length"),
    clip_ratio: float = typer.Option(0.2, help="PPO clip ratio"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    entropy_coeff: float = typer.Option(0.01, help="Entropy bonus coefficient"),
    value_coeff: float = typer.Option(0.5, help="Value loss coefficient"),
    # Reward shaping
    collision_penalty: float = typer.Option(0.0, help="Collision termination penalty"),
    existential_tax: float = typer.Option(-0.02, help="Per-step existence cost"),
    velocity_weight: float = typer.Option(0.0, help="Forward-velocity heuristic weight"),
    # RND curiosity
    intrinsic_coeff_init: float = typer.Option(1.0, help="Initial RND intrinsic coefficient"),
    intrinsic_coeff_final: float = typer.Option(0.01, help="Final RND intrinsic coefficient"),
    intrinsic_anneal_steps: int = typer.Option(500_000, help="Steps to anneal intrinsic coeff"),
    rnd_learning_rate: float = typer.Option(3e-5, help="RND predictor learning rate"),
    rnd_learning_rate_final: float = typer.Option(3e-6, help="Final RND learning rate"),
    # Episodic memory
    memory_capacity: int = typer.Option(100_000, help="Episodic memory max entries"),
    memory_exclusion_window: int = typer.Option(50, help="Exclude recent N steps from query"),
    loop_threshold: float = typer.Option(0.85, help="Cosine similarity loop threshold"),
    loop_penalty_coeff: float = typer.Option(2.0, help="Loop penalty coefficient"),
    enable_episodic_memory: bool = typer.Option(
        True,
        help="Keep episodic-memory query/add active (disable for attribution only).",
    ),
    enable_reward_shaping: bool = typer.Option(
        True,
        help="Keep canonical reward shaping active (disable for attribution only).",
    ),
    # Checkpointing
    checkpoint_every: int = typer.Option(25_000, help="Checkpoint interval (0 = disabled)"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
    checkpoint: str = typer.Option("", help="Resume from checkpoint path (.pt)"),
    # Logging
    log_every: int = typer.Option(100, help="Steps between log messages"),
    # Telemetry fan-out (performance)
    telemetry_actor_id: int = typer.Option(0, help="[Deprecated, ignored] Actor ID 0 is always used."),
    telemetry_all_actors: bool = typer.Option(
        False, help="[Deprecated, ignored] Only actor 0 telemetry is emitted."
    ),
    emit_observation_stream: bool = typer.Option(
        True,
        help="Emit low-volume DistanceMatrix frames for the selected telemetry actor; use --telemetry-all-actors for diagnostic fan-out.",
    ),
    dashboard_observation_hz: float = typer.Option(
        10.0,
        min=1.0,
        help="Target passive dashboard observation cadence in Hz for published observation actors.",
    ),
    emit_training_telemetry: bool = typer.Option(
        True,
        help="Emit actor.training step, update, and episode telemetry.",
    ),
    emit_update_loss_telemetry: bool = typer.Option(
        False,
        help="Diagnostic-only: materialize PPO loss scalars for actor.training.ppo.update payloads.",
    ),
    emit_perf_telemetry: bool = typer.Option(
        True,
        help="Emit actor/environment performance telemetry events.",
    ),
    emit_internal_stats: bool | None = typer.Option(
        None,
        "--emit-internal-stats/--no-emit-internal-stats",
        help="Write internal run-scoped machine-readable performance metrics under the run root.",
    ),
    attach_resource_snapshots: bool | None = typer.Option(
        None,
        "--attach-resource-snapshots/--no-attach-resource-snapshots",
        help="Attach coarse process and CUDA memory snapshots to internal metrics records.",
    ),
    print_performance_summary: bool | None = typer.Option(
        None,
        "--print-performance-summary/--no-print-performance-summary",
        help="Print a concise internal performance summary and metrics paths from the train command.",
    ),
    profile_cuda_events: bool = typer.Option(
        False,
        help="Diagnostic-only: force CUDA event timing and synchronization around PPO learner stages.",
    ),
    reward_shaping_torch_compile: bool = typer.Option(
        True,
        "--reward-shaping-torch-compile/--no-reward-shaping-torch-compile",
        help="Compile tensor-only actor reward shaping helper graphs with torch.compile when supported.",
    ),
    datasets: str = typer.Option(
        "",
        help="Comma-separated dataset name filter (e.g. 'ai-habitat_ReplicaCAD_baked_lighting'). Empty = all datasets.",
    ),
    exclude_datasets: str = typer.Option(
        "",
        help="Comma-separated dataset names to exclude from training corpus.",
    ),
    actor_pub: str = typer.Option(None, help="Actor PUB bind address"),
) -> None:
    """Single canonical PPO training surface with direct in-process sdfdag stepping."""
    setup_logging("navi_actor_train")
    run_context = get_or_create_run_context("actor-train")
    command_metrics_sink: JsonlMetricsSink | None = None

    trainer: PpoTrainer | None = None
    trainer_started = False
    try:
        default_config = ActorConfig()
        resolved_temporal_core = _resolve_temporal_core(temporal_core, default_config)

        # Resolve actor config
        config = ActorConfig(
            pub_address=actor_pub or default_config.pub_address,
            mode="step",
            temporal_core=resolved_temporal_core,
            azimuth_bins=azimuth_bins,
            elevation_bins=elevation_bins,
            n_actors=actors,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            learning_rate_final=learning_rate_final,
            rollout_length=rollout_length,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            bptt_len=bptt_len,
            clip_ratio=clip_ratio,
            gamma=gamma,
            gae_lambda=gae_lambda,
            entropy_coeff=entropy_coeff,
            value_coeff=value_coeff,
            collision_penalty=collision_penalty,
            existential_tax=existential_tax,
            velocity_weight=velocity_weight,
            intrinsic_coeff_init=intrinsic_coeff_init,
            intrinsic_coeff_final=intrinsic_coeff_final,
            intrinsic_anneal_steps=intrinsic_anneal_steps,
            rnd_learning_rate=rnd_learning_rate,
            rnd_learning_rate_final=rnd_learning_rate_final,
            memory_capacity=memory_capacity,
            memory_exclusion_window=memory_exclusion_window,
            loop_threshold=loop_threshold,
            loop_penalty_coeff=loop_penalty_coeff,
            enable_episodic_memory=enable_episodic_memory,
            enable_reward_shaping=enable_reward_shaping,
            emit_observation_stream=emit_observation_stream,
            dashboard_observation_hz=dashboard_observation_hz,
            emit_training_telemetry=emit_training_telemetry,
            emit_update_loss_telemetry=emit_update_loss_telemetry,
            emit_perf_telemetry=emit_perf_telemetry,
            emit_internal_stats=(
                default_config.emit_internal_stats
                if emit_internal_stats is None
                else emit_internal_stats
            ),
            attach_resource_snapshots=(
                default_config.attach_resource_snapshots
                if attach_resource_snapshots is None
                else attach_resource_snapshots
            ),
            print_performance_summary=(
                default_config.print_performance_summary
                if print_performance_summary is None
                else print_performance_summary
            ),
            profile_cuda_events=profile_cuda_events,
            reward_shaping_torch_compile=reward_shaping_torch_compile,
        )

        if config.emit_internal_stats:
            command_metrics_sink = JsonlMetricsSink(
                run_context.metrics_root / "actor_training.command.jsonl",
                run_id=run_context.run_id,
                project_name="navi_actor_train",
            )

        if profile_cuda_events:
            warnings.warn(
                "CUDA event profiling is diagnostic-only and will lower training throughput while timings synchronize.",
                stacklevel=1,
            )

        t_corpus_prepare = time.perf_counter()
        corpus = _load_training_scenes(
            scene=scene,
            manifest=manifest,
            gmdag_file=gmdag_file,
            corpus_root=corpus_root,
            gmdag_root=gmdag_root,
            compile_resolution=compile_resolution,
            min_scene_bytes=min_scene_bytes,
            force_corpus_refresh=force_corpus_refresh,
        )
        entries = list(corpus.scene_entries)
        if datasets:
            allowed = {d.strip() for d in datasets.split(",") if d.strip()}
            entries = [e for e in entries if e.dataset in allowed]
            if not entries:
                typer.echo(f"No scenes matched --datasets filter: {datasets}", err=True)
                raise typer.Exit(1)
        if exclude_datasets:
            excluded = {d.strip() for d in exclude_datasets.split(",") if d.strip()}
            entries = [e for e in entries if e.dataset not in excluded]
            if not entries:
                typer.echo(f"All scenes excluded by --exclude-datasets: {exclude_datasets}", err=True)
                raise typer.Exit(1)
        scenes = [str(entry.compiled_path) for entry in entries]
        _emit_command_phase_metric(
            command_metrics_sink,
            operation="corpus_prepare",
            started_at=t_corpus_prepare,
            include_resources=config.attach_resource_snapshots,
            metadata={
                "cuda_expected": False,
                "scene_count": len(scenes),
                "source_root": str(corpus.source_root),
                "compiled_manifest_path": str(corpus.compiled_manifest_path),
                "force_corpus_refresh": bool(force_corpus_refresh),
            },
        )

        if shuffle:
            _random.shuffle(scenes)

        bootstrap_scene = _select_bootstrap_scene(scenes)
        resolved_checkpoint_dir = checkpoint_dir
        if checkpoint_dir == "checkpoints":
            resolved_checkpoint_dir = str(run_context.run_root / "checkpoints")

        total_steps_label = "continuous" if total_steps <= 0 else f"{total_steps} total steps"
        typer.echo(
            f"Scene pool: {len(scenes)} scenes, {actors} actors, "
            f"{total_steps_label}, shuffle={shuffle}, runtime=sdfdag, temporal={config.temporal_core}"
        )
        typer.echo(f"  Source root: {corpus.source_root}")
        typer.echo(f"  Source manifest: {corpus.source_manifest_path}")
        typer.echo(f"  Compiled manifest: {corpus.compiled_manifest_path}")
        for i, s in enumerate(scenes[:5]):
            typer.echo(f"  [{i + 1}] {Path(s).stem}")
        if len(scenes) > 5:
            typer.echo(f"  ... and {len(scenes) - 5} more")
        typer.echo(f"  Run ID: {run_context.run_id}")
        typer.echo(f"  Run Root: {run_context.run_root}")
        typer.echo(
            "  Internal stats: "
            f"metrics={'on' if config.emit_internal_stats else 'off'} "
            f"snapshots={'on' if config.attach_resource_snapshots else 'off'} "
            f"perf-telemetry={'on' if config.emit_perf_telemetry else 'off'}"
        )
        if config.emit_internal_stats:
            typer.echo(f"  Metrics root: {run_context.metrics_root}")

        write_process_manifest(
            "navi_actor_train",
            context=run_context,
            metadata={
                "command": "train",
                "actors": actors,
                "temporal_core": config.temporal_core,
                "azimuth_bins": azimuth_bins,
                "elevation_bins": elevation_bins,
                "scene_count": len(scenes),
                "bootstrap_scene": bootstrap_scene,
                "checkpoint_dir": resolved_checkpoint_dir,
                "checkpoint_every": checkpoint_every,
                "log_every": log_every,
                "emit_internal_stats": config.emit_internal_stats,
                "attach_resource_snapshots": config.attach_resource_snapshots,
                "print_performance_summary": config.print_performance_summary,
            },
            file_name="navi_actor_train.command.json",
        )

        _validate_bindable(config.pub_address, "Actor PUB")
        _configure_torch_training_runtime()

        t_trainer_build = time.perf_counter()
        trainer = _build_canonical_trainer(
            config,
            gmdag_file=bootstrap_scene,
            scene_pool=tuple(scenes),
        )
        _emit_command_phase_metric(
            command_metrics_sink,
            operation="trainer_build",
            started_at=t_trainer_build,
            include_resources=config.attach_resource_snapshots,
            metadata={
                "cuda_expected": True,
                "actors": actors,
                "temporal_core": config.temporal_core,
                "bootstrap_scene": bootstrap_scene,
            },
        )

        if checkpoint:
            t_checkpoint_load = time.perf_counter()
            trainer.load_training_state(checkpoint)
            _emit_command_phase_metric(
                command_metrics_sink,
                operation="checkpoint_load",
                started_at=t_checkpoint_load,
                include_resources=config.attach_resource_snapshots,
                metadata={
                    "cuda_expected": True,
                    "checkpoint_path": checkpoint,
                },
            )
            typer.echo(f"Loaded checkpoint: {checkpoint}")

        t_trainer_start = time.perf_counter()
        trainer.start()  # type: ignore[no-untyped-call]
        trainer_started = True
        _emit_command_phase_metric(
            command_metrics_sink,
            operation="trainer_start",
            started_at=t_trainer_start,
            include_resources=config.attach_resource_snapshots,
            metadata={
                "cuda_expected": True,
                "actors": actors,
            },
        )

        t_train_run = time.perf_counter()
        metrics = trainer.train(
            total_steps=total_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=resolved_checkpoint_dir,
        )
        _emit_command_phase_metric(
            command_metrics_sink,
            operation="train_run",
            started_at=t_train_run,
            include_resources=config.attach_resource_snapshots,
            metadata={
                "cuda_expected": True,
                "total_steps": metrics.total_steps,
                "episodes": metrics.episodes,
                "reward_ema": metrics.reward_ema,
                "sps_mean": metrics.sps_mean,
                "ppo_update_ms_mean": metrics.ppo_update_ms_mean,
            },
        )

        final_ckpt = Path(resolved_checkpoint_dir) / "policy_final.pt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        t_final_checkpoint = time.perf_counter()
        trainer.save_training_state(final_ckpt)
        _emit_command_phase_metric(
            command_metrics_sink,
            operation="final_checkpoint_save",
            started_at=t_final_checkpoint,
            include_resources=config.attach_resource_snapshots,
            metadata={
                "cuda_expected": True,
                "checkpoint_path": str(final_ckpt),
                "step_id": metrics.total_steps,
            },
        )
        typer.echo(
            f"\nTraining complete | {len(scenes)} scenes | "
            f"steps={metrics.total_steps} episodes={metrics.episodes} "
            f"reward_ema={metrics.reward_ema:.4f} "
            f"policy_loss={metrics.policy_loss:.4f} | "
            f"Final checkpoint: {final_ckpt}"
        )
        if config.print_performance_summary:
            typer.echo(
                "Performance summary | "
                f"sps_mean={metrics.sps_mean:.2f} "
                f"ppo_update_ms_mean={metrics.ppo_update_ms_mean:.2f} "
                f"internal_stats={'on' if config.emit_internal_stats else 'off'} "
                f"resource_snapshots={'on' if config.attach_resource_snapshots else 'off'}"
            )
            if config.emit_internal_stats:
                typer.echo(
                    "Internal metrics | "
                    f"command={run_context.metrics_root / 'actor_training.command.jsonl'} "
                    f"trainer={run_context.metrics_root / 'actor_training.jsonl'}"
                )
    finally:
        if trainer is not None and trainer_started:
            t_trainer_stop = time.perf_counter()
            trainer.stop()  # type: ignore[no-untyped-call]
            _emit_command_phase_metric(
                command_metrics_sink,
                operation="trainer_stop",
                started_at=t_trainer_stop,
                include_resources=(
                    config.attach_resource_snapshots if "config" in locals() else True
                ),
                metadata={"cuda_expected": True},
            )
        if command_metrics_sink is not None:
            command_metrics_sink.close()


@app.command("profile")  # type: ignore[untyped-decorator]
def profile(
    ctx: click.Context,
    scene: str = typer.Option("", help="Source .gmdag file"),
    steps: int = typer.Option(512, help="Steps to profile"),
    actors: int = typer.Option(4, help="Parallel actors"),
    azimuth_bins: int = typer.Option(256),
    elevation_bins: int = typer.Option(48),
) -> None:
    """Run a fixed-length rollout with CUDA profiling active (Phase 14)."""
    del ctx
    setup_logging("navi_actor_profile")
    import torch

    from navi_actor.training.ppo_trainer import PpoTrainer
    from navi_environment.backends.sdfdag_backend import SdfDagBackend
    from navi_environment.config import EnvironmentConfig

    _configure_torch_training_runtime()

    config = ActorConfig(
        n_actors=actors,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
    )
    env_cfg = EnvironmentConfig(
        gmdag_file=scene,
        n_actors=actors,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
    )
    runtime = SdfDagBackend(env_cfg)
    trainer = PpoTrainer(config, runtime=runtime, gmdag_file=scene)

    trainer.start()  # type: ignore[no-untyped-call]
    typer.echo(f"Starting CUDA profile for {steps} steps...")

    # Warmup
    trainer.train(actors * 2)

    torch.cuda.profiler.start()  # type: ignore[no-untyped-call]
    trainer.train(steps)
    torch.cuda.profiler.stop()  # type: ignore[no-untyped-call]

    trainer.stop()  # type: ignore[no-untyped-call]
    typer.echo("Profiling complete.")


@app.command("brain")  # type: ignore[untyped-decorator]
def brain(
    ctx: click.Context,
    mode: str = typer.Argument(..., help="Operation mode: train, serve, audit, or profile"),
    scene: str = typer.Option(
        "", help="[train/audit/profile] Explicit scene name or path override"
    ),
    checkpoint: str = typer.Option("", help="[train/serve] Checkpoint path (.pt) to load"),
    actors: int = typer.Option(4, help="[train/audit/profile] Number of parallel actors"),
    azimuth_bins: int = typer.Option(256, help="Azimuth resolution"),
    elevation_bins: int = typer.Option(48, help="Elevation resolution"),
) -> None:
    """Unified Brain service entry point (Phase 10/14)."""
    del ctx
    if mode == "train":
        train_args = [
            "train",
            "--azimuth-bins",
            str(azimuth_bins),
            "--elevation-bins",
            str(elevation_bins),
            "--actors",
            str(actors),
        ]
        if scene:
            train_args.extend(["--scene", scene])
        if checkpoint:
            train_args.extend(["--checkpoint", checkpoint])
        app(train_args)
    elif mode == "serve":
        serve_args = [
            "serve",
            "--azimuth-bins",
            str(azimuth_bins),
            "--elevation-bins",
            str(elevation_bins),
        ]
        if checkpoint:
            serve_args.extend(["--policy-checkpoint", checkpoint])
        app(serve_args)
    elif mode == "audit":
        import subprocess

        audit_cmd = [
            "uv",
            "run",
            "navi-auditor",
            "dataset-audit",
            "--actors",
            str(actors),
            "--azimuth-bins",
            str(azimuth_bins),
            "--elevation-bins",
            str(elevation_bins),
        ]
        if scene:
            audit_cmd.extend(["--gmdag-file", scene])
        typer.echo(f"Delegating to auditor: {' '.join(audit_cmd)}")
        subprocess.run(audit_cmd, check=True)  # noqa: S603
    elif mode == "profile":
        profile_args = [
            "profile",
            "--azimuth-bins",
            str(azimuth_bins),
            "--elevation-bins",
            str(elevation_bins),
            "--actors",
            str(actors),
        ]
        if scene:
            profile_args.extend(["--scene", scene])
        app(profile_args)
    else:
        typer.echo(f"Unknown brain mode: {mode}", err=True)
        raise typer.Exit(1)


@app.command("bc-pretrain")  # type: ignore[untyped-decorator]
def bc_pretrain(
    demonstrations: str = typer.Option(
        "artifacts/demonstrations",
        help="Directory containing .npz demonstration files.",
    ),
    output: str = typer.Option(
        "artifacts/checkpoints/bc_base_model.pt",
        help="Output path for the BC checkpoint.",
    ),
    checkpoint: str = typer.Option(
        "",
        help="Resume training from an existing checkpoint (.pt) to incrementally improve the model.",
    ),
    temporal_core: str = typer.Option(
        "",
        help="Temporal core selector: mamba2 (default), gru, or mambapy.",
    ),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension."),
    azimuth_bins: int = typer.Option(256, help="Observation azimuth resolution."),
    elevation_bins: int = typer.Option(48, help="Observation elevation resolution."),
    epochs: int = typer.Option(50, help="Number of training epochs."),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate."),
    bptt_len: int = typer.Option(8, help="BPTT sequence length."),
    minibatch_size: int = typer.Option(32, help="Sequences per minibatch."),
    entropy_coeff: float = typer.Option(0.01, help="Entropy regularization coefficient."),
    max_grad_norm: float = typer.Option(0.5, help="Gradient clipping threshold."),
    freeze_log_std: bool = typer.Option(
        True, help="Freeze policy log-std to preserve exploration capacity for PPO."
    ),
) -> None:
    """Behavioral cloning pre-training from recorded human demonstrations.

    Trains the full CognitiveMambaPolicy pipeline (RayViTEncoder → TemporalCore
    → ActorCriticHeads) via supervised maximum-likelihood on human navigation
    demonstrations.  Produces a v2 checkpoint that can be loaded directly by
    ``navi-actor train --checkpoint <path>`` for RL fine-tuning.

    Workflow:
      1. Record demonstrations: ``uv run explore --record``
      2. Pre-train:            ``uv run brain bc-pretrain``
      3. Fine-tune with RL:    ``uv run brain train --checkpoint artifacts/checkpoints/bc_base_model.pt``

    Incremental workflow (across scenes):
      1. Fly scene 1:  ``uv run explore --record --gmdag-file scene1.gmdag``
      2. Train:        ``uv run brain bc-pretrain``
      3. Fly scene 2:  ``uv run explore --record --gmdag-file scene2.gmdag``
      4. Update model: ``uv run brain bc-pretrain --checkpoint artifacts/checkpoints/bc_base_model.pt``
    """
    setup_logging("navi_actor_bc_pretrain")
    run_context = get_or_create_run_context("actor-bc-pretrain")

    from navi_actor.config import ActorConfig

    config = ActorConfig()
    resolved_temporal: TemporalCoreName = cast(
        TemporalCoreName,
        temporal_core if temporal_core in SUPPORTED_TEMPORAL_CORES else config.temporal_core,
    )

    _LOG.info(
        "BC pre-training — demos=%s temporal=%s epochs=%d lr=%.1e checkpoint=%s",
        demonstrations,
        resolved_temporal,
        epochs,
        learning_rate,
        checkpoint or "(none)",
    )

    from navi_actor.training.bc_trainer import BehavioralCloningTrainer

    resolved_checkpoint = Path(checkpoint) if checkpoint else None
    if resolved_checkpoint is not None and not resolved_checkpoint.exists():
        typer.echo(f"Checkpoint not found: {resolved_checkpoint}", err=True)
        raise typer.Exit(code=1)

    trainer = BehavioralCloningTrainer(
        demo_dir=Path(demonstrations),
        output_path=Path(output),
        checkpoint_path=resolved_checkpoint,
        temporal_core=resolved_temporal,
        embedding_dim=embedding_dim,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        epochs=epochs,
        learning_rate=learning_rate,
        bptt_len=bptt_len,
        minibatch_size=minibatch_size,
        entropy_coeff=entropy_coeff,
        max_grad_norm=max_grad_norm,
        freeze_log_std=freeze_log_std,
    )
    checkpoint_path = trainer.run()
    typer.echo(f"BC checkpoint saved to {checkpoint_path}")
    typer.echo("")
    typer.echo("Next step — fine-tune with RL:")
    typer.echo(f"  uv run brain train --checkpoint {checkpoint_path}")


def brain_main() -> None:
    """Entry point for the 'brain' console script."""
    app()


if __name__ == "__main__":
    app()
