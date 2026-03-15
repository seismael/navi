"""Typer CLI for the Actor service."""

from __future__ import annotations

import random as _random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
import typer

from navi_actor.config import SUPPORTED_TEMPORAL_CORES, ActorConfig, TemporalCoreName
from navi_actor.server import ActorServer
from navi_contracts import setup_logging
from navi_environment.integration.corpus import PreparedSceneCorpus, prepare_training_scene_corpus

if TYPE_CHECKING:
    from navi_actor.training.ppo_trainer import PpoTrainer

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-actor", help="Brain Layer — Sacred Cognitive Engine")


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
        help="Temporal core selector: gru (default) or mambapy.",
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
    total_steps: int = typer.Option(0, help="Total environment steps (0 = continuous until stopped)"),
    min_scene_bytes: int = typer.Option(1000, help="Ignore small scene files"),
    shuffle: bool = typer.Option(True, help="Shuffle scene pool"),
    gmdag_file: str = typer.Option("", help="Single compiled .gmdag cache for canonical sdfdag training"),
    compile_resolution: int = typer.Option(512, help="Compiler voxel resolution for source-scene corpus preparation"),
    force_corpus_refresh: bool = typer.Option(False, help="Force overwrite/recompile of the prepared training corpus"),
    temporal_core: str = typer.Option(
        "",
        help="Temporal core selector: gru (default) or mambapy.",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected azimuth resolution"),
    elevation_bins: int = typer.Option(48, help="Expected elevation resolution"),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension"),
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    learning_rate_final: float = typer.Option(3e-5, help="Final annealed learning rate"),
    rollout_length: int = typer.Option(512, help="Steps per rollout"),
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
    velocity_weight: float = typer.Option(0.1, help="Forward-velocity heuristic weight"),
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
    telemetry_actor_id: int = typer.Option(0, help="Actor ID to emit telemetry for (default: 0)."),
    telemetry_all_actors: bool = typer.Option(False, help="Emit telemetry for all actors (higher overhead)."),
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
    profile_cuda_events: bool = typer.Option(
        False,
        help="Diagnostic-only: force CUDA event timing and synchronization around PPO learner stages.",
    ),
    reward_shaping_torch_compile: bool = typer.Option(
        True,
        "--reward-shaping-torch-compile/--no-reward-shaping-torch-compile",
        help="Compile tensor-only actor reward shaping helper graphs with torch.compile when supported.",
    ),
    actor_pub: str = typer.Option(None, help="Actor PUB bind address"),
) -> None:
    """Single canonical PPO training surface with direct in-process sdfdag stepping."""
    setup_logging("navi_actor_train")

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
        telemetry_actor_id=telemetry_actor_id,
        telemetry_all_actors=telemetry_all_actors,
        emit_observation_stream=emit_observation_stream,
        dashboard_observation_hz=dashboard_observation_hz,
        emit_training_telemetry=emit_training_telemetry,
        emit_update_loss_telemetry=emit_update_loss_telemetry,
        emit_perf_telemetry=emit_perf_telemetry,
        profile_cuda_events=profile_cuda_events,
        reward_shaping_torch_compile=reward_shaping_torch_compile,
    )

    if profile_cuda_events:
        warnings.warn(
            "CUDA event profiling is diagnostic-only and will lower training throughput while timings synchronize.",
            stacklevel=1,
        )

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

    scenes = [str(entry.compiled_path) for entry in corpus.scene_entries]

    if shuffle:
        _random.shuffle(scenes)

    bootstrap_scene = _select_bootstrap_scene(scenes)

    total_steps_label = "continuous" if total_steps <= 0 else f"{total_steps} total steps"
    typer.echo(
        f"Scene pool: {len(scenes)} scenes, {actors} actors, "
        f"{total_steps_label}, shuffle={shuffle}, runtime=sdfdag, temporal={config.temporal_core}"
    )
    typer.echo(f"  Source root: {corpus.source_root}")
    typer.echo(f"  Source manifest: {corpus.source_manifest_path}")
    typer.echo(f"  Compiled manifest: {corpus.compiled_manifest_path}")
    for i, s in enumerate(scenes[:5]):
        typer.echo(f"  [{i+1}] {Path(s).stem}")
    if len(scenes) > 5:
        typer.echo(f"  ... and {len(scenes) - 5} more")

    _validate_bindable(config.pub_address, "Actor PUB")

    trainer = _build_canonical_trainer(
        config,
        gmdag_file=bootstrap_scene,
        scene_pool=tuple(scenes),
    )

    if checkpoint:
        trainer.load_training_state(checkpoint)
        typer.echo(f"Loaded checkpoint: {checkpoint}")

    try:
        trainer.start()
        metrics = trainer.train(
            total_steps=total_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        trainer.stop()

    # Save final checkpoint
    final_ckpt = Path(checkpoint_dir) / "policy_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_training_state(final_ckpt)
    typer.echo(
        f"\nTraining complete | {len(scenes)} scenes | "
        f"steps={metrics.total_steps} episodes={metrics.episodes} "
        f"reward_ema={metrics.reward_ema:.4f} "
        f"policy_loss={metrics.policy_loss:.4f} | "
        f"Final checkpoint: {final_ckpt}"
    )


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
    import torch

    from navi_actor.training.ppo_trainer import PpoTrainer
    from navi_environment.backends.sdfdag_backend import SdfDagBackend
    from navi_environment.config import EnvironmentConfig

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

    trainer.start()
    typer.echo(f"Starting CUDA profile for {steps} steps...")

    # Warmup
    trainer.train(actors * 2)

    torch.cuda.profiler.start()
    trainer.train(steps)
    torch.cuda.profiler.stop()

    trainer.stop()
    typer.echo("Profiling complete.")


@app.command("brain")  # type: ignore[untyped-decorator]
def brain(
    ctx: click.Context,
    mode: str = typer.Argument(..., help="Operation mode: train, serve, audit, or profile"),
    scene: str = typer.Option("", help="[train/audit/profile] Explicit scene name or path override"),
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
        serve_args = ["serve", "--azimuth-bins", str(azimuth_bins), "--elevation-bins", str(elevation_bins)]
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


def brain_main() -> None:
    """Entry point for the 'brain' console script."""
    app()


if __name__ == "__main__":
    app()
