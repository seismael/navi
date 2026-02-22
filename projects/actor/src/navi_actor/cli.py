"""Typer CLI for the Actor service."""

from __future__ import annotations

import logging

import typer

from navi_actor.config import ActorConfig
from navi_actor.policy import LearnedSphericalPolicy
from navi_actor.server import ActorServer
from navi_actor.training.online import OnlineSphericalTrainer

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-actor", help="Layer 3: The Brain — Actor service")


@app.command()
def run(
    sub: str = typer.Option("tcp://localhost:5559", help="ZMQ SUB address (Simulation Layer)"),
    pub: str = typer.Option("tcp://*:5557", help="ZMQ PUB bind address"),
    mode: str = typer.Option("async", help="Mode: async (PUB/SUB) or step (REQ/REP training)"),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560",
        help="Section Manager REP address (step mode)",
    ),
    policy: str = typer.Option("shallow", help="Policy type: shallow, learned, or cognitive"),
    policy_checkpoint: str = typer.Option(
        "",
        help="Checkpoint path (.npz for learned, .pt for cognitive)",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(128, help="Expected distance-matrix elevation resolution"),
) -> None:
    """Start the Actor service."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        root_logger.setLevel(logging.INFO)

    config = ActorConfig(
        sub_address=sub,
        pub_address=pub,
        mode=mode,
        step_endpoint=step_endpoint,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
    )

    runtime_policy = None
    if policy == "learned":
        if not policy_checkpoint:
            typer.echo("Error: --policy-checkpoint is required when --policy=learned", err=True)
            raise typer.Exit(code=1)
        checkpoint = LearnedSphericalPolicy.load_checkpoint(policy_checkpoint)
        runtime_policy = LearnedSphericalPolicy(checkpoint=checkpoint)
    elif policy == "cognitive":
        from navi_actor.cognitive_policy import CognitiveMambaPolicy

        if policy_checkpoint:
            runtime_policy = CognitiveMambaPolicy.load_checkpoint(
                policy_checkpoint,
                embedding_dim=config.embedding_dim,
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
                azimuth_bins=azimuth_bins,
                elevation_bins=elevation_bins,
                max_forward=config.max_forward,
                max_vertical=config.max_vertical,
                max_lateral=config.max_lateral,
                max_yaw=config.max_yaw,
            )

    server = ActorServer(config=config, policy=runtime_policy)
    typer.echo(
        f"Actor starting — mode={config.mode}, sub={config.sub_address}, pub={config.pub_address}, policy={policy}"
    )
    server.run()


@app.command()
def train(
    sub: str = typer.Option("tcp://localhost:5559", help="ZMQ SUB address (DistanceMatrix stream)"),
    pub: str = typer.Option(
        "",
        help="ZMQ PUB bind address for training telemetry (empty disables)",
    ),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560",
        help="Section Manager REP address (step mode)",
    ),
    steps: int = typer.Option(3000, help="Number of online training steps"),
    log_every: int = typer.Option(100, help="Log interval in steps"),
    learning_rate: float = typer.Option(3e-3, help="Policy update learning rate"),
    sigma_forward: float = typer.Option(0.12, help="Forward action exploration stddev"),
    sigma_yaw: float = typer.Option(0.16, help="Yaw action exploration stddev"),
    max_forward: float = typer.Option(1.2, help="Max forward velocity for trainer policy"),
    max_yaw: float = typer.Option(1.2, help="Max yaw rate for trainer policy"),
    save_checkpoint: str = typer.Option(
        "",
        help="Optional output path (.npz) to save trained policy",
    ),
    checkpoint_every: int = typer.Option(
        0,
        help="Save periodic checkpoints every N steps (0 disables)",
    ),
    checkpoint_dir: str = typer.Option(
        "",
        help="Directory for periodic checkpoints",
    ),
    checkpoint_prefix: str = typer.Option(
        "policy_step",
        help="Filename prefix for periodic checkpoints",
    ),
    eval_every: int = typer.Option(
        0,
        help="Run deterministic evaluation every N steps (0 disables)",
    ),
    eval_episodes: int = typer.Option(
        0,
        help="Evaluation episodes per eval run",
    ),
    eval_horizon: int = typer.Option(
        100,
        help="Steps per evaluation episode",
    ),
    eval_csv: str = typer.Option(
        "",
        help="Optional CSV output path for periodic evaluation metrics",
    ),
    eval_plot: str = typer.Option(
        "",
        help="Optional PNG output path for evaluation progress plots",
    ),
) -> None:
    """Train a collision-aware policy from full spherical distance matrices."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        root_logger.setLevel(logging.INFO)

    trainer = OnlineSphericalTrainer(
        sub_address=sub,
        step_endpoint=step_endpoint,
        pub_address=pub,
        learning_rate=learning_rate,
        sigma_forward=sigma_forward,
        sigma_yaw=sigma_yaw,
        max_forward=max_forward,
        max_yaw=max_yaw,
    )
    trainer.start()
    try:
        metrics = trainer.train(
            steps=steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
            checkpoint_prefix=checkpoint_prefix,
            eval_every=eval_every,
            eval_episodes=eval_episodes,
            eval_horizon=eval_horizon,
        )
        if save_checkpoint:
            trainer.save_checkpoint(save_checkpoint)
    finally:
        trainer.stop()

    if eval_csv:
        trainer.save_eval_csv(eval_csv, metrics.eval_history)
    if eval_plot:
        trainer.plot_eval_progress(eval_plot, metrics.eval_history)

    typer.echo(
        "training complete | "
        f"steps={metrics.steps} "
        f"reward_mean={metrics.reward_mean:.4f} "
        f"reward_ema={metrics.reward_ema:.4f} "
        f"collision_rate={metrics.collision_rate:.3f} "
        f"novelty_rate={metrics.novelty_rate:.3f} "
        f"visited_cells={metrics.visited_cells} "
        f"forward_mean={metrics.forward_mean:.3f} "
        f"yaw_abs_mean={metrics.yaw_abs_mean:.3f} "
        f"vertical_abs_mean={metrics.vertical_abs_mean:.3f} "
        f"lateral_abs_mean={metrics.lateral_abs_mean:.3f} "
        f"eval_reward_mean={metrics.eval_reward_mean:.4f} "
        f"eval_collision_rate={metrics.eval_collision_rate:.3f} "
        f"eval_novelty_rate={metrics.eval_novelty_rate:.3f} "
        f"eval_coverage_mean={metrics.eval_coverage_mean:.3f}"
    )
    if save_checkpoint:
        typer.echo(f"checkpoint saved: {save_checkpoint}")
    if eval_csv:
        typer.echo(f"evaluation csv saved: {eval_csv}")
    if eval_plot:
        typer.echo(f"evaluation plot saved: {eval_plot}")


@app.command("train-ppo")
def train_ppo(
    sub: str = typer.Option("tcp://localhost:5559", help="ZMQ SUB address"),
    pub: str = typer.Option("tcp://*:5557", help="ZMQ PUB bind address for telemetry"),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560", help="Section Manager REP address",
    ),
    steps: int = typer.Option(10000, help="Total environment steps"),
    log_every: int = typer.Option(100, help="Log interval in steps"),
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension"),
    rollout_length: int = typer.Option(512, help="Steps per rollout"),
    ppo_epochs: int = typer.Option(4, help="PPO epochs per update"),
    minibatch_size: int = typer.Option(64, help="Minibatch size"),
    bptt_len: int = typer.Option(32, help="BPTT sequence length (0 = random)"),
    clip_ratio: float = typer.Option(0.2, help="PPO clip ratio"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    entropy_coeff: float = typer.Option(0.01, help="Entropy bonus coefficient"),
    value_coeff: float = typer.Option(0.005, help="Value loss coefficient (low to prevent gradient domination)"),
    # Reward shaping
    collision_penalty: float = typer.Option(0.0, help="Collision termination penalty (backend supplies its own)"),
    existential_tax: float = typer.Option(-0.01, help="Per-step existence cost"),
    velocity_weight: float = typer.Option(0.1, help="Forward-velocity heuristic weight"),
    # RND curiosity
    intrinsic_coeff_init: float = typer.Option(1.0, help="Initial RND intrinsic coefficient"),
    intrinsic_coeff_final: float = typer.Option(0.01, help="Final RND intrinsic coefficient"),
    intrinsic_anneal_steps: int = typer.Option(500_000, help="Steps to anneal intrinsic coeff"),
    rnd_learning_rate: float = typer.Option(3e-5, help="RND predictor learning rate"),
    # Episodic memory
    memory_capacity: int = typer.Option(10_000, help="Episodic memory max entries"),
    memory_exclusion_window: int = typer.Option(50, help="Exclude recent N steps from query"),
    loop_threshold: float = typer.Option(0.85, help="Cosine similarity loop threshold"),
    loop_penalty_coeff: float = typer.Option(0.5, help="Loop penalty coefficient"),
    # Checkpointing
    checkpoint_every: int = typer.Option(0, help="Checkpoint interval (0 = disabled)"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
    checkpoint: str = typer.Option("", help="Resume from checkpoint path (.pt)"),
) -> None:
    """Train a CognitiveMambaPolicy using PPO with RND curiosity and episodic memory."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        root_logger.setLevel(logging.INFO)

    config = ActorConfig(
        sub_address=sub,
        pub_address=pub,
        mode="step",
        step_endpoint=step_endpoint,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
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
        memory_capacity=memory_capacity,
        memory_exclusion_window=memory_exclusion_window,
        loop_threshold=loop_threshold,
        loop_penalty_coeff=loop_penalty_coeff,
    )

    from navi_actor.training.ppo_trainer import PpoTrainer

    trainer = PpoTrainer(config)

    if checkpoint:
        trainer.load_training_state(checkpoint)
        typer.echo(f"Resumed from checkpoint: {checkpoint}")

    trainer.start()
    try:
        metrics = trainer.train(
            total_steps=steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        trainer.stop()

    typer.echo(
        "PPO training complete | "
        f"steps={metrics.total_steps} "
        f"episodes={metrics.episodes} "
        f"reward_mean={metrics.reward_mean:.4f} "
        f"reward_ema={metrics.reward_ema:.4f} "
        f"policy_loss={metrics.policy_loss:.4f} "
        f"value_loss={metrics.value_loss:.4f} "
        f"entropy={metrics.entropy:.4f} "
        f"rnd_loss={metrics.rnd_loss:.4f} "
        f"intrinsic_mean={metrics.intrinsic_reward_mean:.4f} "
        f"loop_detections={metrics.loop_detections} "
        f"beta_final={metrics.beta_final:.4f}"
    )


@app.command("train-sequential")
def train_sequential(
    manifest: str = typer.Option(
        "data/scenes/sample_episodes.json",
        help="Path to scene manifest JSON file",
    ),
    actors: int = typer.Option(
        2,
        help="Number of actors sharing each scene",
    ),
    steps_per_scene: int = typer.Option(
        100_000, help="Environment steps per scene (across all actors)",
    ),
    backend: str = typer.Option("mesh", help="Simulation backend"),
    # PPO hyper-parameters
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension"),
    rollout_length: int = typer.Option(512, help="Steps per rollout"),
    ppo_epochs: int = typer.Option(4, help="PPO epochs per update"),
    minibatch_size: int = typer.Option(64, help="Minibatch size"),
    bptt_len: int = typer.Option(32, help="BPTT sequence length (0 = random)"),
    clip_ratio: float = typer.Option(0.2, help="PPO clip ratio"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    entropy_coeff: float = typer.Option(0.01, help="Entropy bonus coefficient"),
    value_coeff: float = typer.Option(0.005, help="Value loss coefficient (low to prevent gradient domination)"),
    # Reward shaping
    collision_penalty: float = typer.Option(0.0, help="Collision penalty (backend supplies its own)"),
    existential_tax: float = typer.Option(-0.01, help="Per-step existence cost"),
    velocity_weight: float = typer.Option(0.1, help="Forward-velocity weight"),
    # RND curiosity
    intrinsic_coeff_init: float = typer.Option(1.0, help="Initial RND coefficient"),
    intrinsic_coeff_final: float = typer.Option(0.01, help="Final RND coefficient"),
    intrinsic_anneal_steps: int = typer.Option(500_000, help="RND anneal steps"),
    rnd_learning_rate: float = typer.Option(3e-5, help="RND predictor learning rate"),
    # Episodic memory
    memory_capacity: int = typer.Option(10_000, help="Episodic memory capacity"),
    memory_exclusion_window: int = typer.Option(50, help="Exclude recent N steps from query"),
    loop_threshold: float = typer.Option(0.85, help="Loop similarity threshold"),
    loop_penalty_coeff: float = typer.Option(0.5, help="Loop penalty coefficient"),
    # Checkpointing
    checkpoint_every: int = typer.Option(0, help="Checkpoint interval (0 = disabled)"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
    checkpoint: str = typer.Option("", help="Resume from checkpoint path (.pt)"),
    # ZMQ ports (shared with SM)
    sm_pub: str = typer.Option("tcp://*:5559", help="SM PUB bind address"),
    sm_rep: str = typer.Option("tcp://*:5560", help="SM REP bind address"),
    actor_pub: str = typer.Option("tcp://*:5557", help="Actor PUB bind address"),
) -> None:
    """Sequential multi-scene training with N actors per scene.

    Loads scenes one at a time from a manifest. For each scene, launches
    a Section-Manager backend with ``--actors N`` and an Actor/Trainer
    that steps all N actors in round-robin. Policy weights (and optimizer
    / RND / reward-shaper state) accumulate across scenes.
    """
    import json
    from pathlib import Path

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        root_logger.setLevel(logging.INFO)

    # Load scene manifest
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        typer.echo(f"Manifest not found: {manifest_path}")
        raise typer.Exit(1)

    with manifest_path.open(encoding="utf-8-sig") as f:
        scene_data = json.load(f)

    # Extract scene paths
    scenes: list[str] = []
    raw_scenes: list[object] = []
    if isinstance(scene_data, list):
        raw_scenes = scene_data
    elif isinstance(scene_data, dict) and "scenes" in scene_data:
        raw_scenes = scene_data["scenes"]
    elif isinstance(scene_data, dict) and "episodes" in scene_data:
        # Extract unique scene_id values from episodes format
        seen: set[str] = set()
        for ep in scene_data["episodes"]:
            sid = ep.get("scene_id") or ep.get("scene")
            if sid and str(sid) not in seen:
                seen.add(str(sid))
                scenes.append(str(sid))
    for entry in raw_scenes:
        if isinstance(entry, dict):
            # Support both "scene" and "path" keys
            p = entry.get("scene") or entry.get("path")
            if p:
                scenes.append(str(p))
        elif isinstance(entry, str):
            scenes.append(entry)

    if not scenes:
        typer.echo("No scenes found in manifest.")
        raise typer.Exit(1)

    # Filter to scenes that exist on disk
    valid_scenes = [s for s in scenes if Path(s).exists()]
    skipped = len(scenes) - len(valid_scenes)
    if skipped:
        typer.echo(f"Skipping {skipped} scenes (files not found on disk)")
    scenes = valid_scenes

    if not scenes:
        typer.echo("No valid scene files found.")
        raise typer.Exit(1)

    typer.echo(
        f"Sequential training: {len(scenes)} scenes, {actors} actors/scene, "
        f"{steps_per_scene} steps/scene, backend={backend}"
    )

    config = ActorConfig(
        sub_address=f"tcp://localhost:{sm_pub.split(':')[-1]}",
        pub_address=actor_pub,
        mode="step",
        step_endpoint=f"tcp://localhost:{sm_rep.split(':')[-1]}",
        n_actors=actors,
        embedding_dim=embedding_dim,
        learning_rate=learning_rate,
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
        memory_capacity=memory_capacity,
        memory_exclusion_window=memory_exclusion_window,
        loop_threshold=loop_threshold,
        loop_penalty_coeff=loop_penalty_coeff,
    )

    from navi_actor.training.ppo_trainer import PpoTrainer
    from navi_section_manager.config import SectionManagerConfig  # type: ignore[import-not-found]
    from navi_section_manager.server import SectionManagerServer  # type: ignore[import-not-found]

    ckpt_path_latest: str | None = checkpoint or None

    for scene_idx, scene_path in enumerate(scenes):
        typer.echo(f"\n{'='*60}")
        typer.echo(f"Scene {scene_idx + 1}/{len(scenes)}: {scene_path}")
        typer.echo(f"{'='*60}")

        # Build SM config + backend for this scene
        sm_config = SectionManagerConfig(
            pub_address=sm_pub,
            rep_address=sm_rep,
            mode="step",
            n_actors=actors,
            backend=backend,
            habitat_scene=scene_path,
            azimuth_bins=config.azimuth_bins,
            elevation_bins=config.elevation_bins,
            max_distance=30.0,
            compute_overhead=False,
        )

        # Import and build backend
        if backend == "mesh":
            from navi_section_manager.backends.mesh_backend import MeshSceneBackend  # type: ignore[import-not-found]  # noqa: I001
            sim_backend = MeshSceneBackend(sm_config)
        else:
            from navi_section_manager.backends.voxel import VoxelBackend  # type: ignore[import-not-found]  # noqa: I001
            from navi_section_manager.generators.arena import ArenaGenerator  # type: ignore[import-not-found]
            gen = ArenaGenerator(seed=42)
            sim_backend = VoxelBackend(sm_config, gen)

        sm_server = SectionManagerServer(config=sm_config, backend=sim_backend)

        # Start SM in a background thread
        import threading
        sm_thread = threading.Thread(target=sm_server.run, daemon=True)
        sm_thread.start()

        # Small delay for ZMQ bind + PUB/SUB subscription handshake
        import time as _time
        _time.sleep(1.0)

        # Build trainer (create fresh for correct ZMQ lifecycle)
        trainer = PpoTrainer(config)

        # Load accumulated knowledge from previous scene
        if ckpt_path_latest:
            trainer.load_training_state(ckpt_path_latest)
            typer.echo(f"  Loaded checkpoint: {ckpt_path_latest}")

        trainer.start()
        try:
            metrics = trainer.train(
                total_steps=steps_per_scene,
                log_every=100,
                checkpoint_every=checkpoint_every,
                checkpoint_dir=checkpoint_dir,
            )
        finally:
            trainer.stop()
            sm_server.stop()
            sm_thread.join(timeout=5.0)

        # Save accumulated state
        scene_ckpt = (
            Path(checkpoint_dir)
            / f"policy_scene_{scene_idx:04d}.pt"
        )
        scene_ckpt.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_training_state(scene_ckpt)
        ckpt_path_latest = str(scene_ckpt)

        typer.echo(
            f"  Scene {scene_idx + 1} complete | "
            f"steps={metrics.total_steps} episodes={metrics.episodes} "
            f"reward_ema={metrics.reward_ema:.4f} "
            f"policy_loss={metrics.policy_loss:.4f}"
        )

        # Allow OS to release ports before next scene rebinds
        _time.sleep(1.0)

    typer.echo(f"\nAll {len(scenes)} scenes complete. Final checkpoint: {ckpt_path_latest}")


if __name__ == "__main__":
    app()
