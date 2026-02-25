"""Typer CLI for the Actor service."""

from __future__ import annotations

import logging

import typer

from navi_actor.config import ActorConfig
from navi_actor.server import ActorServer

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-actor", help="Layer 2: The Brain — Actor service")


@app.command()
def run(
    sub: str = typer.Option("tcp://localhost:5559", help="ZMQ SUB address (Simulation Layer)"),
    pub: str = typer.Option("tcp://*:5557", help="ZMQ PUB bind address"),
    mode: str = typer.Option("async", help="Mode: async (PUB/SUB) or step (REQ/REP training)"),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560",
        help="Environment REP address (step mode)",
    ),
    policy_checkpoint: str = typer.Option(
        "",
        help="Checkpoint path (.pt) to load pre-trained weights",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(128, help="Expected distance-matrix elevation resolution"),
    encoder: str = typer.Option("cnn", help="Encoder type: cnn or vit"),
) -> None:
    """Start the Actor service (always uses CognitiveMambaPolicy)."""
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
        encoder_type=encoder,
    )

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
            encoder_type=config.encoder_type,
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
            encoder_type=config.encoder_type,
        )

    server = ActorServer(config=config, policy=runtime_policy)
    typer.echo(
        f"Actor starting — mode={config.mode}, sub={config.sub_address}, "
        f"pub={config.pub_address}, policy=cognitive"
    )
    server.run()


@app.command("train-ppo")
def train_ppo(
    sub: str = typer.Option("tcp://localhost:5559", help="ZMQ SUB address"),
    pub: str = typer.Option("tcp://*:5557", help="ZMQ PUB bind address for telemetry"),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560", help="Environment REP address",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(128, help="Expected distance-matrix elevation resolution"),
    encoder: str = typer.Option("cnn", help="Encoder type: cnn or vit"),
    actors: int = typer.Option(1, help="Number of parallel environments"),
    steps: int = typer.Option(10000, help="Total environment steps"),
    log_every: int = typer.Option(100, help="Log interval in steps"),
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    learning_rate_final: float = typer.Option(3e-5, help="Final annealed learning rate"),
    embedding_dim: int = typer.Option(128, help="Encoder embedding dimension"),
    rollout_length: int = typer.Option(512, help="Steps per rollout"),
    ppo_epochs: int = typer.Option(4, help="PPO epochs per update"),
    minibatch_size: int = typer.Option(64, help="Minibatch size"),
    bptt_len: int = typer.Option(32, help="BPTT sequence length (0 = random)"),
    clip_ratio: float = typer.Option(0.2, help="PPO clip ratio"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    gae_lambda: float = typer.Option(0.95, help="GAE lambda"),
    entropy_coeff: float = typer.Option(0.01, help="Entropy bonus coefficient"),
    value_coeff: float = typer.Option(0.5, help="Value loss coefficient"),
    # Reward shaping
    collision_penalty: float = typer.Option(0.0, help="Collision termination penalty (backend supplies its own)"),
    existential_tax: float = typer.Option(-0.02, help="Per-step existence cost"),
    velocity_weight: float = typer.Option(0.1, help="Forward-velocity heuristic weight"),
    # RND curiosity
    intrinsic_coeff_init: float = typer.Option(1.0, help="Initial RND intrinsic coefficient"),
    intrinsic_coeff_final: float = typer.Option(0.01, help="Final RND intrinsic coefficient"),
    intrinsic_anneal_steps: int = typer.Option(500_000, help="Steps to anneal intrinsic coeff"),
    rnd_learning_rate: float = typer.Option(3e-5, help="RND predictor learning rate"),
    rnd_learning_rate_final: float = typer.Option(3e-6, help="Final RND learning rate"),
    # Episodic memory
    memory_capacity: int = typer.Option(10_000, help="Episodic memory max entries"),
    memory_exclusion_window: int = typer.Option(50, help="Exclude recent N steps from query"),
    loop_threshold: float = typer.Option(0.85, help="Cosine similarity loop threshold"),
    loop_penalty_coeff: float = typer.Option(2.0, help="Loop penalty coefficient"),
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
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        encoder_type=encoder,
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
        "data/scenes/scene_manifest.json",
        help="Path to scene manifest JSON file",
    ),
    actors: int = typer.Option(
        2,
        help="Number of actors sharing each scene",
    ),
    azimuth_bins: int = typer.Option(256, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(128, help="Expected distance-matrix elevation resolution"),
    encoder: str = typer.Option("cnn", help="Encoder type: cnn or vit"),
    total_steps: int = typer.Option(
        500_000, help="Total environment steps (across all scenes)",
    ),
    shuffle: bool = typer.Option(
        True, help="Shuffle scene order for diversity",
    ),
    min_scene_bytes: int = typer.Option(
        100_000, help="Minimum scene file size in bytes (filters stubs)",
    ),
    backend: str = typer.Option("mesh", help="Simulation backend"),
    # PPO hyper-parameters
    learning_rate: float = typer.Option(3e-4, help="Adam learning rate"),
    learning_rate_final: float = typer.Option(3e-5, help="Final annealed learning rate"),
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
    rnd_learning_rate_final: float = typer.Option(3e-6, help="Final RND learning rate"),
    # Episodic memory
    memory_capacity: int = typer.Option(10_000, help="Episodic memory capacity"),
    memory_exclusion_window: int = typer.Option(50, help="Exclude recent N steps from query"),
    loop_threshold: float = typer.Option(0.85, help="Loop similarity threshold"),
    loop_penalty_coeff: float = typer.Option(0.5, help="Loop penalty coefficient"),
    # Checkpointing
    checkpoint_every: int = typer.Option(0, help="Checkpoint interval (0 = disabled)"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint directory"),
    checkpoint: str = typer.Option("", help="Resume from checkpoint path (.pt)"),
    # Logging
    log_every: int = typer.Option(100, help="Steps between log messages"),
    # ZMQ ports (shared with environment)
    env_pub: str = typer.Option("tcp://*:5559", help="Environment PUB bind address"),
    env_rep: str = typer.Option("tcp://*:5560", help="Environment REP bind address"),
    actor_pub: str = typer.Option("tcp://*:5557", help="Actor PUB bind address"),
) -> None:
    """Multi-scene PPO training with automatic scene cycling.

    Loads scene paths from a manifest JSON, filters to valid files,
    and passes the full pool to a single MeshSceneBackend.  The
    backend automatically cycles to the next scene every time
    ``n_actors`` natural episode completions occur (collisions
    or horizon truncations).

    The policy, optimizer, RND, and reward-shaper state accumulate
    continuously across all scenes — no checkpoint hand-off needed.
    """
    import json
    import random as _random
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

    # Extract scene paths from manifest (support multiple formats)
    scenes: list[str] = []
    raw_scenes: list[object] = []
    if isinstance(scene_data, list):
        raw_scenes = scene_data
    elif isinstance(scene_data, dict) and "scenes" in scene_data:
        raw_scenes = scene_data["scenes"]
    elif isinstance(scene_data, dict) and "episodes" in scene_data:
        seen: set[str] = set()
        for ep in scene_data["episodes"]:
            sid = ep.get("scene_id") or ep.get("scene")
            if sid and str(sid) not in seen:
                seen.add(str(sid))
                scenes.append(str(sid))
    for entry in raw_scenes:
        if isinstance(entry, dict):
            p = entry.get("scene") or entry.get("path")
            if p:
                scenes.append(str(p))
        elif isinstance(entry, str):
            scenes.append(entry)

    if not scenes:
        typer.echo("No scenes found in manifest.")
        raise typer.Exit(1)

    # Filter to existing files above minimum size (catches stubs)
    valid_scenes: list[str] = []
    for s in scenes:
        p = Path(s)
        if p.exists() and p.stat().st_size >= min_scene_bytes:
            valid_scenes.append(s)
    skipped = len(scenes) - len(valid_scenes)
    if skipped:
        typer.echo(f"Filtered out {skipped} scenes (missing or < {min_scene_bytes} bytes)")
    scenes = valid_scenes

    if not scenes:
        typer.echo("No valid scene files found.")
        raise typer.Exit(1)

    if shuffle:
        _random.shuffle(scenes)

    typer.echo(
        f"Scene pool: {len(scenes)} scenes, {actors} actors, "
        f"{total_steps} total steps, shuffle={shuffle}, backend={backend}"
    )
    for i, s in enumerate(scenes[:5]):
        typer.echo(f"  [{i+1}] {Path(s).stem}")
    if len(scenes) > 5:
        typer.echo(f"  ... and {len(scenes) - 5} more")

    config = ActorConfig(
        sub_address=f"tcp://localhost:{env_pub.split(':')[-1]}",
        pub_address=actor_pub,
        mode="step",
        step_endpoint=f"tcp://localhost:{env_rep.split(':')[-1]}",
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        encoder_type=encoder,
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
    )

    from navi_actor.training.ppo_trainer import PpoTrainer
    from navi_environment.config import EnvironmentConfig  # type: ignore[import-not-found]
    from navi_environment.server import EnvironmentServer  # type: ignore[import-not-found]

    # Build environment config with scene pool
    env_config = EnvironmentConfig(
        pub_address=env_pub,
        rep_address=env_rep,
        mode="step",
        training_mode=True,
        n_actors=actors,
        backend=backend,
        habitat_scene=scenes[0],
        scene_pool=tuple(scenes),
        azimuth_bins=config.azimuth_bins,
        elevation_bins=config.elevation_bins,
        max_distance=30.0,
        compute_overhead=False,
    )

    # Build backend with scene pool
    if backend == "mesh":
        from navi_environment.backends.mesh_backend import MeshSceneBackend  # type: ignore[import-not-found]  # noqa: I001
        sim_backend = MeshSceneBackend(env_config)
    else:
        from navi_environment.backends.voxel import VoxelBackend  # type: ignore[import-not-found]  # noqa: I001
        from navi_environment.generators.arena import ArenaGenerator  # type: ignore[import-not-found]
        gen = ArenaGenerator(seed=42)
        sim_backend = VoxelBackend(env_config, gen)

    env_server = EnvironmentServer(config=env_config, backend=sim_backend)

    # Start environment in a background thread
    import threading
    import time as _time
    env_thread = threading.Thread(target=env_server.run, daemon=True)
    env_thread.start()
    _time.sleep(3.0)  # allow scene loading + ZMQ bind

    # Build trainer
    trainer = PpoTrainer(config)

    if checkpoint:
        trainer.load_training_state(checkpoint)
        typer.echo(f"Loaded checkpoint: {checkpoint}")

    trainer.start()
    try:
        metrics = trainer.train(
            total_steps=total_steps,
            log_every=log_every,
            checkpoint_every=checkpoint_every,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        trainer.stop()
        env_server.stop()
        env_thread.join(timeout=5.0)

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


if __name__ == "__main__":
    app()
