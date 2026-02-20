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
    azimuth_bins: int = typer.Option(64, help="Expected distance-matrix azimuth resolution"),
    elevation_bins: int = typer.Option(32, help="Expected distance-matrix elevation resolution"),
) -> None:
    """Start the Actor service."""
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
    # Reward shaping
    collision_penalty: float = typer.Option(-10.0, help="Collision termination penalty"),
    existential_tax: float = typer.Option(-0.01, help="Per-step existence cost"),
    velocity_weight: float = typer.Option(0.1, help="Forward-velocity heuristic weight"),
    # RND curiosity
    intrinsic_coeff_init: float = typer.Option(1.0, help="Initial RND intrinsic coefficient"),
    intrinsic_coeff_final: float = typer.Option(0.01, help="Final RND intrinsic coefficient"),
    intrinsic_anneal_steps: int = typer.Option(500_000, help="Steps to anneal intrinsic coeff"),
    rnd_learning_rate: float = typer.Option(1e-3, help="RND predictor learning rate"),
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
        from navi_actor.cognitive_policy import CognitiveMambaPolicy

        trainer._policy = CognitiveMambaPolicy.load_checkpoint(
            checkpoint,
            embedding_dim=embedding_dim,
        ).to(trainer._device)
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


if __name__ == "__main__":
    app()
