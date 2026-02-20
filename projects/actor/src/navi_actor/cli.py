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
    policy: str = typer.Option("shallow", help="Policy type: shallow or learned"),
    policy_checkpoint: str = typer.Option(
        "",
        help="Checkpoint path (.npz) when --policy=learned",
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


if __name__ == "__main__":
    app()
