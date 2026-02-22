"""Typer CLI for the Auditor service."""

from __future__ import annotations

import typer

from navi_auditor.config import AuditorConfig
from navi_auditor.matrix_viewer import MatrixViewer
from navi_auditor.recorder import Recorder
from navi_auditor.rewinder import Rewinder
from navi_auditor.storage.zarr_backend import ZarrBackend

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-auditor", help="Layer 5: Viz & Replay — Auditor service")


@app.command()
def record(
    sub: str = typer.Option(
        "tcp://localhost:5559,tcp://localhost:5557",
        help="Comma-separated ZMQ SUB addresses",
    ),
    out: str = typer.Option("session.zarr", help="Output path"),
) -> None:
    """Record DistanceMatrix v2 and Action v2 messages to storage."""
    addresses = tuple(s.strip() for s in sub.split(","))
    config = AuditorConfig(sub_addresses=addresses, output_path=out)
    storage = ZarrBackend()
    recorder = Recorder(config=config, backend=storage)
    typer.echo(f"Auditor recording — subs={addresses}, out={out}")
    recorder.run()


@app.command()
def replay(
    input_path: str = typer.Option(..., "--input", help="Input session path"),
    pub: str = typer.Option("tcp://*:5558", help="ZMQ PUB bind address"),
    speed: float = typer.Option(1.0, help="Playback speed multiplier"),
) -> None:
    """Replay a recorded session via ZMQ PUB."""
    config = AuditorConfig(output_path=input_path, pub_address=pub)
    storage = ZarrBackend()
    rewinder = Rewinder(config=config, backend=storage)
    rewinder.start()
    typer.echo(f"Auditor replaying — input={input_path}, pub={pub}, speed={speed}x")
    count = rewinder.replay(speed=speed)
    typer.echo(f"Replayed {count} messages.")
    rewinder.stop()


@app.command()
def dashboard(
    matrix_sub: str = typer.Option(
        "tcp://localhost:5559",
        help="Simulation Layer DistanceMatrix SUB address",
    ),
    actor_sub: str = typer.Option(
        "tcp://localhost:5557",
        help="Actor/Trainer PUB SUB address (action_v2 + telemetry)",
    ),
    step_endpoint: str = typer.Option(
        "tcp://localhost:5560",
        help="Section Manager REP address for Tab-toggle manual stepping",
    ),
    actors: int = typer.Option(
        1, help="Number of actors (adds per-actor tabs when > 1)",
    ),
    hz: float = typer.Option(30.0, help="Dashboard + teleop tick rate"),
    linear_speed: float = typer.Option(1.5, help="Max horizontal linear speed"),
    yaw_rate: float = typer.Option(1.5, help="Max yaw rate"),
    scene: str = typer.Option(
        "",
        help="Path to .glb/.obj mesh for 3D environment view",
    ),
) -> None:
    """Run live high-performance RL training visualiser (PyQtGraph)."""
    typer.echo("Initialising Ghost-Matrix RL Dashboard...")
    typer.echo(f"  Actors: {actors}")
    typer.echo("  Tab=toggle manual/AI  WASD/arrows=move  ESC=quit")

    scene_path = scene if scene else None
    dashboard_runner = MatrixViewer(
        matrix_sub=matrix_sub,
        actor_sub=actor_sub,
        step_endpoint=step_endpoint,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        scene_path=scene_path,
        n_actors=actors,
    )
    dashboard_runner.run()


if __name__ == "__main__":
    app()
