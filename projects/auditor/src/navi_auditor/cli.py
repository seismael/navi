"""Typer CLI for the Auditor service."""

from __future__ import annotations

import typer

from navi_auditor.config import AuditorConfig
from navi_auditor.matrix_viewer import MatrixViewer
from navi_auditor.recorder import Recorder
from navi_auditor.rewinder import Rewinder
from navi_auditor.storage.zarr_backend import ZarrBackend
from navi_contracts import setup_logging

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-auditor", help="Layer 3: Viz & Replay — Auditor service")


@app.command()
def record(
    sub: str = typer.Option(
        None,
        help="Comma-separated ZMQ SUB addresses",
    ),
    out: str = typer.Option(None, help="Output path"),
) -> None:
    """Record DistanceMatrix v2 and Action v2 messages to storage."""
    setup_logging("navi_auditor_recorder")
    config = AuditorConfig()

    final_out = out or config.output_path
    addresses = tuple(s.strip() for s in sub.split(",")) if sub else config.sub_addresses

    config = AuditorConfig(
        matrix_sub_address=addresses[0] if len(addresses) > 0 else config.matrix_sub_address,
        actor_sub_address=addresses[1] if len(addresses) > 1 else config.actor_sub_address,
        output_path=final_out
    )

    storage = ZarrBackend()
    recorder = Recorder(config=config, backend=storage)
    typer.echo(f"Auditor recording — subs={addresses}, out={final_out}")
    recorder.run()


@app.command()
def replay(
    input_path: str = typer.Option(..., "--input", help="Input session path"),
    pub: str = typer.Option(None, help="ZMQ PUB bind address"),
    speed: float = typer.Option(1.0, help="Playback speed multiplier"),
) -> None:
    """Replay a recorded session via ZMQ PUB."""
    setup_logging("navi_auditor_rewinder")
    config = AuditorConfig()
    final_pub = pub or config.pub_address

    config = AuditorConfig(output_path=input_path, pub_address=final_pub)
    storage = ZarrBackend()
    rewinder = Rewinder(config=config, backend=storage)
    rewinder.start()
    typer.echo(f"Auditor replaying — input={input_path}, pub={final_pub}, speed={speed}x")
    count = rewinder.replay(speed=speed)
    typer.echo(f"Replayed {count} messages.")
    rewinder.stop()


@app.command()
def dashboard(
    matrix_sub: str = typer.Option(
        None,
        help="Simulation Layer DistanceMatrix SUB address",
    ),
    actor_sub: str = typer.Option(
        None,
        help="Actor/Trainer PUB SUB address (action_v2 + telemetry)",
    ),
    step_endpoint: str = typer.Option(
        None,
        help="Environment REP address for Tab-toggle manual stepping",
    ),
    passive: bool = typer.Option(
        False,
        help="Run in passive actor-only mode with no environment SUB/REQ sockets.",
    ),
    actor_id: int = typer.Option(
        0,
        help="Actor ID to display (default: 0).",
    ),
    enable_actor_selector: bool = typer.Option(
        False,
        help="Enable actor selector UI. Disabled by default for maximum throughput.",
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
    setup_logging("navi_auditor_dashboard")
    config = AuditorConfig()

    # CLI overrides
    m_sub = "" if passive else (matrix_sub if matrix_sub is not None else config.matrix_sub_address)
    a_sub = actor_sub if actor_sub is not None else config.actor_sub_address
    s_end = "" if passive else (step_endpoint if step_endpoint is not None else config.step_endpoint)

    typer.echo("Initialising Ghost-Matrix RL Dashboard...")
    typer.echo("  Tab=toggle manual/AI  WASD/arrows=move  ESC=quit")
    if passive:
        typer.echo("  Mode=passive actor-only (no environment control sockets)")

    scene_path = scene if scene else None
    dashboard_runner = MatrixViewer(
        matrix_sub=m_sub,
        actor_sub=a_sub,
        step_endpoint=s_end,
        actor_id=actor_id,
        enable_actor_selector=enable_actor_selector,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        scene_path=scene_path,
    )
    dashboard_runner.run()


def dashboard_shortcut() -> None:
    """Shortcut for 'navi-auditor dashboard' command."""
    app(["dashboard"])


if __name__ == "__main__":
    app()
