"""Typer CLI for the Auditor service."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer
import zmq

from navi_auditor.config import AuditorConfig
from navi_contracts import TOPIC_ACTION, TOPIC_DISTANCE_MATRIX, TOPIC_TELEMETRY_EVENT, setup_logging

if TYPE_CHECKING:
    from navi_auditor.matrix_viewer import MatrixViewer
    from navi_auditor.recorder import Recorder
    from navi_auditor.rewinder import Rewinder
    from navi_auditor.storage.zarr_backend import ZarrBackend

MatrixViewer: Any | None = None
Recorder: Any | None = None
Rewinder: Any | None = None
ZarrBackend: Any | None = None


def _get_matrix_viewer_class() -> Any:
    global MatrixViewer
    if MatrixViewer is None:
        from navi_auditor.matrix_viewer import MatrixViewer as _MatrixViewer

        MatrixViewer = _MatrixViewer
    return MatrixViewer


def _get_recorder_class() -> Any:
    global Recorder
    if Recorder is None:
        from navi_auditor.recorder import Recorder as _Recorder

        Recorder = _Recorder
    return Recorder


def _get_rewinder_class() -> Any:
    global Rewinder
    if Rewinder is None:
        from navi_auditor.rewinder import Rewinder as _Rewinder

        Rewinder = _Rewinder
    return Rewinder


def _get_zarr_backend_class() -> Any:
    global ZarrBackend
    if ZarrBackend is None:
        from navi_auditor.storage.zarr_backend import ZarrBackend as _ZarrBackend

        ZarrBackend = _ZarrBackend
    return ZarrBackend

__all__: list[str] = ["app"]

app = typer.Typer(name="navi-auditor", help="Layer 3: Viz & Replay — Auditor service")


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent
    msg = "Unable to locate repository root from auditor CLI"
    raise RuntimeError(msg)


def _environment_project_root() -> Path:
    return _repo_root() / "projects" / "environment"


def _environment_cli_command() -> list[str]:
    return ["uv", "run", "--project", str(_environment_project_root()), "navi-environment"]


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        msg = f"Expected JSON object in {path}, got {type(payload).__name__}"
        raise RuntimeError(msg)
    return cast("dict[str, Any]", payload)


def _parse_json_stdout(command_name: str, stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        root_start = stripped.find("{")
        if root_start < 0:
            msg = f"Environment CLI {command_name} did not emit parseable JSON: {exc}"
            raise RuntimeError(msg) from exc
        try:
            payload = json.loads(stripped[root_start:])
        except json.JSONDecodeError as nested_exc:
            msg = f"Environment CLI {command_name} did not emit parseable JSON: {nested_exc}"
            raise RuntimeError(msg) from nested_exc
    if not isinstance(payload, dict):
        msg = f"Environment CLI {command_name} emitted {type(payload).__name__}, expected a JSON object"
        raise RuntimeError(msg)
    return cast("dict[str, Any]", payload)


def _run_environment_json_command(command_name: str, arguments: list[str]) -> tuple[int, dict[str, Any]]:
    completed = subprocess.run(  # noqa: S603 - command is fixed to the repo-local environment CLI surface
        [*_environment_cli_command(), *arguments, "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
    )
    stdout = completed.stdout.strip()
    if not stdout:
        stderr = completed.stderr.strip()
        msg = f"Environment CLI {command_name} produced no JSON output"
        if stderr:
            msg = f"{msg}: {stderr}"
        raise RuntimeError(msg)
    return completed.returncode, _parse_json_stdout(command_name, stdout)


def _resolve_benchmark_gmdag_file(gmdag_file: str, check_payload: dict[str, Any]) -> str:
    if gmdag_file:
        return str(Path(gmdag_file).expanduser().resolve())

    corpus_payload = check_payload.get("corpus")
    if not isinstance(corpus_payload, dict):
        msg = "dataset-audit benchmark requires --gmdag-file or a valid promoted corpus manifest"
        raise RuntimeError(msg)

    manifest_value = corpus_payload.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        msg = "dataset-audit benchmark could not resolve the promoted corpus manifest"
        raise RuntimeError(msg)

    manifest_path = Path(manifest_value).expanduser().resolve()
    manifest_payload = _load_json_file(manifest_path)
    scenes = manifest_payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        msg = f"Promoted corpus manifest {manifest_path} does not contain any compiled scenes"
        raise RuntimeError(msg)

    first_scene = scenes[0]
    if not isinstance(first_scene, dict):
        msg = f"Promoted corpus manifest {manifest_path} has an invalid scene entry"
        raise RuntimeError(msg)

    gmdag_path_value = first_scene.get("gmdag_path")
    if not isinstance(gmdag_path_value, str) or not gmdag_path_value:
        msg = f"Promoted corpus manifest {manifest_path} is missing a scene gmdag_path"
        raise RuntimeError(msg)

    gmdag_path = Path(gmdag_path_value).expanduser()
    if not gmdag_path.is_absolute():
        gmdag_path = (manifest_path.parent / gmdag_path).resolve()
    else:
        gmdag_path = gmdag_path.resolve()
    return str(gmdag_path)


def _echo_dataset_audit_summary(summary: dict[str, Any]) -> None:
    typer.echo(
        "dataset-audit - "
        f"ok={summary['ok']}, "
        f"check_ok={summary['check']['ok']}, "
        f"benchmark_ok={summary['benchmark']['ok'] if summary['benchmark'] is not None else 'skipped'}"
    )
    typer.echo(f"check_profile={summary['check']['profile']}")
    if summary["benchmark"] is not None:
        typer.echo(
            "benchmark - "
            f"gmdag={summary['benchmark']['gmdag_file']}, "
            f"actors={summary['benchmark']['actors']}, "
            f"steps={summary['benchmark']['steps']}, "
            f"measured_sps={summary['benchmark']['measured_sps']:.2f}"
        )
    for issue in cast("list[str]", summary["issues"]):
        typer.echo(f"issue={issue}", err=True)


def _wait_for_dashboard_attach(actor_sub: str, timeout_seconds: float) -> dict[str, Any]:
    context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    socket = context.socket(zmq.SUB)
    poller = zmq.Poller()
    started_at = time.perf_counter()
    topic_filters = (TOPIC_TELEMETRY_EVENT, TOPIC_ACTION, TOPIC_DISTANCE_MATRIX)

    try:
        socket.connect(actor_sub)
        for topic in topic_filters:
            socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
        poller.register(socket, zmq.POLLIN)

        while True:
            elapsed_seconds = time.perf_counter() - started_at
            remaining_ms = int(max(0.0, (timeout_seconds - elapsed_seconds) * 1000.0))
            if remaining_ms <= 0:
                break
            events = dict(poller.poll(min(remaining_ms, 250)))
            if socket in events:
                topic_bytes, payload = socket.recv_multipart()
                topic = topic_bytes.decode("utf-8")
                return {
                    "profile": "dashboard-attach-check",
                    "ok": True,
                    "actor_sub": actor_sub,
                    "topic": topic,
                    "payload_bytes": len(payload),
                    "elapsed_seconds": round(time.perf_counter() - started_at, 6),
                }
    finally:
        poller.unregister(socket)
        socket.close(linger=0)
        context.term()

    return {
        "profile": "dashboard-attach-check",
        "ok": False,
        "actor_sub": actor_sub,
        "topic": "",
        "payload_bytes": 0,
        "elapsed_seconds": round(time.perf_counter() - started_at, 6),
        "issues": [f"No dashboard-observable actor traffic arrived on {actor_sub} within {timeout_seconds:.1f}s"],
    }


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
    addresses = tuple(s.strip() for s in sub.split(",") if s.strip()) if sub else config.sub_addresses

    matrix_sub_address = config.matrix_sub_address
    actor_sub_address = config.actor_sub_address
    if sub:
        if len(addresses) == 1:
            matrix_sub_address = addresses[0]
            actor_sub_address = ""
        elif len(addresses) >= 2:
            matrix_sub_address = addresses[0]
            actor_sub_address = addresses[1]

    config = AuditorConfig(
        matrix_sub_address=matrix_sub_address,
        actor_sub_address=actor_sub_address,
        output_path=final_out
    )

    storage = _get_zarr_backend_class()()
    recorder = _get_recorder_class()(config=config, backend=storage)
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
    storage = _get_zarr_backend_class()()
    rewinder = _get_rewinder_class()(config=config, backend=storage)
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
    dashboard_runner = _get_matrix_viewer_class()(
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


@app.command("dashboard-attach-check")
def dashboard_attach_check(
    actor_sub: str = typer.Option(None, help="Actor or replay PUB SUB address to observe."),
    timeout_seconds: float = typer.Option(15.0, min=0.1, help="Maximum time to wait for actor-visible traffic."),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON summary instead of text."),
) -> None:
    """Headless proof that the passive dashboard surface can attach to a live actor-style stream."""
    setup_logging("navi_auditor_dashboard_attach_check")
    config = AuditorConfig()
    resolved_actor_sub = actor_sub if actor_sub is not None else config.actor_sub_address
    summary = _wait_for_dashboard_attach(resolved_actor_sub, timeout_seconds)

    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    elif summary["ok"]:
        typer.echo(
            "dashboard-attach-check - "
            f"ok=True, actor_sub={summary['actor_sub']}, topic={summary['topic']}, "
            f"payload_bytes={summary['payload_bytes']}, elapsed_seconds={summary['elapsed_seconds']:.3f}"
        )
    else:
        typer.echo(
            "dashboard-attach-check - "
            f"ok=False, actor_sub={summary['actor_sub']}, elapsed_seconds={summary['elapsed_seconds']:.3f}",
            err=True,
        )
        for issue in cast("list[str]", summary.get("issues", [])):
            typer.echo(f"issue={issue}", err=True)

    if not summary["ok"]:
        raise typer.Exit(code=1)


@app.command("dataset-audit")
def dataset_audit(
    gmdag_file: str = typer.Option(
        "",
        help="Optional .gmdag asset override. When omitted, use the first promoted corpus asset.",
    ),
    expected_resolution: int = typer.Option(512, help="Expected canonical compiled resolution."),
    benchmark: bool = typer.Option(True, help="Run a real-runtime benchmark after runtime preflight succeeds."),
    actors: int = typer.Option(1, min=1, help="Number of actors for the benchmark batch."),
    steps: int = typer.Option(8, min=1, help="Measured benchmark batch steps."),
    warmup_steps: int = typer.Option(1, min=0, help="Unmeasured benchmark warmup steps."),
    azimuth_bins: int = typer.Option(64, min=1, help="Benchmark azimuth bins."),
    elevation_bins: int = typer.Option(16, min=1, help="Benchmark elevation bins."),
    max_distance: float = typer.Option(30.0, min=0.01, help="Benchmark distance normalization range."),
    sdf_max_steps: int = typer.Option(256, min=1, help="Benchmark maximum sphere-tracing iterations per ray."),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON summary instead of text."),
) -> None:
    """Run the first passive runtime-backed dataset QA surface through the environment CLI."""
    setup_logging("navi_auditor_dataset_audit")

    check_arguments = ["check-sdfdag", "--expected-resolution", str(expected_resolution)]
    if gmdag_file:
        check_arguments.extend(["--gmdag-file", gmdag_file])

    check_returncode, check_payload = _run_environment_json_command("check-sdfdag", check_arguments)
    issues = list(cast("list[str]", check_payload.get("issues", [])))

    benchmark_payload: dict[str, Any] | None = None
    benchmark_ok = not benchmark
    if benchmark and check_returncode == 0:
        resolved_gmdag_file = _resolve_benchmark_gmdag_file(gmdag_file, check_payload)
        benchmark_arguments = [
            "bench-sdfdag",
            "--gmdag-file",
            resolved_gmdag_file,
            "--actors",
            str(actors),
            "--steps",
            str(steps),
            "--warmup-steps",
            str(warmup_steps),
            "--azimuth-bins",
            str(azimuth_bins),
            "--elevation-bins",
            str(elevation_bins),
            "--max-distance",
            str(max_distance),
            "--sdf-max-steps",
            str(sdf_max_steps),
        ]
        benchmark_returncode, benchmark_payload = _run_environment_json_command("bench-sdfdag", benchmark_arguments)
        benchmark_ok = benchmark_returncode == 0
        if not benchmark_ok:
            issues.extend(cast("list[str]", benchmark_payload.get("issues", [])))

    summary = {
        "profile": "dataset-audit",
        "check": {
            **check_payload,
            "ok": check_returncode == 0 and bool(check_payload.get("ok", False)),
        },
        "benchmark": None,
        "issues": issues,
        "ok": check_returncode == 0 and benchmark_ok and not issues,
    }
    if benchmark_payload is not None:
        summary["benchmark"] = {
            **benchmark_payload,
            "ok": benchmark_ok,
        }

    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    else:
        _echo_dataset_audit_summary(summary)

    if not summary["ok"]:
        raise typer.Exit(code=1)


def dashboard_shortcut() -> None:
    """Shortcut for 'navi-auditor dashboard' command."""
    app(["dashboard"])


if __name__ == "__main__":
    app()
