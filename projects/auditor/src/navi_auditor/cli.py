"""Typer CLI for the Auditor service."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import cv2
import numpy as np
import typer
import zmq

from navi_auditor.config import AuditorConfig
from navi_auditor.dashboard.renderers import depth_to_viridis, render_first_person
from navi_contracts import (
    TOPIC_ACTION,
    TOPIC_DISTANCE_MATRIX,
    TOPIC_TELEMETRY_EVENT,
    get_or_create_run_context,
    setup_logging,
)

if TYPE_CHECKING:
    from navi_auditor.matrix_viewer import MatrixViewer as _MatrixViewerType
    from navi_auditor.recorder import Recorder as _RecorderType
    from navi_auditor.rewinder import Rewinder as _RewinderType
    from navi_auditor.storage.zarr_backend import ZarrBackend as _ZarrBackendType

MatrixViewer: type[_MatrixViewerType] | None = None
Recorder: type[_RecorderType] | None = None
Rewinder: type[_RewinderType] | None = None
ZarrBackend: type[_ZarrBackendType] | None = None


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
    import sys
    import os
    if sys.platform == "win32":
        python_exe = _environment_project_root() / ".venv" / "Scripts" / "python.exe"
    else:
        python_exe = _environment_project_root() / ".venv" / "bin" / "python"
    
    if python_exe.exists():
        return [str(python_exe), "-m", "navi_environment.cli"]
        
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


def _run_environment_json_command(
    command_name: str, arguments: list[str]
) -> tuple[int, dict[str, Any]]:
    import os
    env = {}
    for k, v in os.environ.items():
        if k.startswith("UV_"): continue
        if k.startswith("PYTHON"): continue
        if k == "VIRTUAL_ENV": continue
        env[k] = v
    if "PATH" in env:
        env["PATH"] = os.pathsep.join(
            p for p in env["PATH"].split(os.pathsep) if ".venv" not in p and "projects\\auditor" not in p
        )

    completed = subprocess.run(  # noqa: S603 - command is fixed to the repo-local environment CLI surface
        [*_environment_cli_command(), *arguments, "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=_repo_root(),
        env=env,
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
        "issues": [
            f"No dashboard-observable actor traffic arrived on {actor_sub} within {timeout_seconds:.1f}s"
        ],
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
    addresses = (
        tuple(s.strip() for s in sub.split(",") if s.strip()) if sub else config.sub_addresses
    )

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
        output_path=final_out,
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
    hz: float = typer.Option(30.0, help="Dashboard + teleop tick rate"),
    linear_speed: float = typer.Option(1.5, help="Max horizontal linear speed"),
    yaw_rate: float = typer.Option(1.5, help="Max yaw rate"),
    max_distance: float | None = typer.Option(
        None, min=0.01, help="Observation normalization horizon in meters."
    ),
    scene: str = typer.Option(
        "",
        help="Path to .glb/.obj mesh for 3D environment view",
    ),
) -> None:
    """Run live high-performance RL training visualiser (PyQtGraph)."""
    setup_logging("navi_auditor_dashboard")
    config = AuditorConfig()

    # CLI overrides
    m_sub = (
        "" if passive else (matrix_sub if matrix_sub is not None else config.matrix_sub_address)
    )
    a_sub = actor_sub if actor_sub is not None else config.actor_sub_address
    s_end = (
        "" if passive else (step_endpoint if step_endpoint is not None else config.step_endpoint)
    )
    resolved_max_distance = (
        float(max_distance)
        if max_distance is not None
        else float(config.observation_max_distance_m)
    )

    typer.echo("Initialising Ghost-Matrix RL Dashboard...")
    typer.echo("  Tab=toggle manual/AI  WASD/arrows=move  ESC=quit")
    if passive:
        typer.echo("  Mode=passive actor-only (no environment control sockets)")

    scene_path = scene if scene else None
    dashboard_runner = _get_matrix_viewer_class()(
        matrix_sub=m_sub,
        actor_sub=a_sub,
        step_endpoint=s_end,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        max_distance_m=resolved_max_distance,
        scene_path=scene_path,
    )
    dashboard_runner.run()


@app.command()
def explore(
    gmdag_file: str = typer.Option(
        "",
        help="Path to .gmdag world cache. Auto-discovers from corpus if omitted.",
    ),
    pub_address: str = typer.Option(
        "tcp://localhost:5559",
        help="ZMQ PUB address for the environment server.",
    ),
    rep_address: str = typer.Option(
        "tcp://localhost:5560",
        help="ZMQ REP address for step control.",
    ),
    hz: float = typer.Option(30.0, help="Dashboard tick rate"),
    linear_speed: float = typer.Option(1.5, help="Max horizontal linear speed"),
    yaw_rate: float = typer.Option(1.5, help="Max yaw rate"),
    max_distance: float | None = typer.Option(
        None, min=0.01, help="Observation normalization horizon in meters."
    ),
    azimuth_bins: int = typer.Option(256, help="Distance-matrix azimuth bins"),
    elevation_bins: int = typer.Option(48, help="Distance-matrix elevation bins"),
    record: bool = typer.Option(False, help="Enable demonstration recording (auto-starts, B to pause/resume)."),
    drone_max_speed: float = typer.Option(5.0, help="Drone max forward speed for action normalization."),
    drone_climb_rate: float = typer.Option(2.0, help="Drone max climb rate for action normalization."),
    drone_strafe_speed: float = typer.Option(3.0, help="Drone max strafe speed for action normalization."),
    drone_yaw_rate: float = typer.Option(3.0, help="Drone max yaw rate for action normalization."),
    max_steps: int = typer.Option(0, help="Auto-close after N recorded steps (0 = unlimited)."),
) -> None:
    """Launch a standalone manual explorer: environment + dashboard with keyboard navigation.

    Starts the environment server as a subprocess and opens the dashboard
    in manual mode so you can navigate the scene with WASD / arrow keys
    without any training process running.
    """
    import atexit
    import signal

    setup_logging("navi_auditor_explore")
    config = AuditorConfig()

    resolved_max_distance = (
        float(max_distance)
        if max_distance is not None
        else float(config.observation_max_distance_m)
    )

    # Build environment serve command
    env_cmd = [
        *_environment_cli_command(),
        "serve",
        "--mode", "step",
        "--actors", "1",
        "--pub", f"tcp://*:{pub_address.rsplit(':', 1)[-1]}",
        "--rep", f"tcp://*:{rep_address.rsplit(':', 1)[-1]}",
        "--azimuth-bins", str(azimuth_bins),
        "--elevation-bins", str(elevation_bins),
        "--max-distance", str(resolved_max_distance),
    ]
    if gmdag_file:
        env_cmd.extend(["--gmdag-file", gmdag_file])

    typer.echo("Starting environment server for manual exploration...")
    typer.echo(f"  Environment command: {' '.join(env_cmd)}")

    # Launch environment as subprocess — clean env to avoid venv cross-talk
    import os

    env_vars: dict[str, str] = {}
    for k, v in os.environ.items():
        if k.startswith("UV_"):
            continue
        if k.startswith("PYTHON"):
            continue
        if k == "VIRTUAL_ENV":
            continue
        env_vars[k] = v
    if "PATH" in env_vars:
        env_vars["PATH"] = os.pathsep.join(
            p
            for p in env_vars["PATH"].split(os.pathsep)
            if ".venv" not in p and "projects\\auditor" not in p
        )

    _creation_flags: int = (
        subprocess.CREATE_NEW_PROCESS_GROUP
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP")
        else 0
    )
    env_proc = subprocess.Popen(  # noqa: S603
        env_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=_repo_root(),
        env=env_vars,
        creationflags=_creation_flags,
    )

    def _kill_env() -> None:
        if env_proc.poll() is None:
            try:
                if hasattr(signal, "CTRL_BREAK_EVENT"):
                    env_proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    env_proc.terminate()
                env_proc.wait(timeout=5)
            except Exception:  # noqa: S110
                env_proc.kill()

    atexit.register(_kill_env)

    # Wait for the environment to become ready
    typer.echo("  Waiting for environment to start...")
    _ready = False
    for _attempt in range(30):
        if env_proc.poll() is not None:
            stderr_out = (env_proc.stderr.read() or b"").decode("utf-8", errors="replace")
            typer.echo(f"  Environment exited unexpectedly: {stderr_out[:500]}", err=True)
            raise typer.Exit(code=1)

        ctx = zmq.Context()
        probe = ctx.socket(zmq.REQ)
        probe.setsockopt(zmq.RCVTIMEO, 500)
        probe.setsockopt(zmq.SNDTIMEO, 500)
        probe.setsockopt(zmq.LINGER, 0)
        probe.setsockopt(zmq.REQ_RELAXED, 1)
        probe.setsockopt(zmq.REQ_CORRELATE, 1)
        try:
            probe.connect(rep_address)
            # Send a no-op step request to test if the server is alive
            from navi_contracts import Action, StepRequest, serialize

            noop_action = Action(
                env_ids=np.array([0], dtype=np.int32),
                linear_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                angular_velocity=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                policy_id="explore-probe",
                step_id=0,
                timestamp=time.time(),
            )
            probe.send(serialize(StepRequest(
                action=noop_action,
                step_id=0,
                timestamp=time.time(),
            )))
            _reply = probe.recv()
            _ready = True
        except zmq.Again:
            time.sleep(1.0)
            continue
        except Exception:
            time.sleep(1.0)
            continue
        finally:
            probe.close(linger=0)
            ctx.term()
        if _ready:
            break

    if not _ready:
        typer.echo("  Environment did not become ready within 30 seconds.", err=True)
        _kill_env()
        raise typer.Exit(code=1)

    typer.echo("  Environment ready!")
    typer.echo("")
    typer.echo("Ghost-Matrix Explorer")
    typer.echo("  WASD / Arrow keys = navigate")
    typer.echo("  Tab = toggle manual mode")
    typer.echo("  F12 = capture snapshot")
    if record:
        typer.echo("  Recording starts automatically — B toggles pause/resume")
        if max_steps > 0:
            typer.echo(f"  Auto-close after {max_steps} recorded steps")
    typer.echo("  ESC / Q = quit")
    typer.echo("")

    dashboard_runner = _get_matrix_viewer_class()(
        matrix_sub=pub_address,
        actor_sub="",
        step_endpoint=rep_address,
        hz=hz,
        linear_speed=linear_speed,
        yaw_rate=yaw_rate,
        max_distance_m=resolved_max_distance,
        scene_path=gmdag_file or None,
        start_manual=True,
        enable_recording=record,
        drone_max_speed=drone_max_speed,
        drone_climb_rate=drone_climb_rate,
        drone_strafe_speed=drone_strafe_speed,
        drone_yaw_rate=drone_yaw_rate,
        max_steps=max_steps,
    )
    dashboard_runner.run()

    # Dashboard closed — kill environment
    _kill_env()


@app.command("dashboard-attach-check")
def dashboard_attach_check(
    actor_sub: str = typer.Option(None, help="Actor or replay PUB SUB address to observe."),
    timeout_seconds: float = typer.Option(
        15.0, min=0.1, help="Maximum time to wait for actor-visible traffic."
    ),
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


@app.command("dashboard-capture-frame")
def dashboard_capture_frame(
    actor_sub: str = typer.Option(None, help="Actor PUB SUB address to observe."),
    actor_id: int = typer.Option(0, min=0, help="Actor env_id to capture."),
    timeout_seconds: float = typer.Option(
        15.0, min=0.1, help="Maximum time to wait for a matching frame."
    ),
    output_dir: str = typer.Option("", help="Optional output directory override."),
    max_distance: float | None = typer.Option(
        None, min=0.01, help="Observation normalization horizon in meters."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON summary instead of text."),
) -> None:
    """Capture one live dashboard-visible DistanceMatrix frame plus rendered diagnostics."""
    setup_logging("navi_auditor_dashboard_capture_frame")
    config = AuditorConfig()
    resolved_actor_sub = actor_sub if actor_sub is not None else config.actor_sub_address
    summary = _capture_dashboard_frame(
        actor_sub=resolved_actor_sub,
        actor_id=actor_id,
        timeout_seconds=timeout_seconds,
        output_dir=Path(output_dir) if output_dir else None,
        max_distance_m=float(max_distance)
        if max_distance is not None
        else float(config.observation_max_distance_m),
    )

    if json_output:
        typer.echo(json.dumps(summary, indent=2))
    elif summary["ok"]:
        typer.echo(
            "dashboard-capture-frame - "
            f"ok=True, actor_id={summary['actor_id']}, output_dir={summary['output_dir']}, "
            f"valid_ratio={summary['valid_ratio']:.4f}, min_depth={summary['depth_min']:.4f}, max_depth={summary['depth_max']:.4f}"
        )
    else:
        typer.echo(
            "dashboard-capture-frame - "
            f"ok=False, actor_id={summary['actor_id']}, elapsed_seconds={summary['elapsed_seconds']:.3f}",
            err=True,
        )
        for issue in cast("list[str]", summary.get("issues", [])):
            typer.echo(f"issue={issue}", err=True)

    if not summary["ok"]:
        raise typer.Exit(code=1)


def _capture_dashboard_frame(
    *,
    actor_sub: str,
    actor_id: int,
    timeout_seconds: float,
    output_dir: Path | None,
    max_distance_m: float,
) -> dict[str, Any]:
    from navi_contracts import DistanceMatrix, deserialize

    started = time.perf_counter()
    context: zmq.Context[zmq.Socket[bytes]] = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 200)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(actor_sub)
    socket.setsockopt(zmq.SUBSCRIBE, TOPIC_DISTANCE_MATRIX.encode("utf-8"))
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    try:
        deadline = time.perf_counter() + timeout_seconds
        while time.perf_counter() < deadline:
            events = dict(poller.poll(250))
            if socket not in events:
                continue
            topic_bytes, data = socket.recv_multipart()
            if topic_bytes.decode("utf-8") != TOPIC_DISTANCE_MATRIX:
                continue
            msg = deserialize(data)
            if not isinstance(msg, DistanceMatrix):
                continue
            current_actor_id = int(msg.env_ids[0]) if len(msg.env_ids) > 0 else 0
            if current_actor_id != actor_id:
                continue

            run_context = get_or_create_run_context("dashboard-capture")
            artifact_root = output_dir or (
                run_context.run_root / "captures" / f"dashboard-actor{actor_id}"
            )
            artifact_root.mkdir(parents=True, exist_ok=True)

            raw_depth = np.asarray(msg.depth[0], dtype=np.float32)
            raw_valid = np.asarray(msg.valid_mask[0], dtype=bool)
            raw_semantic = np.asarray(msg.semantic[0], dtype=np.int32)

            perspective_img, center_distance = render_first_person(
                raw_depth,
                raw_semantic,
                raw_valid,
                960,
                720,
                pitch=msg.robot_pose.pitch,
            )
            raw_img = depth_to_viridis(raw_depth.T, raw_valid.T)
            raw_img = cv2.resize(raw_img, (960, 240), interpolation=cv2.INTER_NEAREST)

            perspective_path = artifact_root / "perspective.png"
            raw_path = artifact_root / "raw_distance_matrix.png"
            npz_path = artifact_root / "distance_matrix_arrays.npz"
            summary_path = artifact_root / "summary.json"

            cv2.imwrite(str(perspective_path), perspective_img)
            cv2.imwrite(str(raw_path), raw_img)
            np.savez_compressed(
                npz_path,
                depth=raw_depth,
                valid_mask=raw_valid,
                semantic=raw_semantic,
            )

            valid_values = raw_depth[raw_valid]
            summary = {
                "profile": "dashboard-capture-frame",
                "ok": True,
                "run_id": run_context.run_id,
                "actor_sub": actor_sub,
                "actor_id": actor_id,
                "step_id": int(msg.step_id),
                "timestamp": float(msg.timestamp),
                "output_dir": str(artifact_root),
                "perspective_image": str(perspective_path),
                "raw_image": str(raw_path),
                "arrays": str(npz_path),
                "matrix_shape": [int(raw_depth.shape[0]), int(raw_depth.shape[1])],
                "valid_ratio": float(np.mean(raw_valid.astype(np.float32))),
                "depth_min": float(np.min(valid_values)) if valid_values.size > 0 else 0.0,
                "depth_max": float(np.max(valid_values)) if valid_values.size > 0 else 0.0,
                "depth_mean": float(np.mean(valid_values)) if valid_values.size > 0 else 0.0,
                "center_distance_m": float(center_distance),
                "max_distance_m": float(max_distance_m),
                "elapsed_seconds": float(time.perf_counter() - started),
            }
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        return {
            "profile": "dashboard-capture-frame",
            "ok": False,
            "actor_sub": actor_sub,
            "actor_id": actor_id,
            "elapsed_seconds": float(time.perf_counter() - started),
            "issues": ["No matching distance_matrix_v2 frame arrived before timeout."],
        }
    finally:
        socket.close()
        context.term()


@app.command("dataset-audit")
def dataset_audit(
    gmdag_file: str = typer.Option(
        "",
        help="Optional .gmdag asset override. When omitted, use the first promoted corpus asset.",
    ),
    expected_resolution: int = typer.Option(512, help="Expected canonical compiled resolution."),
    benchmark: bool = typer.Option(
        True, help="Run a real-runtime benchmark after runtime preflight succeeds."
    ),
    actors: int = typer.Option(1, min=1, help="Number of actors for the benchmark batch."),
    steps: int = typer.Option(8, min=1, help="Measured benchmark batch steps."),
    warmup_steps: int = typer.Option(1, min=0, help="Unmeasured benchmark warmup steps."),
    azimuth_bins: int = typer.Option(64, min=1, help="Benchmark azimuth bins."),
    elevation_bins: int = typer.Option(16, min=1, help="Benchmark elevation bins."),
    max_distance: float = typer.Option(
        30.0, min=0.01, help="Benchmark distance normalization range."
    ),
    sdf_max_steps: int = typer.Option(
        256, min=1, help="Benchmark maximum sphere-tracing iterations per ray."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit a JSON summary instead of text."),
) -> None:
    """Run the first passive runtime-backed dataset QA surface through the environment CLI."""
    setup_logging("navi_auditor_dataset_audit")

    check_arguments = ["check-sdfdag", "--expected-resolution", str(expected_resolution)]
    if gmdag_file:
        check_arguments.extend(["--gmdag-file", gmdag_file])

    check_returncode, check_payload = _run_environment_json_command(
        "check-sdfdag", check_arguments
    )
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
        benchmark_returncode, benchmark_payload = _run_environment_json_command(
            "bench-sdfdag", benchmark_arguments
        )
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


def explore_shortcut() -> None:
    """Shortcut for 'navi-auditor explore' command."""
    import sys

    app(["explore", *sys.argv[1:]])


if __name__ == "__main__":
    app()
