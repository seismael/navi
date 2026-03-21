from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _bootstrap_repo_imports() -> None:
    repo_root = _repo_root()
    source_roots = [
        repo_root / "projects" / "contracts" / "src",
        repo_root / "projects" / "environment" / "src",
        repo_root / "projects" / "auditor" / "src",
        repo_root / "projects" / "actor" / "src",
    ]
    for source_root in reversed(source_roots):
        source_text = str(source_root)
        if source_root.exists() and source_text not in sys.path:
            sys.path.insert(0, source_text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run machine-readable Navi surfaces without shell CLI nesting.")
    parser.add_argument("--module", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--error-path", default="")
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _strip_separator(argv: list[str]) -> list[str]:
    if argv and argv[0] == "--":
        return argv[1:]
    return argv


def _emit(payload: dict[str, Any], exit_code: int) -> int:
    sys.stdout.write(json.dumps(payload, indent=2))
    return exit_code


def _emit_to_paths(payload: dict[str, Any], exit_code: int, output_path: str, error_path: str) -> int:
    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        sys.stdout.write(json.dumps(payload, indent=2))

    if error_path:
        error_text = ""
        if exit_code != 0:
            issues = payload.get("issues")
            if isinstance(issues, list) and issues:
                error_text = "\n".join(str(issue) for issue in issues)
            else:
                error_text = str(payload.get("error", ""))
        Path(error_path).write_text(error_text, encoding="utf-8")
    return exit_code


def _parse_flag_value(argv: list[str], name: str, default: str = "") -> str:
    if name not in argv:
        return default
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise RuntimeError(f"missing value for {name}")
    return argv[index + 1]


def _parse_int_flag(argv: list[str], name: str, default: int) -> int:
    value = _parse_flag_value(argv, name, "")
    return default if value == "" else int(value)


def _parse_float_flag(argv: list[str], name: str, default: float) -> float:
    value = _parse_flag_value(argv, name, "")
    return default if value == "" else float(value)


def _has_flag(argv: list[str], name: str) -> bool:
    return name in argv


def _run_environment_check(argv: list[str]) -> tuple[int, dict[str, Any]]:
    from navi_contracts import setup_logging
    from navi_environment.integration.corpus import validate_compiled_scene_corpus
    from navi_environment.integration.voxel_dag import probe_sdfdag_runtime

    setup_logging("navi_environment_sdfdag_check")
    gmdag_file = _parse_flag_value(argv, "--gmdag-file", "")
    gmdag_root = _parse_flag_value(argv, "--gmdag-root", "")
    manifest = _parse_flag_value(argv, "--manifest", "")
    expected_resolution = _parse_int_flag(argv, "--expected-resolution", 512)

    status = probe_sdfdag_runtime(Path(gmdag_file) if gmdag_file else None)
    corpus_validation = None
    if not gmdag_file:
        corpus_validation = validate_compiled_scene_corpus(
            Path(gmdag_root) if gmdag_root else None,
            manifest_path=Path(manifest) if manifest else None,
            expected_resolution=expected_resolution,
        )

    issues = list(status.issues)
    if corpus_validation is not None:
        issues.extend(corpus_validation.issues)

    payload = {
        "profile": "check-sdfdag",
        "gmdag_file": str(Path(gmdag_file).expanduser()) if gmdag_file else "",
        "expected_resolution": expected_resolution,
        "runtime": {
            "compiler_ready": status.compiler_ready,
            "torch_ready": status.torch_ready,
            "cuda_ready": status.cuda_ready,
            "torch_sdf_ready": status.torch_sdf_ready,
            "asset_loaded": status.asset_loaded,
            "compiler_path": str(status.compiler_path) if status.compiler_path is not None else None,
            "gmdag_path": str(status.gmdag_path) if status.gmdag_path is not None else None,
            "resolution": status.resolution,
            "node_count": status.node_count,
            "bbox_min": list(status.bbox_min) if status.bbox_min is not None else None,
            "bbox_max": list(status.bbox_max) if status.bbox_max is not None else None,
        },
        "corpus": None
        if corpus_validation is None
        else {
            "root": str(corpus_validation.gmdag_root),
            "manifest": str(corpus_validation.manifest_path),
            "manifest_present": corpus_validation.manifest_present,
            "scene_count": corpus_validation.scene_count,
            "compiled_resolutions": list(corpus_validation.compiled_resolutions),
        },
        "issues": issues,
        "ok": not issues,
    }
    return (0 if not issues else 1, payload)


def _run_environment_bench(argv: list[str]) -> tuple[int, dict[str, Any]]:
    from navi_contracts import setup_logging
    from navi_environment.cli import _aggregate_bench_runs, _run_bench_iteration
    from navi_environment.config import EnvironmentConfig

    setup_logging("navi_environment_sdfdag_bench")
    gmdag_file = _parse_flag_value(argv, "--gmdag-file")
    actors = _parse_int_flag(argv, "--actors", 4)
    steps = _parse_int_flag(argv, "--steps", 200)
    warmup_steps = _parse_int_flag(argv, "--warmup-steps", 25)
    repeats = _parse_int_flag(argv, "--repeats", 1)
    azimuth_bins = _parse_int_flag(argv, "--azimuth-bins", 256)
    elevation_bins = _parse_int_flag(argv, "--elevation-bins", 48)
    max_distance = _parse_float_flag(argv, "--max-distance", 30.0)
    sdf_max_steps = _parse_int_flag(argv, "--sdf-max-steps", 256)

    default_config = EnvironmentConfig()
    config = EnvironmentConfig(
        pub_address=default_config.pub_address,
        rep_address=default_config.rep_address,
        action_sub_address=default_config.action_sub_address,
        mode="step",
        training_mode=True,
        backend="sdfdag",
        gmdag_file=gmdag_file,
        sdf_max_steps=sdf_max_steps,
        azimuth_bins=azimuth_bins,
        elevation_bins=elevation_bins,
        max_distance=max_distance,
        n_actors=actors,
    )
    runs = [
        _run_bench_iteration(
            config=config,
            actors=actors,
            steps=steps,
            warmup_steps=warmup_steps,
        )
        for _ in range(repeats)
    ]
    aggregated = _aggregate_bench_runs(runs)
    payload = {
        "profile": "bench-sdfdag",
        "gmdag_file": str(Path(gmdag_file).expanduser()),
        "actors": actors,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "repeats": repeats,
        "azimuth_bins": azimuth_bins,
        "elevation_bins": elevation_bins,
        "max_distance": max_distance,
        "sdf_max_steps": sdf_max_steps,
        **aggregated,
    }
    return (0, payload)


def _resolve_benchmark_gmdag_file(gmdag_file: str, check_payload: dict[str, Any]) -> str:
    if gmdag_file:
        return str(Path(gmdag_file).expanduser().resolve())

    corpus_payload = check_payload.get("corpus")
    if not isinstance(corpus_payload, dict):
        raise RuntimeError("dataset-audit benchmark requires --gmdag-file or a valid promoted corpus manifest")

    manifest_value = corpus_payload.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise RuntimeError("dataset-audit benchmark could not resolve the promoted corpus manifest")

    manifest_path = Path(manifest_value).expanduser().resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    scenes = manifest_payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError(f"Promoted corpus manifest {manifest_path} does not contain any compiled scenes")

    first_scene = scenes[0]
    if not isinstance(first_scene, dict):
        raise RuntimeError(f"Promoted corpus manifest {manifest_path} has an invalid scene entry")

    gmdag_path_value = first_scene.get("gmdag_path")
    if not isinstance(gmdag_path_value, str) or not gmdag_path_value:
        raise RuntimeError(f"Promoted corpus manifest {manifest_path} is missing a scene gmdag_path")

    gmdag_path = Path(gmdag_path_value).expanduser()
    if not gmdag_path.is_absolute():
        gmdag_path = (manifest_path.parent / gmdag_path).resolve()
    else:
        gmdag_path = gmdag_path.resolve()
    return str(gmdag_path)


def _environment_python() -> str:
    """Return the path to the environment project's venv Python."""
    repo_root = _repo_root()
    import sys
    if sys.platform == "win32":
        candidate = repo_root / "projects" / "environment" / ".venv" / "Scripts" / "python.exe"
    else:
        candidate = repo_root / "projects" / "environment" / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(f"Environment venv Python not found: {candidate}")


def _run_environment_subprocess(command: str, argv: list[str]) -> tuple[int, dict[str, Any]]:
    """Run an environment CLI command via subprocess and parse JSON output."""
    import subprocess

    env_python = _environment_python()
    full_args = [env_python, "-m", "navi_environment.cli", command, *argv, "--json"]
    completed = subprocess.run(  # noqa: S603
        full_args,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(_repo_root()),
    )
    stdout = completed.stdout.strip()
    if not stdout:
        msg = f"Environment CLI {command} produced no JSON output"
        stderr = completed.stderr.strip()
        if stderr:
            msg = f"{msg}: {stderr}"
        raise RuntimeError(msg)
    root_start = stdout.find("{")
    if root_start < 0:
        raise RuntimeError(f"Environment CLI {command} did not emit JSON: {stdout[:200]}")
    payload = json.loads(stdout[root_start:])
    if not isinstance(payload, dict):
        raise RuntimeError(f"Environment CLI {command} emitted {type(payload).__name__}, expected dict")
    return completed.returncode, payload


def _run_dataset_audit(argv: list[str]) -> tuple[int, dict[str, Any]]:
    from navi_contracts import setup_logging

    setup_logging("navi_auditor_dataset_audit")
    gmdag_file = _parse_flag_value(argv, "--gmdag-file", "")
    expected_resolution = _parse_int_flag(argv, "--expected-resolution", 512)
    benchmark = not (_has_flag(argv, "--benchmark") and _parse_flag_value(argv, "--benchmark", "True").lower() == "false")
    actors = _parse_int_flag(argv, "--actors", 1)
    steps = _parse_int_flag(argv, "--steps", 8)
    warmup_steps = _parse_int_flag(argv, "--warmup-steps", 1)
    azimuth_bins = _parse_int_flag(argv, "--azimuth-bins", 64)
    elevation_bins = _parse_int_flag(argv, "--elevation-bins", 16)
    max_distance = _parse_float_flag(argv, "--max-distance", 30.0)
    sdf_max_steps = _parse_int_flag(argv, "--sdf-max-steps", 256)

    check_returncode, check_payload = _run_environment_subprocess(
        "check-sdfdag",
        [
            "--expected-resolution",
            str(expected_resolution),
            *(["--gmdag-file", gmdag_file] if gmdag_file else []),
        ],
    )
    issues = list(check_payload.get("issues", []))

    benchmark_payload: dict[str, Any] | None = None
    benchmark_ok = not benchmark
    if benchmark and check_returncode == 0:
        resolved_gmdag_file = _resolve_benchmark_gmdag_file(gmdag_file, check_payload)
        benchmark_returncode, benchmark_payload = _run_environment_subprocess(
            "bench-sdfdag",
            [
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
            ],
        )
        benchmark_ok = benchmark_returncode == 0
        if not benchmark_ok and benchmark_payload is not None:
            issues.extend(benchmark_payload.get("issues", []))

    payload = {
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
        payload["benchmark"] = {
            **benchmark_payload,
            "ok": benchmark_ok,
        }
    return (0 if payload["ok"] else 1, payload)


def _run_dashboard_attach(argv: list[str]) -> tuple[int, dict[str, Any]]:
    from navi_auditor.cli import _wait_for_dashboard_attach
    from navi_auditor.config import AuditorConfig
    from navi_contracts import setup_logging

    setup_logging("navi_auditor_dashboard_attach_check")
    actor_sub = _parse_flag_value(argv, "--actor-sub", "")
    timeout_seconds = _parse_float_flag(argv, "--timeout-seconds", 15.0)
    config = AuditorConfig()
    resolved_actor_sub = actor_sub if actor_sub else config.actor_sub_address
    payload = _wait_for_dashboard_attach(resolved_actor_sub, timeout_seconds)
    return (0 if payload.get("ok", False) else 1, payload)


def _dispatch(module: str, argv: list[str]) -> tuple[int, dict[str, Any]]:
    if not argv:
        raise RuntimeError("missing subcommand argv")

    command = argv[0]
    if module == "navi_environment.cli":
        if command == "check-sdfdag":
            return _run_environment_check(argv)
        if command == "bench-sdfdag":
            return _run_environment_bench(argv)
    elif module == "navi_auditor.cli":
        if command == "dataset-audit":
            return _run_dataset_audit(argv)
        if command == "dashboard-attach-check":
            return _run_dashboard_attach(argv)

    raise RuntimeError(f"unsupported machine-readable surface: {module} {' '.join(argv)}")


def main() -> int:
    _bootstrap_repo_imports()
    args = _parse_args()
    argv = _strip_separator(list(args.argv))
    if argv and argv[-1] == "--json":
        argv = argv[:-1]

    try:
        exit_code, payload = _dispatch(args.module, argv)
    except Exception as exc:
        return _emit_to_paths({"ok": False, "issues": [str(exc)], "error": str(exc)}, 1, args.output_path, args.error_path)
    return _emit_to_paths(payload, exit_code, args.output_path, args.error_path)


if __name__ == "__main__":
    raise SystemExit(main())