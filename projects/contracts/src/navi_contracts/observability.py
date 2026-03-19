"""Shared run-context and metrics helpers for Navi operational surfaces."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID_ENV = "NAVI_RUN_ID"
_RUN_ROOT_ENV = "NAVI_RUN_ROOT"
_LOG_ROOT_ENV = "NAVI_LOG_ROOT"
_METRICS_ROOT_ENV = "NAVI_METRICS_ROOT"
_MANIFEST_ROOT_ENV = "NAVI_MANIFEST_ROOT"
_REPO_ROOT_ENV = "NAVI_REPO_ROOT"
_STARTED_AT_ENV = "NAVI_RUN_STARTED_AT"

_RUN_CONTEXT_LOCK = threading.Lock()


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _sanitize_profile(profile: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in profile.strip())
    normalized = "-".join(part for part in normalized.split("-") if part)
    return normalized or "run"


def _resolve_repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()

    env_repo_root = os.getenv(_REPO_ROOT_ENV)
    if env_repo_root:
        return Path(env_repo_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "AGENTS.md").exists():
            return parent

    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        if (parent / "AGENTS.md").exists():
            return parent

    return cwd


def generate_run_id(profile: str, *, now: datetime | None = None, pid: int | None = None) -> str:
    timestamp = (now or _now_utc()).strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_profile(profile)}-{timestamp}-p{pid or os.getpid()}"


@dataclass(frozen=True)
class RunContext:
    run_id: str
    profile: str
    started_at: str
    repo_root: Path
    run_root: Path
    log_root: Path
    metrics_root: Path
    manifest_root: Path

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: str(value) if isinstance(value, Path) else value for key, value in payload.items()
        }


def get_or_create_run_context(
    profile: str,
    *,
    repo_root: str | Path | None = None,
    base_relative_root: str | Path = Path("artifacts") / "runs",
) -> RunContext:
    with _RUN_CONTEXT_LOCK:
        resolved_repo_root = _resolve_repo_root(repo_root)
        profile_name = _sanitize_profile(profile)
        run_id = os.getenv(_RUN_ID_ENV) or generate_run_id(profile_name)
        started_at = os.getenv(_STARTED_AT_ENV) or _now_utc().isoformat(timespec="seconds")

        env_run_root = os.getenv(_RUN_ROOT_ENV)
        run_root = (
            Path(env_run_root).resolve()
            if env_run_root
            else (resolved_repo_root / base_relative_root / run_id).resolve()
        )
        log_root = Path(os.getenv(_LOG_ROOT_ENV, run_root / "logs")).resolve()
        metrics_root = Path(os.getenv(_METRICS_ROOT_ENV, run_root / "metrics")).resolve()
        manifest_root = Path(os.getenv(_MANIFEST_ROOT_ENV, run_root / "manifests")).resolve()

        for directory in (run_root, log_root, metrics_root, manifest_root):
            directory.mkdir(parents=True, exist_ok=True)

        os.environ[_RUN_ID_ENV] = run_id
        os.environ[_RUN_ROOT_ENV] = str(run_root)
        os.environ[_LOG_ROOT_ENV] = str(log_root)
        os.environ[_METRICS_ROOT_ENV] = str(metrics_root)
        os.environ[_MANIFEST_ROOT_ENV] = str(manifest_root)
        os.environ[_REPO_ROOT_ENV] = str(resolved_repo_root)
        os.environ[_STARTED_AT_ENV] = started_at

        return RunContext(
            run_id=run_id,
            profile=profile_name,
            started_at=started_at,
            repo_root=resolved_repo_root,
            run_root=run_root,
            log_root=log_root,
            metrics_root=metrics_root,
            manifest_root=manifest_root,
        )


def get_run_id(default: str = "unknown") -> str:
    return os.getenv(_RUN_ID_ENV, default)


def _bytes_to_mebibytes(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value) / (1024.0 * 1024.0)
    return None


def collect_process_resource_snapshot(*, cuda_device: Any | None = None) -> dict[str, Any]:
    """Collect a coarse process and CUDA resource snapshot.

    The snapshot is designed for run-scoped metrics emission at operation
    boundaries and coarse heartbeat cadence only. It intentionally avoids any
    forced synchronization or repeated external polling inside hot per-step
    loops.
    """

    snapshot: dict[str, Any] = {
        "sampled_at": _now_utc().isoformat(timespec="milliseconds"),
        "pid": os.getpid(),
        "cuda_available": False,
    }

    try:
        psutil_mod = importlib.import_module("psutil")
    except ImportError:
        psutil_mod = None

    if psutil_mod is not None:
        try:
            process = psutil_mod.Process(os.getpid())
            with process.oneshot():
                memory_info = process.memory_info()
                snapshot["proc_rss_mb"] = _bytes_to_mebibytes(getattr(memory_info, "rss", None))
                snapshot["proc_vms_mb"] = _bytes_to_mebibytes(getattr(memory_info, "vms", None))
                snapshot["proc_cpu_percent"] = float(process.cpu_percent(interval=None))
                snapshot["proc_threads"] = int(process.num_threads())
                num_handles = getattr(process, "num_handles", None)
                if callable(num_handles):
                    snapshot["proc_handles"] = int(num_handles())
        except Exception as exc:  # pragma: no cover - best-effort diagnostics
            snapshot["proc_snapshot_error"] = str(exc)

    try:
        torch_mod = importlib.import_module("torch")
    except ImportError:
        torch_mod = None

    if torch_mod is None:
        return snapshot

    try:
        cuda_available = bool(torch_mod.cuda.is_available())
    except Exception as exc:  # pragma: no cover - defensive runtime path
        snapshot["cuda_snapshot_error"] = str(exc)
        return snapshot

    snapshot["cuda_available"] = cuda_available
    if not cuda_available:
        return snapshot

    try:
        device = torch_mod.device(cuda_device) if cuda_device is not None else torch_mod.device("cuda")
        device_index = device.index
        if device_index is None:
            device_index = int(torch_mod.cuda.current_device())
        free_bytes, total_bytes = torch_mod.cuda.mem_get_info(device_index)
        snapshot.update(
            {
                "cuda_device_index": int(device_index),
                "cuda_device_name": str(torch_mod.cuda.get_device_name(device_index)),
                "cuda_allocated_mb": _bytes_to_mebibytes(
                    torch_mod.cuda.memory_allocated(device_index)
                ),
                "cuda_reserved_mb": _bytes_to_mebibytes(
                    torch_mod.cuda.memory_reserved(device_index)
                ),
                "cuda_max_allocated_mb": _bytes_to_mebibytes(
                    torch_mod.cuda.max_memory_allocated(device_index)
                ),
                "cuda_max_reserved_mb": _bytes_to_mebibytes(
                    torch_mod.cuda.max_memory_reserved(device_index)
                ),
                "cuda_free_mb": _bytes_to_mebibytes(free_bytes),
                "cuda_total_mb": _bytes_to_mebibytes(total_bytes),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        snapshot["cuda_snapshot_error"] = str(exc)

    return snapshot


def build_phase_metrics_payload(
    operation: str,
    *,
    started_at: float | None = None,
    elapsed_ms: float | None = None,
    step_id: int | None = None,
    cuda_device: Any | None = None,
    include_resources: bool = True,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one append-only phase metrics payload with an attached resource snapshot."""

    resolved_elapsed_ms = elapsed_ms
    if resolved_elapsed_ms is None:
        if started_at is None:
            raise ValueError("build_phase_metrics_payload requires started_at or elapsed_ms")
        resolved_elapsed_ms = (time.perf_counter() - started_at) * 1000.0

    payload: dict[str, Any] = {
        "operation": operation,
        "elapsed_ms": float(resolved_elapsed_ms),
    }
    if step_id is not None:
        payload["step_id"] = int(step_id)
    if metadata:
        payload.update(dict(metadata))
    if include_resources:
        payload.update(collect_process_resource_snapshot(cuda_device=cuda_device))
    return payload


def write_process_manifest(
    project_name: str,
    *,
    context: RunContext,
    metadata: Mapping[str, Any] | None = None,
    file_name: str | None = None,
) -> Path:
    manifest_path = context.manifest_root / (file_name or f"{project_name}.json")
    payload: dict[str, Any] = {
        "project_name": project_name,
        "run": context.to_manifest_dict(),
        "process": {
            "pid": os.getpid(),
            "cwd": str(Path.cwd()),
            "python_executable": sys.executable,
            "argv": sys.argv,
            "platform": platform.platform(),
        },
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


class JsonlMetricsSink:
    """Append-only JSONL sink for machine-readable review artifacts."""

    def __init__(self, path: Path, *, run_id: str, project_name: str) -> None:
        self._path = path
        self._run_id = run_id
        self._project_name = project_name
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def emit(self, stream: str, payload: Mapping[str, Any]) -> None:
        record = {
            "timestamp": _now_utc().isoformat(timespec="milliseconds"),
            "run_id": self._run_id,
            "project_name": self._project_name,
            "stream": stream,
            "payload": dict(payload),
        }
        encoded = json.dumps(record, sort_keys=True)
        with self._lock:
            self._handle.write(encoded)
            self._handle.write("\n")
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()
