from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a command and capture stdout/stderr to files.")
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--stdout-path", required=True)
    parser.add_argument("--stderr-path", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _normalize_command(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("missing command to execute")
    return command


def _status_path_for(stdout_path: Path) -> Path:
    return stdout_path.with_name(stdout_path.name + ".launcher-status.json")


def _write_status(status_path: Path, **fields: object) -> None:
    status_path.write_text(json.dumps(fields, separators=(",", ":")), encoding="utf-8")


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(process.pid, 9)
    except OSError:
        process.kill()


def main() -> int:
    args = _parse_args()
    result: dict[str, object] = {
        "launcher_ok": False,
        "exit_code": None,
        "timed_out": False,
        "stdout_path": args.stdout_path,
        "stderr_path": args.stderr_path,
        "error": None,
    }

    try:
        command = _normalize_command(list(args.command))
        stdout_path = Path(args.stdout_path).resolve()
        stderr_path = Path(args.stderr_path).resolve()
        status_path = _status_path_for(stdout_path)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        _write_status(status_path, phase="prepared", command=command, cwd=args.cwd, timeout_seconds=args.timeout_seconds)

        popen_kwargs: dict[str, object] = {
            "cwd": args.cwd,
            "stdout": None,
            "stderr": None,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            popen_kwargs["start_new_session"] = True

        with stdout_path.open("w", encoding="utf-8", newline="") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", newline=""
        ) as stderr_file:
            popen_kwargs["stdout"] = stdout_file
            popen_kwargs["stderr"] = stderr_file
            process = subprocess.Popen(command, **popen_kwargs)
            _write_status(status_path, phase="started", pid=process.pid, timeout_seconds=args.timeout_seconds)
            if args.timeout_seconds <= 0:
                while True:
                    exit_code = process.poll()
                    if exit_code is not None:
                        break
                    time.sleep(0.5)
            else:
                deadline = time.monotonic() + args.timeout_seconds
                exit_code = process.poll()
                _write_status(status_path, phase="polling", pid=process.pid, timeout_seconds=args.timeout_seconds)
                while exit_code is None and time.monotonic() < deadline:
                    time.sleep(0.5)
                    exit_code = process.poll()
                if exit_code is None:
                    result["timed_out"] = True
                    _write_status(status_path, phase="timed_out", pid=process.pid, timeout_seconds=args.timeout_seconds)
                    _kill_process_tree(process)
                    kill_deadline = time.monotonic() + 5.0
                    exit_code = process.poll()
                    while exit_code is None and time.monotonic() < kill_deadline:
                        time.sleep(0.2)
                        exit_code = process.poll()
            _write_status(status_path, phase="completed", pid=process.pid, exit_code=exit_code, timed_out=result["timed_out"])

        result["launcher_ok"] = True
        result["exit_code"] = exit_code
    except Exception as exc:
        try:
            _write_status(_status_path_for(Path(args.stdout_path)), phase="error", error=str(exc))
        except Exception:
            pass
        result["error"] = str(exc)
        sys.stdout.write(json.dumps(result, separators=(",", ":")))
        return 1

    sys.stdout.write(json.dumps(result, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())