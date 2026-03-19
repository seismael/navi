"""Unified logging architecture for the Navi Ghost-Matrix system."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from navi_contracts.observability import (
    get_or_create_run_context,
    get_run_id,
    write_process_manifest,
)

__all__: list[str] = ["setup_logging"]


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = get_run_id()
        return True


def _build_rotating_handler(
    *,
    file_path: Path,
    level: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    max_bytes = int(os.getenv("NAVI_LOG_MAX_BYTES", str(1 * 1024 * 1024)))
    backup_count = int(os.getenv("NAVI_LOG_BACKUP_COUNT", "10"))
    handler = RotatingFileHandler(
        filename=file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    handler.setLevel(level)
    handler.addFilter(_RunContextFilter())
    return handler


def setup_logging(
    project_name: str, log_dir: str | Path = "logs", level: int = logging.INFO
) -> None:
    """Initialize a standardized, high-quality, cyclic logging architecture.

    Args:
        project_name: The name of the project (e.g., "navi_actor").
        log_dir: Directory to store log files.
        level: Base logging level.
    """
    run_context = get_or_create_run_context(project_name)
    stable_log_dir = (
        (run_context.repo_root / log_dir).resolve()
        if not Path(log_dir).is_absolute()
        else Path(log_dir).resolve()
    )
    stable_log_dir.mkdir(parents=True, exist_ok=True)
    run_context.log_root.mkdir(parents=True, exist_ok=True)

    stable_log_file = stable_log_dir / f"{project_name}.log"
    run_log_file = run_context.log_root / f"{project_name}.log"

    # Professional format: Timestamp, aligned level, module:line, message
    log_format = (
        "[%(asctime)s] [%(levelname)-8s] [run=%(run_id)s] [%(name)s:%(lineno)d] - %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    file_handlers: list[RotatingFileHandler] = [
        _build_rotating_handler(file_path=stable_log_file, level=level, formatter=formatter),
    ]
    if run_log_file != stable_log_file:
        file_handlers.append(
            _build_rotating_handler(file_path=run_log_file, level=level, formatter=formatter)
        )

    # 2. Console Handler
    # Keep machine-readable command output on stdout by sending logs to stderr.
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    console_handler.addFilter(_RunContextFilter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to prevent duplicate logs if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    for file_handler in file_handlers:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    write_process_manifest(project_name, context=run_context)

    # Announce
    logging.getLogger(__name__).info(
        "Unified logging initialized for %s (run_root=%s, run_log=%s).",
        project_name,
        run_context.run_root,
        run_log_file,
    )
