"""Unified logging architecture for the Navi Ghost-Matrix system."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__: list[str] = ["setup_logging"]


def setup_logging(project_name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> None:
    """Initialize a standardized, high-quality, cyclic logging architecture.

    Args:
        project_name: The name of the project (e.g., "navi_actor").
        log_dir: Directory to store log files.
        level: Base logging level.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_file = log_dir_path / f"{project_name}.log"

    # Professional format: Timestamp, aligned level, module:line, message
    log_format = "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)

    # 1. Cyclic File Handler (1MB max, 10 backups)
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=1 * 1024 * 1024,  # 1 MB
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # 2. Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to prevent duplicate logs if called multiple times
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Silence noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    
    # Announce
    logging.getLogger(__name__).info("Unified logging initialized for %s.", project_name)
