"""
GUARDIAN ML — Logging Utility
Centralized structured logging via loguru with rich console output.
"""

import sys
import logging
from pathlib import Path
from loguru import logger as _loguru_logger


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""

    def emit(self, record: logging.LogRecord):
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logger(name: str, config: dict = None) -> _loguru_logger.__class__:
    """
    Configure and return a loguru logger instance.

    Args:
        name:   Module/component name for context labeling.
        config: Optional logging config dict from config.yaml.

    Returns:
        Configured loguru logger.
    """
    config = config or {}
    level   = config.get("level", "INFO")
    fmt     = config.get(
        "format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan> | "
        "<level>{message}</level>",
    )
    log_file = config.get("file", "logs/guardian.log")
    rotation = config.get("rotation", "10 MB")
    retention = config.get("retention", "30 days")

    # Remove default loguru handler
    _loguru_logger.remove()

    # Console handler (rich color)
    _loguru_logger.add(
        sys.stdout,
        format=fmt,
        level=level,
        colorize=True,
        enqueue=True,
    )

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _loguru_logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True,
    )

    # Intercept standard library logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for lib in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(lib).handlers = [InterceptHandler()]

    return _loguru_logger.bind(name=name)


# Convenience module-level logger
logger = setup_logger("guardian.utils")
