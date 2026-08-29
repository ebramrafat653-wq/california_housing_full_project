# src/utils/logger.py

"""
Centralized logging configuration for the application.

Provides a single point of control for logging setup across all modules,
with support for console output, optional file logging, and safe reconfiguration.
"""

import logging
import sys
import threading
import warnings
from pathlib import Path
from typing import Optional

_logger_configured: bool = False
_configure_lock: threading.Lock = threading.Lock()


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configure the root logger for the application.

    This function should be called once at application startup, typically
    in main.py or the notebook initialization cell. Subsequent calls have
    no effect unless `force=True`.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to write logs to. Parent directories are
                created automatically if they do not exist.
        force: If True, reconfigure logging even if already initialized.
            Useful for testing or dynamic reload scenarios.

    Example:
        >>> from src.utils.logger import setup_logging
        >>> setup_logging(level=logging.DEBUG, log_file="logs/app.log")
    """
    global _logger_configured

    with _configure_lock:
        if _logger_configured and not force:
            return

        if force:
            logging.root.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handlers: list[logging.Handler] = []

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)

        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True,
        )

        _logger_configured = True
        logging.getLogger(__name__).debug("Logging system initialized.")


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger instance.

    If logging has not been configured via setup_logging(), this function
    will auto-initialize with default settings and emit a warning. For
    production deployments, always call setup_logging() explicitly at startup.

    Args:
        name: Logger name, typically __name__ from the calling module.

    Returns:
        Configured logging.Logger instance.

    Warning:
        Auto-initialization uses default settings (INFO level, console only).
        Call setup_logging() explicitly to customize log level or file output.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    if not _logger_configured:
        warnings.warn(
            "Logger auto-initialized with defaults. "
            "Call setup_logging() explicitly at startup for full control.",
            RuntimeWarning,
            stacklevel=2,
        )
        setup_logging()

    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
