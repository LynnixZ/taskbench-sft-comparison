"""Centralised logging configuration.

Per the implementation principles we use the :mod:`logging` module everywhere
instead of bare ``print`` calls, so that every exclusion / repair / truncation /
parse failure is recorded with a consistent format and can be redirected to a
file per run.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
) -> None:
    """Configure the root logger once.

    Args:
        level: Logging level for the root logger.
        log_file: Optional path; if given, logs are also written there.
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(level)

    if not _CONFIGURED:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(stream_handler)
        _CONFIGURED = True

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        # Avoid attaching the same file handler twice.
        existing = {
            getattr(h, "baseFilename", None) for h in root.handlers
        }
        if str(log_file.resolve()) not in existing:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
            root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging lazily on first use."""
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
