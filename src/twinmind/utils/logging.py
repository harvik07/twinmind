"""Consistent logging across scripts, the twin and the agent layer."""

from __future__ import annotations

import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured: set[str] = set()


def _level_from_env() -> int:
    raw = os.getenv("TWINMIND_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, raw, logging.INFO)
    return level if isinstance(level, int) else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to stderr at $TWINMIND_LOG_LEVEL (default INFO)."""
    logger = logging.getLogger(name)
    logger.setLevel(_level_from_env())
    if name not in _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False
        _configured.add(name)
    return logger
