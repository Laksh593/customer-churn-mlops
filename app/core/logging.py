"""
app/core/logging.py
───────────────────
Reusable logging configuration for the application.

Provides a single ``get_logger`` factory so every module obtains a logger
with consistent formatting without repeating boilerplate.  The root logger
is configured once and all child loggers inherit the handler/level.

Design decisions:
- Standard-library ``logging`` only — no third-party dependency.
- JSON-style output is intentionally deferred to a later milestone when we
  add proper log aggregation (e.g. via Loki / Cloud Logging).  Plain text
  is friendlier during local development.
- Log level defaults to INFO but can be overridden via the LOG_LEVEL env var
  (wired through ``Settings``).
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger for the application.

    Call this **once** at application startup (e.g. in ``main.py`` or the
    FastAPI lifespan).  Subsequent calls are safe but have no effect because
    the handler is added only when none exists yet.

    Args:
        level: A standard logging level string such as ``"DEBUG"``,
               ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or ``"CRITICAL"``.
    """
    root = logging.getLogger()

    # Avoid adding duplicate handlers when the function is called more than once.
    if root.handlers:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Args:
        name: Typically ``__name__`` of the calling module so that the
              logger hierarchy mirrors the package structure.

    Returns:
        A :class:`logging.Logger` instance.

    Example::

        from app.core.logging import get_logger

        logger = get_logger(__name__)
        logger.info("Application started")
    """
    return logging.getLogger(name)
