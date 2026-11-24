"""Structured logging through structlog, on top of the standard library.

The package never configures logging itself. Applications call `configure`, or set up
structlog and `logging` their own way. Until then, `get_logger` loggers show INFO and above.
Times are UTC.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import Processor

DEFAULT_LEVEL = logging.INFO


def level_number(level: str | int) -> int:
    """A `logging` level from its number or its name in any case."""
    if isinstance(level, int):
        return level
    try:
        return logging.getLevelNamesMapping()[level.upper()]
    except KeyError:
        raise ValueError(
            f"unknown log level {level!r}, expected debug, info, warning, error or critical"
        ) from None


def processors(json: bool = False) -> list[Processor]:
    """Level, UTC time and either readable colored lines or one JSON object per line."""
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(), pad_event_to=0, sort_keys=False
        )
    )
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso" if json else "%H:%M:%SZ", utc=True),
        renderer,
    ]


def stderr_logger(*_: Any) -> structlog.PrintLogger:
    """Looks up stderr on every call, so a replaced stream (e.g. pytest capture) is honored."""
    return structlog.PrintLogger(sys.stderr)


def configure(level: str | int = "INFO", json: bool = False) -> None:
    """Log to stderr from `level` (a name in any case, or a number) up."""
    structlog.configure(
        processors=processors(json),
        wrapper_class=structlog.make_filtering_bound_logger(level_number(level)),
        logger_factory=stderr_logger,
        cache_logger_on_first_use=False,
    )
