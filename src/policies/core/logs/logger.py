"""Library loggers that stay quiet below INFO until the application configures structlog."""

from functools import cache
from typing import Any

import structlog
from structlog.typing import Processor

from policies.core.logs.base import DEFAULT_LEVEL, processors, stderr_logger


@cache
def _default_processors() -> list[Processor]:
    return processors()


_DEFAULT_WRAPPER = structlog.make_filtering_bound_logger(DEFAULT_LEVEL)


class LibraryLogger:
    """Uses the application's structlog setup once there is one, else `configure`'s defaults.

    Decided on every call, so configuring structlog after import still takes effect.
    """

    def __init__(self, name: str, context: dict[str, Any]) -> None:
        self._name, self._context = name, context

    def bind(self, **values: Any) -> "LibraryLogger":
        return LibraryLogger(self._name, {**self._context, **values})

    def _logger(self) -> Any:
        if structlog.is_configured():
            return structlog.get_logger(self._name, **self._context)
        return structlog.wrap_logger(
            stderr_logger(),
            processors=_default_processors(),
            wrapper_class=_DEFAULT_WRAPPER,
            **self._context,
        )

    def __getattr__(self, method: str) -> Any:
        """`debug`, `info`, `warning`, `error` and the rest of structlog's methods."""
        return getattr(self._logger(), method)


def get_logger(name: str) -> Any:
    """A structlog logger. Pass context as keywords: `log.info("checkpoint", path=path)`."""
    return LibraryLogger(name, {})
