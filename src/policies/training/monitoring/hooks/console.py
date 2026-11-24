"""Logs events through structlog."""

from typing import Any

from policies.core.config import Config
from policies.core.logs import get_logger
from policies.training.monitoring.events import Event, EventKind
from policies.training.monitoring.hooks.base import hooks
from policies.training.monitoring.hooks.scalars import scalars


class ConsoleHookConfig(Config):
    every: int = 10
    max_metrics: int = 6
    """Scalars shown per metric event."""


class ConsoleHook:
    """Logs metric events with their first few scalars, and other events with all their data."""

    def __init__(self, config: ConsoleHookConfig, **_: Any) -> None:
        self.every = config.every
        self.max_metrics = config.max_metrics
        self.log = get_logger("policies.training")

    def __call__(self, event: Event) -> None:
        fields = event.data
        if event.traits.scalar:
            values = list(scalars(event.data).items())[: self.max_metrics]
            fields = {key: float(f"{value:.4g}") for key, value in values}
        method = self.log.error if event.kind == EventKind.ERROR else self.log.info
        method(event.kind, step=event.step, **fields)


hooks.register("console", ConsoleHookConfig)(ConsoleHook)
