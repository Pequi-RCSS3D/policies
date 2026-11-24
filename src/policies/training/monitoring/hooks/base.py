"""The hook protocol, the registry and plain functions as hooks."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from policies.core.registry import Registry
from policies.training.monitoring.events import Event


@runtime_checkable
class Hook(Protocol):
    """Receives events. `every` thins out throttled kinds, see `event_traits`."""

    every: int

    def __call__(self, event: Event) -> None: ...


hooks: Registry[Hook] = Registry("hook")
"""Factories are called as `factory(config, run_directory=..., **context)`. The trainer adds
`trainer` and `spec`."""


@dataclass
class FunctionHook:
    """A plain function used as a hook."""

    function: Callable[[Event], None]
    every: int = 1

    def __call__(self, event: Event) -> None:
        self.function(event)
