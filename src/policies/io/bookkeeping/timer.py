"""Time since the last reset."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.io.bookkeeping.arrays import select, zeros
from policies.io.bookkeeping.base import TrackerContext, trackers


class TimerConfig(Config):
    name: str
    dt: float = 1.0


class Timer:
    """Time since the last reset, in steps times `dt`."""

    def __init__(self, config: TimerConfig, context: TrackerContext) -> None:
        self.context, self.dt = context, config.dt
        self.sources: tuple[str, ...] = ()
        self.name = f"timer.{config.name}"
        self.slots = {self.name: (1,)}
        self.value = zeros(context, (1,))

    def reset(self, mask: Any = None) -> None:
        cleared = self.context.xp.zeros_like(self.value)
        self.value = cleared if mask is None else select(self.context, mask, cleared, self.value)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        return {self.name: self.value}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        self.value = self.value + self.dt


trackers.register("timer", TimerConfig)(Timer)
