"""The previous value of an output."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.io.bookkeeping.arrays import select, source_dtype, source_shape, zeros
from policies.io.bookkeeping.base import TrackerContext, trackers


class LastConfig(Config):
    source: str


class Last:
    """The previous step's value of an output. Zeros after a reset."""

    def __init__(self, config: LastConfig, context: TrackerContext) -> None:
        self.context, self.source = context, config.source
        self.sources = (config.source,)
        self.name = f"last.{config.source}"
        self.slots = {self.name: source_shape(context, config.source)}
        self.value = zeros(context, self.slots[self.name], source_dtype(context, config.source))

    def reset(self, mask: Any = None) -> None:
        cleared = self.context.xp.zeros_like(self.value)
        self.value = cleared if mask is None else select(self.context, mask, cleared, self.value)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        return {self.name: self.value}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        if self.source in outputs:
            self.value = self.context.xp.astype(outputs[self.source], self.value.dtype)


trackers.register("last", LastConfig)(Last)
