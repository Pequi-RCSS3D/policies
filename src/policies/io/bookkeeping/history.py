"""The last observations of a source."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.io.bookkeeping.arrays import select, source_dtype, source_shape, zeros
from policies.io.bookkeeping.base import TrackerContext, trackers


class HistoryConfig(Config):
    source: str
    length: int


class History:
    """The last `length` observations, oldest first. Refilled with the first value after a reset."""

    def __init__(self, config: HistoryConfig, context: TrackerContext) -> None:
        self.context, self.source = context, config.source
        self.sources = (config.source,)
        shape = source_shape(context, config.source)
        self.name = f"history.{config.source}"
        self.slots = {self.name: (config.length, *shape)}
        self.buffer = zeros(context, self.slots[self.name], source_dtype(context, config.source))
        self.fresh = context.xp.ones(context.batch, dtype=context.xp.bool, device=context.device)

    def reset(self, mask: Any = None) -> None:
        xp = self.context.xp
        self.fresh = xp.ones_like(self.fresh) if mask is None else xp.logical_or(self.fresh, mask)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        xp = self.context.xp
        value = xp.astype(observation[self.source], self.buffer.dtype)
        rolled = xp.concat(
            [
                self.buffer[..., 1:, *(slice(None),) * (value.ndim - len(self.context.batch))],
                xp.expand_dims(value, axis=len(self.context.batch)),
            ],
            axis=len(self.context.batch),
        )
        filled = xp.stack([value] * self.slots[self.name][0], axis=len(self.context.batch))
        buffer = select(self.context, self.fresh, filled, rolled)
        if update:
            self.buffer = buffer
            self.fresh = xp.zeros_like(self.fresh)
        return {self.name: buffer}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        pass


trackers.register("history", HistoryConfig)(History)
