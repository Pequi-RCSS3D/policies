"""A Legendre memory of a source over a sliding time window."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import Field

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.bookkeeping.arrays import select, source_dtype, source_shape, zeros
from policies.io.bookkeeping.base import TrackerContext, trackers
from policies.io.bookkeeping.legendre_reset import LegendreReset
from policies.io.bookkeeping.legendre_system import legendre_system
from policies.io.bookkeeping.zero_order_hold import zero_order_hold


class LegendreConfig(Config):
    source: str
    window: float = Field(gt=0)
    order: int = Field(ge=1)
    dt: float = Field(gt=0)
    name: str
    reset: LegendreReset = LegendreReset.ZERO


class Legendre:
    """The last `window` seconds of a source as `order` Legendre coefficients per feature.

    With `r` the fraction of the window ago, `x(t - r * window) ~ sum_i c_i P_i(2 r - 1)`. A fixed
    linear recurrence, the LMU system held over each step of `dt`, so gradients never flow
    through time. Observations advance it when observed, output parts when committed, so it ends
    at the previous action. Zeros after a reset, then `reset` sets how the next input starts it.
    Float64 if the source is, else float32.
    """

    def __init__(self, config: LegendreConfig, context: TrackerContext) -> None:
        self.context, self.source = context, config.source
        self.sources = (config.source,)
        shape = source_shape(context, config.source)
        wide = source_dtype(context, config.source) == DType.FLOAT64
        self.name = f"legendre.{config.name}"
        self.slots = {self.name: (*shape, config.order)}
        self.state = zeros(context, self.slots[self.name], DType.FLOAT64 if wide else DType.FLOAT32)
        transition, intake = zero_order_hold(
            *legendre_system(config.window, config.order), config.dt
        )
        # Rows weigh the state, the input, and the input again on a fresh start, whose state is 0.
        weights = np.concatenate(
            [transition.T, intake[None], (config.reset.start(config.order) @ transition.T)[None]]
        )
        xp, device = context.xp, context.device
        self.weights = xp.asarray(weights, dtype=self.state.dtype, device=device)
        self.fresh = xp.ones(context.batch, dtype=xp.bool, device=device)

    def reset(self, mask: Any = None) -> None:
        xp = self.context.xp
        cleared = xp.zeros_like(self.state)
        if mask is None:
            self.state, self.fresh = cleared, xp.ones_like(self.fresh)
        else:
            self.state = select(self.context, mask, cleared, self.state)
            self.fresh = xp.logical_or(self.fresh, mask)

    def _advance(self, value: Any) -> Any:
        xp, batch, state = self.context.xp, self.context.batch, self.state
        x = xp.expand_dims(xp.astype(value, state.dtype), axis=-1)
        fresh = xp.reshape(
            xp.astype(self.fresh, state.dtype), (*batch, *(1,) * (x.ndim - len(batch)))
        )
        stacked = xp.concat([state, x, x * fresh], axis=-1)
        # One flat matrix product, far faster in NumPy than one per sample.
        flat = xp.reshape(stacked, (-1, stacked.shape[-1]))
        return xp.reshape(xp.matmul(flat, self.weights), state.shape)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        if self.source not in observation:
            # An output part, advanced on commit.
            return {self.name: self.state}
        state = self._advance(observation[self.source])
        if update:
            self.state, self.fresh = state, self.context.xp.zeros_like(self.fresh)
        return {self.name: state}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        if self.source in outputs:
            self.state = self._advance(outputs[self.source])
            self.fresh = self.context.xp.zeros_like(self.fresh)


trackers.register("legendre", LegendreConfig)(Legendre)
