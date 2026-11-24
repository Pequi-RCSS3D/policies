"""Real-valued arrays, optionally clipped."""

from collections.abc import Mapping
from typing import Any

from policies.io.outputs.kinds.base import kinds
from policies.io.outputs.kinds.shaped import ShapedConfig


class ContinuousConfig(ShapedConfig):
    low: float | None = None
    high: float | None = None


class Continuous:
    distribution: Mapping[str, Any] = {"type": "normal"}

    def __init__(self, config: ContinuousConfig, **_: Any) -> None:
        self.config = config
        self.shape = config.event_shape

    def decode(self, action: Any, xp: Any) -> Any:
        if self.config.low is None and self.config.high is None:
            return action
        return xp.clip(action, self.config.low, self.config.high)


kinds.register("continuous", ContinuousConfig)(Continuous)
