"""A single real value, optionally clipped."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.io.outputs.kinds.base import kinds


class ScalarConfig(Config):
    low: float | None = None
    high: float | None = None


class Scalar:
    shape: tuple[int, ...] = ()
    distribution: Mapping[str, Any] = {"type": "normal"}

    def __init__(self, config: ScalarConfig, **_: Any) -> None:
        self.config = config

    def decode(self, action: Any, xp: Any) -> Any:
        if self.config.low is None and self.config.high is None:
            return action
        return xp.clip(action, self.config.low, self.config.high)


kinds.register("scalar", ScalarConfig)(Scalar)
