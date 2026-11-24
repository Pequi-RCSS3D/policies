"""Cosine decay to a minimum learning rate."""

import math
from typing import Any

from policies.core.config import Config
from policies.learning.schedules.base import schedules


class CosineConfig(Config):
    iterations: int
    """Iterations until the learning rate reaches `min_lr`."""
    min_lr: float = 0.0


class Cosine:
    def __init__(self, config: CosineConfig, base_lr: float, **_: Any) -> None:
        self.config, self.base = config, base_lr

    def step(self, lr: float, iteration: int, metrics: dict[str, float]) -> float:
        t = min(iteration / self.config.iterations, 1.0)
        return self.config.min_lr + 0.5 * (self.base - self.config.min_lr) * (
            1 + math.cos(math.pi * t)
        )


schedules.register("cosine", CosineConfig)(Cosine)
