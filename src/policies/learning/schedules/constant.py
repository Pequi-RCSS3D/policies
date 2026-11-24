"""Keeps the learning rate fixed."""

from typing import Any

from policies.core.config import Config
from policies.learning.schedules.base import schedules


class ConstantConfig(Config):
    pass


class Constant:
    def __init__(self, config: ConstantConfig, **_: Any) -> None:
        pass

    def step(self, lr: float, iteration: int, metrics: dict[str, float]) -> float:
        return lr


schedules.register("constant", ConstantConfig)(Constant)
