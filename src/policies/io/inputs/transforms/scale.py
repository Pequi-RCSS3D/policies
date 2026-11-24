"""Multiplies by a constant."""

from typing import Any

from policies.core.config import Config
from policies.io.inputs.transforms.base import Transform, transforms


class ScaleConfig(Config):
    factor: float


@transforms.register("scale", ScaleConfig)
def scale(config: ScaleConfig, **_: Any) -> Transform:
    return lambda x, xp: x * config.factor
