"""Clips to a range."""

from typing import Any

from policies.core.config import Config
from policies.io.inputs.transforms.base import Transform, transforms


class ClipConfig(Config):
    low: float
    high: float


@transforms.register("clip", ClipConfig)
def clip(config: ClipConfig, **_: Any) -> Transform:
    return lambda x, xp: xp.clip(x, config.low, config.high)
