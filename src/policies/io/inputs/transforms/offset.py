"""Adds a constant."""

from typing import Any

from policies.core.config import Config
from policies.io.inputs.transforms.base import Transform, transforms


class OffsetConfig(Config):
    value: float


@transforms.register("offset", OffsetConfig)
def offset(config: OffsetConfig, **_: Any) -> Transform:
    return lambda x, xp: x + config.value
