"""Converts to another dtype."""

from typing import Any

from policies.core.config import Config
from policies.io.inputs.transforms.base import Transform, transforms


class CastConfig(Config):
    dtype: str = "float32"


@transforms.register("cast", CastConfig)
def cast(config: CastConfig, **_: Any) -> Transform:
    return lambda x, xp: xp.astype(x, getattr(xp, config.dtype))
