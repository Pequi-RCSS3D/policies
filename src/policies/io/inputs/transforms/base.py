"""The transform contract and registry."""

from collections.abc import Callable
from typing import Annotated, Any

from policies.core.registry import Registry

type Transform = Callable[[Any, Any], Any]
"""`transform(x, xp) -> y`, with `xp` the array namespace of `x`."""


transforms: Registry[Transform] = Registry("transform")

type TransformConfig = Annotated[dict[str, Any], transforms]
"""A transform config, validated and stored canonical."""
