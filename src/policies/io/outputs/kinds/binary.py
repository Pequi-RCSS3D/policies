"""Independent on-or-off flags."""

from collections.abc import Mapping
from typing import Any

from policies.io.outputs.kinds.base import kinds
from policies.io.outputs.kinds.shaped import ShapedConfig


class BinaryConfig(ShapedConfig):
    pass


class Binary:
    """Decodes to booleans."""

    distribution: Mapping[str, Any] = {"type": "bernoulli"}

    def __init__(self, config: BinaryConfig, **_: Any) -> None:
        self.shape = config.event_shape

    def decode(self, action: Any, xp: Any) -> Any:
        return xp.astype(action, xp.bool)


kinds.register("binary", BinaryConfig)(Binary)
