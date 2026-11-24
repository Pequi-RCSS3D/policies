"""World gravity in the body frame of a floating base, from its orientation."""

from typing import Any

from policies.core.config import Config
from policies.io.inputs.transforms.base import Transform, transforms


class ProjectedGravityConfig(Config):
    pass


@transforms.register("projected_gravity", ProjectedGravityConfig)
def projected_gravity(config: ProjectedGravityConfig, **_: Any) -> Transform:
    """Quaternion (..., 4) in w, x, y, z order to world gravity (0, 0, -1) in the body frame."""

    def apply(quaternion: Any, xp: Any) -> Any:
        norm = xp.sqrt(xp.sum(quaternion * quaternion, axis=-1, keepdims=True))
        unit = quaternion / norm
        w, x, y, z = unit[..., 0], unit[..., 1], unit[..., 2], unit[..., 3]
        return xp.stack(
            [2 * (w * y - x * z), -2 * (w * x + y * z), 2 * (x * x + y * y) - 1], axis=-1
        )

    return apply
