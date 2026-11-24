"""Units a gyroscope reports angular velocity in."""

import math
from enum import StrEnum


class GyroscopeUnit(StrEnum):
    RADIANS_PER_SECOND = "rad/s"
    DEGREES_PER_SECOND = "deg/s"

    @property
    def to_radians(self) -> float:
        """Factor from this unit to rad/s."""
        return 1.0 if self is GyroscopeUnit.RADIANS_PER_SECOND else math.pi / 180
