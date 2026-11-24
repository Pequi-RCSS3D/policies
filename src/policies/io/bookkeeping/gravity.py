"""Gravity in the body frame, estimated from a gyroscope and an accelerometer."""

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.bookkeeping.arrays import select, source_dtype, source_shape, zeros
from policies.io.bookkeeping.base import TrackerContext, trackers
from policies.io.bookkeeping.gyroscope_unit import GyroscopeUnit


class GravityConfig(Config):
    gyroscope: str
    accelerometer: str
    dt: float = Field(gt=0)
    gain: float = Field(default=1.0, ge=0)
    gyroscope_unit: GyroscopeUnit = GyroscopeUnit.RADIANS_PER_SECOND
    name: str = "estimate"


def _cross(a: Any, b: Any, xp: Any) -> Any:
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return xp.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-1)


def _norm(x: Any, xp: Any) -> Any:
    return xp.sqrt(xp.sum(x * x, axis=-1, keepdims=True))


def _sinc(x: Any, xp: Any) -> Any:
    """sin(x) / x, 1 at 0."""
    safe = xp.where(x > 0, x, xp.ones_like(x))
    return xp.where(x > 0, xp.sin(safe) / safe, xp.ones_like(x))


def _rotate(vector: Any, rotation: Any, xp: Any) -> Any:
    """Rodrigues rotation of `vector` by the rotation vector `rotation`."""
    angle = _norm(rotation, xp)
    half = _sinc(angle / 2, xp)
    dot = xp.sum(rotation * vector, axis=-1, keepdims=True)
    return (
        xp.cos(angle) * vector
        + _sinc(angle, xp) * _cross(rotation, vector, xp)
        + 0.5 * half * half * dot * rotation
    )


class Gravity:
    """Unit gravity in the body frame, (0, 0, -1) when upright, by a Mahony filter on the sphere.

    Turns the estimate exactly by the gyroscope over `dt`, then toward the accelerometer at rate
    `gain` in 1/s, so 0 trusts the gyroscope alone. The accelerometer reads specific force, up at
    rest. A reset restarts from the accelerometer's gravity. Float64 if either source is, else
    float32.
    """

    def __init__(self, config: GravityConfig, context: TrackerContext) -> None:
        self.context, self.config = context, config
        self.sources = (config.gyroscope, config.accelerometer)
        for source in self.sources:
            if source_shape(context, source) != (3,):
                raise ValueError(f"gravity source {source!r} must have shape (3,)")
        wide = DType.FLOAT64 in {source_dtype(context, source) for source in self.sources}
        dtype = DType.FLOAT64 if wide else DType.FLOAT32
        self.dtype = getattr(context.xp, dtype)
        self.name = f"gravity.{config.name}"
        self.slots = {self.name: (3,)}
        xp = context.xp
        self.down = xp.asarray([0.0, 0.0, -1.0], dtype=self.dtype, device=context.device)
        self.value = zeros(context, (3,), dtype) + self.down
        self.fresh = xp.ones(context.batch, dtype=xp.bool, device=context.device)

    def reset(self, mask: Any = None) -> None:
        xp = self.context.xp
        self.fresh = xp.ones_like(self.fresh) if mask is None else xp.logical_or(self.fresh, mask)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        xp, config = self.context.xp, self.config
        rate = xp.astype(observation[config.gyroscope], self.dtype)
        rate = rate * config.gyroscope_unit.to_radians
        force = xp.astype(observation[config.accelerometer], self.dtype)
        length = _norm(force, xp)
        # Gravity opposes specific force. Without a reading it is unknown, so no correction.
        measured = -force / xp.where(length > 0, length, xp.ones_like(length))
        # A body turning at `rate` sees fixed world vectors turn by -rate.
        predicted = _rotate(self.value, -config.dt * rate, xp)
        error = _cross(measured, predicted, xp)
        turned = _rotate(predicted, -config.dt * config.gain * error, xp)
        turned = turned / _norm(turned, xp)
        initial = xp.where(length > 0, measured, self.down)
        value = select(self.context, self.fresh, initial, turned)
        if update:
            self.value = value
            self.fresh = xp.zeros_like(self.fresh)
        return {self.name: value}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        pass


trackers.register("gravity", GravityConfig)(Gravity)
