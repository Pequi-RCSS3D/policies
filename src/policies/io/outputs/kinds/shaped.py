"""Config of kinds whose actions are a vector of `size` or an array of `shape`."""

from typing import Self

from pydantic import model_validator

from policies.core.config import Config


class ShapedConfig(Config):
    size: int | None = None
    shape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.size is None) == (self.shape is None):
            raise ValueError("give either `size` or `shape`")
        if any(dim < 1 for dim in self.event_shape):
            raise ValueError(f"sizes must be positive, got {self.event_shape}")
        return self

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.shape if self.shape is not None else (self.size or 0,)
