"""Named, typed parts and checks of real arrays against them."""

from collections.abc import Mapping
from typing import Any, Self

from array_api_compat import array_namespace
from pydantic import Field, model_validator

from policies.core.config import Config
from policies.core.parts.dtype import DType


class PartError(ValueError):
    """An array does not match its part spec."""


class PartSpec(Config):
    """One named part. `shape` is per sample, batch dimensions come in front."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    shape: tuple[int, ...] = ()
    dtype: DType = DType.FLOAT32
    low: float | None = None
    high: float | None = None
    privileged: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if any(size < 1 for size in self.shape):
            raise ValueError(f"part {self.name!r}: sizes must be positive, got {self.shape}")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"part {self.name!r}: low {self.low} is above high {self.high}")
        if self.dtype == DType.BOOL and (self.low is not None or self.high is not None):
            raise ValueError(f"part {self.name!r}: bool parts take no bounds")
        return self

    @property
    def size(self) -> int:
        """Number of elements per sample."""
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    def check(self, value: Any) -> tuple[int, ...]:
        """Raise `PartError` on mismatch. Returns the batch shape."""
        xp = array_namespace(value)
        shape = tuple(value.shape)
        ndim = len(self.shape)
        if shape[len(shape) - ndim :] != self.shape or len(shape) < ndim:
            raise PartError(
                f"part {self.name!r}: expected shape (..., {self._dims()}), got {shape}"
            )
        if value.dtype != getattr(xp, self.dtype):
            raise PartError(f"part {self.name!r}: expected {self.dtype}, got {value.dtype}")
        if self.dtype.startswith("float") and not bool(xp.all(xp.isfinite(value))):
            raise PartError(f"part {self.name!r}: contains NaN or infinity")
        if self.low is not None and not bool(xp.all(value >= self.low)):
            raise PartError(f"part {self.name!r}: values below {self.low}")
        if self.high is not None and not bool(xp.all(value <= self.high)):
            raise PartError(f"part {self.name!r}: values above {self.high}")
        return shape[: len(shape) - ndim]

    def _dims(self) -> str:
        return ", ".join(str(dim) for dim in self.shape)


class Parts(Config):
    """An ordered collection of parts with unique names."""

    specs: tuple[PartSpec, ...] = ()

    @model_validator(mode="after")
    def _unique(self) -> Self:
        seen: set[str] = set()
        for spec in self.specs:
            if spec.name in seen:
                raise ValueError(f"duplicate part {spec.name!r}")
            seen.add(spec.name)
        return self

    @classmethod
    def of(cls, *specs: PartSpec) -> Self:
        return cls(specs=specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def __getitem__(self, name: str) -> PartSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no part {name!r}. Parts: {', '.join(self.names) or 'none'}")

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.specs)

    def check(self, values: Mapping[str, Any]) -> tuple[int, ...]:
        """Check every part is present, valid and shares one batch shape. Returns it."""
        missing = [name for name in self.names if name not in values]
        extra = [name for name in values if name not in self]
        if missing or extra:
            raise PartError(f"missing parts {missing}, unexpected parts {extra}")
        batch: tuple[int, ...] | None = None
        for spec in self.specs:
            shape = spec.check(values[spec.name])
            if batch is None:
                batch = shape
            elif shape != batch:
                raise PartError(f"part {spec.name!r}: batch shape {shape} differs from {batch}")
        return batch or ()
