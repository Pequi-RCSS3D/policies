"""Inputs composed of named parts, each read from a source and transformed."""

from collections.abc import Mapping
from typing import Any, Self

from array_api_compat import array_namespace
from pydantic import Field, model_validator

from policies.core.config import Config
from policies.core.parts import DType, PartError, Parts, PartSpec
from policies.io.inputs.transforms import Transform, TransformConfig, transforms


class InputPart(Config):
    """One input part. `source` is an observation key or a bookkeeping slot."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    source: str
    shape: tuple[int, ...] = ()
    dtype: DType = DType.FLOAT32
    privileged: bool = False
    transforms: tuple[TransformConfig, ...] = ()
    normalize: bool = False

    @property
    def spec(self) -> PartSpec:
        return PartSpec(
            name=self.name, shape=self.shape, dtype=self.dtype, privileged=self.privileged
        )


class Inputs(Config):
    """An ordered set of input parts."""

    parts: tuple[InputPart, ...]

    @model_validator(mode="after")
    def _valid(self) -> Self:
        names = [part.name for part in self.parts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate input parts in {names}")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(part.name for part in self.parts)

    @property
    def specs(self) -> Parts:
        return Parts(specs=tuple(part.spec for part in self.parts))

    def __getitem__(self, name: str) -> InputPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"no input part {name!r}. Parts: {', '.join(self.names)}")

    def compile(self, **context: Any) -> "InputPipeline":
        return InputPipeline(self, **context)


class InputPipeline:
    """Builds input parts from observations and bookkeeping slots."""

    def __init__(self, inputs: Inputs, **context: Any) -> None:
        self.inputs = inputs
        self.specs = inputs.specs
        self.checked = False
        self._transforms: dict[str, list[Transform]] = {
            part.name: [transforms.build(t, **context) for t in part.transforms]
            for part in inputs.parts
        }

    def __call__(
        self, observation: Mapping[str, Any], slots: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        slots = slots or {}
        parts: dict[str, Any] = {}
        for part in self.inputs.parts:
            if part.source in observation:
                value = observation[part.source]
            elif part.source in slots:
                value = slots[part.source]
            else:
                raise PartError(f"input part {part.name!r}: no source {part.source!r}")
            xp = array_namespace(value)
            for transform in self._transforms[part.name]:
                value = transform(value, xp)
            parts[part.name] = value
        if not self.checked:
            # Checks force device syncs, so only the first call pays for them.
            self.specs.check(parts)
            self.checked = True
        return parts
