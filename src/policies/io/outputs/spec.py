"""Outputs composed of named parts, each of a registered kind and distribution."""

from collections.abc import Mapping
from typing import Any, Self

from array_api_compat import array_namespace
from pydantic import Field, model_validator

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.outputs.distributions import Distribution, DistributionConfig, distributions
from policies.io.outputs.kinds import Kind, KindConfig, KindWithExtras, kinds


class OutputPart(Config):
    """Kind and distribution are stored canonical, so the fingerprint sees every default."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: KindConfig
    distribution: DistributionConfig
    """Given, it overrides the kind's default. Without a `type`, or with the default's, it
    updates the default."""

    @model_validator(mode="before")
    @classmethod
    def _resolve(cls, data: Any) -> Any:
        """Merges the given distribution into the kind's default."""
        if not isinstance(data, Mapping) or not isinstance(data.get("kind"), Mapping):
            return data
        if data["kind"].get("type") not in kinds:
            return data  # The kind field reports it.
        default = dict(kinds.build(data["kind"]).distribution)
        override = data.get("distribution") or {}
        if override.get("type", default["type"]) != default["type"]:
            return data
        return {**data, "distribution": {**default, **override}}

    @property
    def kind_type(self) -> str:
        """The registered name of the kind."""
        kind_type: str = self.kind["type"]
        return kind_type


class Outputs(Config):
    parts: tuple[OutputPart, ...]

    @model_validator(mode="after")
    def _valid(self) -> Self:
        names = [part.name for part in self.parts]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate output parts in {names}")
        return self

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(part.name for part in self.parts)

    def __getitem__(self, name: str) -> OutputPart:
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"no output part {name!r}. Parts: {', '.join(self.names)}")

    def compile(self, **context: Any) -> "OutputDecoder":
        return OutputDecoder(self, **context)


class OutputDecoder:
    """Raw model outputs to actions, the mode of each part's distribution, and actions to values.

    `context` reaches kinds and distributions, e.g. `initial_std`.
    """

    def __init__(self, outputs: Outputs, **context: Any) -> None:
        self.outputs = outputs
        self.kinds: dict[str, Kind] = {
            part.name: kinds.build(part.kind, **context) for part in outputs.parts
        }
        self.distributions: dict[str, Distribution] = {
            part.name: distributions.build(
                part.distribution,
                **{**context, "shape": self.kinds[part.name].shape},
            )
            for part in outputs.parts
        }
        self.detailed = {
            name: kind for name, kind in self.kinds.items() if isinstance(kind, KindWithExtras)
        }
        """Kinds with per-step extras, by part."""

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Action shapes per part, as bookkeeping keeps them."""
        return {name: distribution.shape for name, distribution in self.distributions.items()}

    @property
    def dtypes(self) -> dict[str, DType]:
        return {name: distribution.dtype for name, distribution in self.distributions.items()}

    def mode(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: distribution.mode(raw[name], array_namespace(raw[name]))
            for name, distribution in self.distributions.items()
        }

    def decode(self, actions: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: kind.decode(actions[name], array_namespace(actions[name]))
            for name, kind in self.kinds.items()
        }

    def extras(self, actions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Per-step extras by part, only for kinds that have them."""
        return {
            name: dict(kind.extras(actions[name], array_namespace(actions[name])))
            for name, kind in self.detailed.items()
        }

    def __call__(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        return self.decode(self.mode(raw))
