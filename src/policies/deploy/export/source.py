"""What export needs, from a learner or from any other training code."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from policies.learning import Actor


class Exportable(Protocol):
    """An actor and the observations its bookkeeping reads. A `Learner` is one.

    Observations missing from `observation_dtypes` are float32. Others than bookkeeping reads
    are left out of the artifact. An exportable may also have `parents`, the ids of artifacts
    it derives from, which `export_artifact` records.
    """

    @property
    def actor(self) -> Actor: ...
    @property
    def observation_shapes(self) -> Mapping[str, tuple[int, ...]]: ...
    @property
    def observation_dtypes(self) -> Mapping[str, str]: ...


@dataclass(frozen=True)
class ExportSource:
    """An `Exportable` from parts, for actors trained without a `Learner`."""

    actor: Actor
    observation_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    observation_dtypes: Mapping[str, str] = field(default_factory=dict)
    parents: tuple[str, ...] = ()
