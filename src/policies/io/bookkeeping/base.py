"""The tracker contract, what trackers are built with, and the tracker registry."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

from policies.core.registry import Registry


class Tracker(Protocol):
    """State kept between steps, exposed as named slots."""

    @property
    def slots(self) -> Mapping[str, tuple[int, ...]]:
        """Slot names and their per-sample shapes."""
        ...

    @property
    def sources(self) -> tuple[str, ...]:
        """Every observation and output it reads, by name, even those it needs no shape of.

        Exported artifacts record only these observations, so deployed policies get no others.
        """
        ...

    def reset(self, mask: Any = None) -> None: ...
    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        """Slot values for this step. `update=False` peeks without changing state."""
        ...

    def commit(self, outputs: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True)
class TrackerContext:
    """What a tracker needs to allocate itself. Sources missing from `dtypes` are float32."""

    shapes: Mapping[str, tuple[int, ...]]
    batch: tuple[int, ...]
    xp: Any
    dtypes: Mapping[str, str] = field(default_factory=dict)
    device: Any = None


trackers: Registry[Tracker] = Registry("tracker")
"""Factories are called as `factory(config, context=...)`."""

type TrackerConfig = Annotated[dict[str, Any], trackers]
"""A tracker config, validated and stored canonical."""
