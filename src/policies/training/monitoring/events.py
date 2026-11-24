"""Events the trainer emits, and how hooks treat each kind."""

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    """Events the trainer emits. Hooks may also receive custom kinds as plain strings."""

    START = "start"
    ITERATION = "iteration"
    UPDATE = "update"
    EVALUATION = "evaluation"
    CHECKPOINT = "checkpoint"
    CONTROL = "control"
    END = "end"
    ERROR = "error"


@dataclass(frozen=True)
class EventTraits:
    throttled: bool = False
    """Hooks thin these out with `every`."""
    scalar: bool = False
    """The data holds metrics, which scalar loggers record."""


NO_TRAITS = EventTraits()

event_traits: dict[str, EventTraits] = {
    EventKind.ITERATION: EventTraits(throttled=True, scalar=True),
    EventKind.UPDATE: EventTraits(throttled=True, scalar=True),
    EventKind.EVALUATION: EventTraits(scalar=True),
}
"""Traits by kind. Add custom kinds here. Kinds not listed have none."""


@dataclass(frozen=True)
class Event:
    """Something that happened during training, at an iteration step."""

    kind: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    time: float = field(default_factory=time.time)

    @property
    def traits(self) -> EventTraits:
        return event_traits.get(self.kind, NO_TRAITS)
