"""Control messages a run applies between iterations."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol


class Command(StrEnum):
    """Commands every trainer knows. Add more through `Trainer.commands`."""

    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    CHECKPOINT = "checkpoint"


type Tunable = float | int | bool | str
"""A value that may change while a job runs."""


class ControlError(ValueError):
    """A control message was refused."""


@dataclass(frozen=True)
class ControlMessage:
    """Either a set of changes or a command."""

    id: int
    changes: dict[str, Tunable] = field(default_factory=dict)
    command: str | None = None


class ControlSource(Protocol):
    def poll(self) -> list[ControlMessage]:
        """Messages since the last poll. Ids increase, so a checkpoint can record a position."""
        ...
