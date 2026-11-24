"""Several trackers acting as one."""

from collections.abc import Mapping, Sequence
from typing import Any

from policies.io.bookkeeping.base import Tracker


class Compose:
    """Several trackers acting as one. Slot names must not collide."""

    def __init__(self, parts: Sequence[Tracker]) -> None:
        self.parts = list(parts)
        self.slots: dict[str, tuple[int, ...]] = {}
        sources: dict[str, None] = {}
        for part in self.parts:
            clash = set(self.slots) & set(part.slots)
            if clash:
                raise ValueError(f"bookkeeping slots defined twice: {sorted(clash)}")
            self.slots.update(part.slots)
            if not hasattr(part, "sources"):
                raise TypeError(
                    f"tracker {type(part).__name__} must declare `sources`, the observations "
                    "and outputs it reads"
                )
            sources.update(dict.fromkeys(part.sources))
        self.sources = tuple(sources)
        """What the parts read, without repeats, in order."""

    def reset(self, mask: Any = None) -> None:
        for part in self.parts:
            part.reset(mask)

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        slots: dict[str, Any] = {}
        for part in self.parts:
            slots.update(part.observe(observation, update))
        return slots

    def commit(self, outputs: Mapping[str, Any]) -> None:
        for part in self.parts:
            part.commit(outputs)
