"""Appends events to a JSON Lines file."""

import json
from pathlib import Path
from typing import Any

from policies.core.config import Config
from policies.training.monitoring.events import Event
from policies.training.monitoring.hooks.base import hooks


class JsonlHookConfig(Config):
    path: str = "events.jsonl"
    every: int = 1


class JsonlHook:
    """Appends one JSON line per event to a file in the run directory."""

    def __init__(self, config: JsonlHookConfig, run_directory: Path, **_: Any) -> None:
        self.every = config.every
        self.path = Path(run_directory) / config.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        record = {"kind": event.kind, "step": event.step, "time": event.time, "data": event.data}
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


hooks.register("jsonl", JsonlHookConfig)(JsonlHook)
