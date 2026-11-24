"""Sending control messages to a run through its `control.jsonl`."""

import json
from pathlib import Path
from typing import Any

import yaml

from policies.training.control.file_control import FILENAME, FileControl
from policies.training.control.messages import ControlError, Tunable
from policies.training.control.tunables import check_bounds, check_path

RUN_CONFIG = "config.yaml"
"""What makes a directory a run directory. The trainer writes it on start."""


def _run_config(run_directory: Path) -> dict[str, Any]:
    path = Path(run_directory) / RUN_CONFIG
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ControlError(
            f"{Path(run_directory).resolve()} is not a run directory: it has no {RUN_CONFIG}"
        ) from None
    return data if isinstance(data, dict) else {}


def _append(run_directory: Path, record: dict[str, Any]) -> int:
    path = Path(run_directory) / FILENAME
    ids = [message.id for message in FileControl(run_directory).poll()]
    record = {"id": max(ids, default=0) + 1, **record}
    prefix = ""
    if path.exists() and path.stat().st_size:
        with path.open("rb") as file:
            file.seek(-1, 2)
            prefix = "" if file.read(1) == b"\n" else "\n"
    with path.open("a", encoding="utf-8") as file:
        file.write(prefix + json.dumps(record) + "\n")
    return int(record["id"])


def send_changes(run_directory: Path, changes: dict[str, Tunable]) -> int:
    """Ask a run to change tunable values. Returns the message id.

    Checks paths and bounds against the run's config first. The run checks types and again
    everything else.
    """
    if not changes:
        raise ControlError("no changes given")
    trainer = _run_config(run_directory).get("trainer") or {}
    tunable, bounds = trainer.get("tunable"), trainer.get("bounds") or {}
    for path, value in changes.items():
        if tunable is not None:
            check_path(path, tunable)
        if path in bounds:
            low, high = bounds[path]
            check_bounds(path, value, (low, high))
    return _append(run_directory, {"set": dict(changes)})


def send_command(run_directory: Path, command: str) -> int:
    """Ask a run to run a built-in `Command` or a custom one. Returns the message id.

    The run reports names it does not know as error events.
    """
    if not command.strip():
        raise ControlError("no command given")
    _run_config(run_directory)
    return _append(run_directory, {"command": command})
