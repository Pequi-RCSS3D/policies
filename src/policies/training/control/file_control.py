"""Reading control messages from `control.jsonl` in a run directory."""

import json
from pathlib import Path

from policies.training.control.messages import ControlMessage

FILENAME = "control.jsonl"


def _decode(line: bytes) -> ControlMessage | None:
    """The message on a line, None for blank or foreign lines. Raises ValueError when malformed."""
    if not line.strip():
        return None
    record = json.loads(line)
    if not isinstance(record, dict) or not isinstance(record.get("id"), int):
        return None
    changes = record.get("set") or {}
    command = record.get("command")
    return ControlMessage(
        id=record["id"],
        changes=dict(changes) if isinstance(changes, dict) else {},
        command=command if isinstance(command, str) else None,
    )


class FileControl:
    """Reads messages appended to `run_directory/control.jsonl` since the last poll.

    Keeps a byte offset, so each poll reads only what was added. A malformed last line is taken
    as still being written and read again next time.
    """

    def __init__(self, run_directory: Path) -> None:
        self.path = Path(run_directory) / FILENAME
        self.offset = 0

    def poll(self) -> list[ControlMessage]:
        try:
            with self.path.open("rb") as file:
                if file.seek(0, 2) < self.offset:
                    self.offset = 0  # The file was replaced.
                file.seek(self.offset)
                data = file.read()
        except FileNotFoundError:
            return []
        *complete, rest = data.split(b"\n")
        messages = []
        for line in complete:
            try:
                message = _decode(line)
            except ValueError:
                continue  # Finished but malformed, e.g. by a writer that crashed.
            if message is not None:
                messages.append(message)
        self.offset += len(data) - len(rest)
        try:
            message = _decode(rest)
        except ValueError:
            return messages
        self.offset += len(rest)
        if message is not None:
            messages.append(message)
        return messages
