import json
from pathlib import Path
from typing import Any

import pytest

from policies.training.control import (
    RUN_CONFIG,
    ControlError,
    ControlMessage,
    FileControl,
    Tunables,
    send_changes,
    send_command,
)

TRAINER = """\
trainer:
  tunable: [optimizer.lr, rollout.steps]
  bounds: {optimizer.lr: [0.0, 0.01]}
"""


@pytest.fixture(autouse=True)
def run_config(tmp_path: Path) -> None:
    """Senders need a run directory, which the trainer marks with its config."""
    (tmp_path / RUN_CONFIG).write_text(TRAINER, encoding="utf-8")


def test_round_trip(tmp_path: Path) -> None:
    control = FileControl(tmp_path)
    assert control.poll() == []
    assert send_changes(tmp_path, {"optimizer.lr": 1e-4}) == 1
    assert send_command(tmp_path, "pause") == 2
    assert control.poll() == [
        ControlMessage(1, {"optimizer.lr": 1e-4}),
        ControlMessage(2, command="pause"),
    ]
    assert control.poll() == []
    send_command(tmp_path, "stop")
    assert control.poll() == [ControlMessage(3, command="stop")]


def test_missing_directory_is_empty(tmp_path: Path) -> None:
    assert FileControl(tmp_path / "nope").poll() == []


def test_poll_reads_only_what_was_appended(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    control = FileControl(tmp_path)
    send_command(tmp_path, "pause")
    assert [message.id for message in control.poll()] == [1]
    assert control.offset == path.stat().st_size
    # Rewriting what was already read changes nothing: it is not read again.
    data = path.read_bytes()
    path.write_bytes(data.replace(b"pause", b"stop!"))
    assert control.poll() == []
    send_command(tmp_path, "resume")
    assert control.poll() == [ControlMessage(2, command="resume")]


def test_a_replaced_file_is_read_from_the_start(tmp_path: Path) -> None:
    control = FileControl(tmp_path)
    send_command(tmp_path, "checkpoint")
    send_command(tmp_path, "checkpoint")
    control.poll()
    (tmp_path / "control.jsonl").write_text('{"id": 1, "command": "stop"}\n', encoding="utf-8")
    assert control.poll() == [ControlMessage(1, command="stop")]


def test_a_complete_last_line_without_newline_is_read(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    path.write_text('{"id": 1, "command": "pause"}', encoding="utf-8")
    control = FileControl(tmp_path)
    assert control.poll() == [ControlMessage(1, command="pause")]
    assert send_command(tmp_path, "resume") == 2
    assert control.poll() == [ControlMessage(2, command="resume")]


@pytest.mark.parametrize("name", ["typo", "empty"])
def test_senders_need_a_run_directory(tmp_path: Path, name: str) -> None:
    (tmp_path / "empty").mkdir()
    directory = tmp_path / name
    message = f"{directory.resolve()} is not a run directory: it has no {RUN_CONFIG}"
    with pytest.raises(ControlError) as command:
        send_command(directory, "stop")
    with pytest.raises(ControlError) as changes:
        send_changes(directory, {"optimizer.lr": 1e-3})
    assert str(command.value) == str(changes.value) == message
    assert not (tmp_path / "typo").exists()


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        (
            {"optimizer.momentum": 0.9},
            "optimizer.momentum is not tunable. Tunable: optimizer.lr, rollout.steps",
        ),
        ({"optimizer.lr": 0.5}, r"optimizer.lr=0.5 is above the bound 0.01"),
        ({"optimizer.lr": -1}, r"optimizer.lr=-1 is below the bound 0.0"),
    ],
)
def test_changes_are_checked_against_the_run_config(
    tmp_path: Path, changes: dict[str, Any], match: str
) -> None:
    with pytest.raises(ControlError, match=match):
        send_changes(tmp_path, {"rollout.steps": 8, **changes})
    assert not (tmp_path / "control.jsonl").exists()
    # Types are left to the run, which knows the current values.
    assert send_changes(tmp_path, {"optimizer.lr": "fast", "rollout.steps": 8}) == 1


def test_partial_last_line_is_read_once_complete(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    send_command(tmp_path, "pause")
    with path.open("a", encoding="utf-8") as file:
        file.write('{"id": 2, "command": "res')
    control = FileControl(tmp_path)
    assert [message.id for message in control.poll()] == [1]
    with path.open("a", encoding="utf-8") as file:
        file.write('ume"}\n')
    assert control.poll() == [ControlMessage(2, command="resume")]


def test_writer_recovers_from_a_truncated_line(tmp_path: Path) -> None:
    path = tmp_path / "control.jsonl"
    path.write_text('{"id": 1, "command": "pause"}\n{"id": 2, "comm', encoding="utf-8")
    assert send_command(tmp_path, "resume") == 2
    control = FileControl(tmp_path)
    assert [(message.id, message.command) for message in control.poll()] == [
        (1, "pause"),
        (2, "resume"),
    ]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["id"] == 2


def test_empty_messages_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ControlError, match="no command"):
        send_command(tmp_path, " ")
    with pytest.raises(ControlError):
        send_changes(tmp_path, {})
    assert not (tmp_path / "control.jsonl").exists()


def test_custom_commands_are_sent(tmp_path: Path) -> None:
    send_command(tmp_path, "evaluate")
    assert FileControl(tmp_path).poll() == [ControlMessage(1, command="evaluate")]


@pytest.fixture
def tunables() -> Tunables:
    values: dict[str, Any] = {
        "optimizer.lr": 3e-4,
        "rollout.steps": 64,
        "eval.render": False,
        "mode": "train",
    }
    return Tunables(values, {"optimizer.lr": (0.0, 1e-2), "rollout.steps": (1, None)})


def test_apply_returns_old_and_new(tunables: Tunables) -> None:
    changed = tunables.apply({"optimizer.lr": 0, "rollout.steps": 128})
    assert changed == {"optimizer.lr": (3e-4, 0.0), "rollout.steps": (64, 128)}
    assert isinstance(tunables.values["optimizer.lr"], float)
    assert tunables.values["rollout.steps"] == 128


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"optimizer.momentum": 0.9}, "not tunable.*optimizer.lr"),
        ({"rollout.steps": 1.5}, "int"),
        ({"rollout.steps": True}, "int"),
        ({"eval.render": 1}, "bool"),
        ({"mode": 3}, "str"),
        ({"optimizer.lr": "fast"}, "float"),
        ({"optimizer.lr": 0.5}, "above"),
        ({"rollout.steps": 0}, "below"),
    ],
)
def test_refusals(tunables: Tunables, changes: dict[str, Any], match: str) -> None:
    before = tunables.values
    with pytest.raises(ControlError, match=match):
        tunables.apply({"optimizer.lr": 1e-3, **changes})
    assert tunables.values == before


def test_bounds_must_name_tunables() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Tunables({"a": 1}, {"b": (0, 1)})
