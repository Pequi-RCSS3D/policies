"""Steer a running job: `set` and the run commands."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from policies.cli.base import CliError, Subcommands
from policies.cli.console import console
from policies.training.control import Command, send_changes, send_command


def _value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _pairs(items: Sequence[str]) -> dict[str, Any]:
    changes = {}
    for item in items:
        key, separator, text = item.partition("=")
        if not separator or not key:
            raise CliError(f"expected key=value, got {item!r}")
        changes[key] = _value(text)
    return changes


def _set(arguments: argparse.Namespace) -> None:
    message = send_changes(Path(arguments.run_directory), _pairs(arguments.changes))
    console().print(f"sent #{message}")


def _command(arguments: argparse.Namespace) -> None:
    message = send_command(Path(arguments.run_directory), arguments.command)
    console().print(f"sent #{message} {arguments.command}")


def add_commands(commands: Subcommands) -> None:
    subparser = commands.add_parser("set", help="change tunable values of a run")
    subparser.add_argument("run_directory")
    subparser.add_argument("changes", nargs="+", metavar="key=value")
    subparser.set_defaults(handler=_set)

    for name in Command:
        subparser = commands.add_parser(name, help=f"ask a run to {name}")
        subparser.add_argument("run_directory")
        subparser.set_defaults(handler=_command, command=name)

    subparser = commands.add_parser("send", help="send a custom command to a run")
    subparser.add_argument("run_directory")
    subparser.add_argument("command", help="a name in the run's `trainer.commands`")
    subparser.set_defaults(handler=_command)
