"""Check an exported artifact: `validate`."""

import argparse
from importlib import import_module
from pathlib import Path

from policies.cli.base import CliError, Subcommands
from policies.cli.console import console


def _validate(arguments: argparse.Namespace) -> None:
    path = Path(arguments.path)
    try:
        result = import_module("policies.deploy.inference").validate_artifact(path)
    except Exception as error:
        raise CliError(f"{path.resolve()}: {error}") from error
    if result.parity:
        console().print("OK, ONNX Runtime matches PyTorch")
    else:
        console().print(f"OK, parity with PyTorch skipped: {result.skipped}")


def add_commands(commands: Subcommands) -> None:
    subparser = commands.add_parser("validate", help="check an artifact directory runs")
    subparser.add_argument("path")
    subparser.set_defaults(handler=_validate)
