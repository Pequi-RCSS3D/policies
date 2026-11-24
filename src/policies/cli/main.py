"""The `pm` command: steer runs, manage artifacts and list components."""

import argparse
from collections.abc import Sequence

from rich.text import Text

from policies.cli import artifacts, components, control, validate
from policies.cli.base import CliError
from policies.cli.console import console
from policies.core.logs import configure
from policies.core.registry import PluginError


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="pm", description="Steer runs, manage artifacts and list components."
    )
    commands = root.add_subparsers(dest="subcommand", required=True)
    control.add_commands(commands)
    artifacts.add_commands(commands)
    validate.add_commands(commands)
    components.add_commands(commands)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    configure()
    try:
        arguments.handler(arguments)
    except (CliError, PluginError, KeyError, ValueError, OSError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        console(stderr=True).print(Text.assemble(("error:", "bold red"), f" {message}"))
        return 1
    return 0
