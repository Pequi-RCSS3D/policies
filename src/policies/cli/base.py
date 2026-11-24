"""What every command group shares."""

import argparse

type Subcommands = argparse._SubParsersAction[argparse.ArgumentParser]
"""Where a group adds its commands."""


class CliError(Exception):
    """A user error, printed without a traceback."""
