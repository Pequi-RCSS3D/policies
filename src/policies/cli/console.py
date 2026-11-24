"""Rich output that stays plain when piped."""

import sys

from rich.console import Console
from rich.table import Table


def console(stderr: bool = False) -> Console:
    """Rich output without markup. Unbounded when piped, so one row stays one line."""
    stream = sys.stderr if stderr else sys.stdout
    return Console(
        file=stream, width=None if stream.isatty() else 100_000, markup=False, highlight=False
    )


def table(*columns: str, header: bool = True) -> Table:
    """Borderless aligned columns, so piped output stays plain and splittable."""
    return Table(*columns, box=None, pad_edge=False, show_header=header, header_style="bold")


def print_table(rows: Table) -> None:
    """Fits a terminal. Piped, each line ends at its content instead of the widest row."""
    output = console()
    if output.is_terminal:
        output.print(rows)
        return
    with output.capture() as captured:
        output.print(rows)
    output.file.write("".join(line.rstrip() + "\n" for line in captured.get().splitlines()))
