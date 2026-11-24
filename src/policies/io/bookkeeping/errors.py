"""Bookkeeping errors."""

from collections.abc import Iterable


class UnknownSourceError(KeyError):
    """A tracker reads a source whose shape is not known."""

    def __init__(self, source: str, known: Iterable[str]) -> None:
        super().__init__(source)
        self.source = source
        self.known = sorted(known)

    def __str__(self) -> str:
        return f"bookkeeping source {self.source!r} is not known. Known: {self.known}"
