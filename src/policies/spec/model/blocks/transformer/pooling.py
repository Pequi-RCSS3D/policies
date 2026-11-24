"""How the current step's tokens become one vector."""

from enum import StrEnum


class Pooling(StrEnum):
    MEAN = "mean"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    SUMMARY = "summary"
    """A learned extra token that attends to the others."""
