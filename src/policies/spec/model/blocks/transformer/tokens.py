"""How a step's input becomes tokens."""

from enum import StrEnum


class Tokens(StrEnum):
    PARTS = "parts"
    """One token per encoded input part, each with its own projection."""
    INPUT = "input"
    """The whole input as one token."""
