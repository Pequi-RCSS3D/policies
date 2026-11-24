"""Errors raised while registering, discovering and looking up components."""

from collections.abc import Sequence
from difflib import get_close_matches


class UnknownComponentError(KeyError):
    """No component is registered under the requested name."""

    def __str__(self) -> str:
        return str(self.args[0])


class DuplicateComponentError(ValueError):
    """A name is registered twice without `replace=True`."""


class PluginError(RuntimeError):
    """An installed plugin failed to load."""


def unknown(kind: str, name: str, known: Sequence[str]) -> UnknownComponentError:
    """The error for `name`, with the closest known name as a hint."""
    hint = get_close_matches(name, known, n=1)
    suggestion = f" Did you mean {hint[0]!r}?" if hint else ""
    return UnknownComponentError(
        f"unknown {kind} {name!r}.{suggestion} Known: {', '.join(known) or 'none'}."
    )
