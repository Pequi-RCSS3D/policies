"""Splits a Gymnasium space into named leaf spaces."""

from typing import Any

from gymnasium import spaces


def leaves(
    space: spaces.Space[Any], name: str, supported: tuple[type[spaces.Space[Any]], ...]
) -> dict[str, spaces.Space[Any]]:
    """A Dict space's entries, or the space itself under `name`. Leaves must be `supported`."""
    found = dict(space.spaces) if isinstance(space, spaces.Dict) else {name: space}
    for key, leaf in found.items():
        if not isinstance(leaf, supported):
            expected = ", ".join(kind.__name__ for kind in supported)
            raise TypeError(f"part {key!r} has unsupported space {leaf}, expected {expected}")
    return found
