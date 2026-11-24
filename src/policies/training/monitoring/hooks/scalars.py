"""Numeric values out of event data."""

from typing import Any


def scalars(data: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Numeric leaves of nested data, keyed by slash-joined paths."""
    out: dict[str, float] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(scalars(value, f"{path}/"))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            out[path] = float(value)
    return out
