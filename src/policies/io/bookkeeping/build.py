"""Instantiating bookkeeping trackers for a batch on an array namespace."""

from collections.abc import Mapping, Sequence
from typing import Any

from policies.io.bookkeeping.base import TrackerContext, trackers
from policies.io.bookkeeping.compose import Compose


def build_bookkeeping(
    configs: Sequence[Mapping[str, Any]],
    *,
    shapes: Mapping[str, tuple[int, ...]],
    batch: tuple[int, ...],
    xp: Any,
    dtypes: Mapping[str, str] | None = None,
    device: Any = None,
) -> Compose:
    """Instantiate trackers for a batch on the given array namespace.

    `shapes` and `dtypes` describe the sources: observations and output actions.
    """
    context = TrackerContext(shapes, batch, xp, dict(dtypes or {}), device)
    return Compose([trackers.build(config, context=context) for config in configs])
