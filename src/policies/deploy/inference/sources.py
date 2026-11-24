"""Which observations a spec's bookkeeping reads."""

from collections.abc import Mapping
from typing import Any

import array_api_compat.numpy as numpy_xp

from policies.io.bookkeeping import UnknownSourceError, build_bookkeeping
from policies.io.outputs import OutputDecoder


class MissingObservationsError(KeyError):
    """Bookkeeping reads observations whose shapes are not known."""

    def __init__(self, missing: list[str]) -> None:
        super().__init__(missing)
        self.missing = missing

    def __str__(self) -> str:
        return f"bookkeeping reads observations of unknown shape: {self.missing}"


def tracker_sources(
    bookkeeping: tuple[Mapping[str, Any], ...],
    decoder: OutputDecoder,
    observation_shapes: Mapping[str, tuple[int, ...]],
    observation_dtypes: Mapping[str, str],
) -> tuple[str, ...]:
    """Observation keys the trackers declare in `sources`. Outputs shadow observations.

    Raises `MissingObservationsError` when a tracker reads an observation of unknown shape.
    """
    try:
        built = build_bookkeeping(
            bookkeeping,
            shapes={**observation_shapes, **decoder.shapes},
            dtypes={**observation_dtypes, **decoder.dtypes},
            batch=(),
            xp=numpy_xp,
        )
    except UnknownSourceError as error:
        raise MissingObservationsError([error.source]) from None
    observations = [name for name in built.sources if name not in decoder.shapes]
    missing = [name for name in observations if name not in observation_shapes]
    if missing:
        raise MissingObservationsError(missing)
    return tuple(observations)
