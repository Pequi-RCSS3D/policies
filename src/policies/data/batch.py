"""One batch of recorded data."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Batch:
    """Named input and target parts, as NumPy or PyTorch arrays.

    Samples are `[batch, ...]`. Sequences are `[time, batch, ...]` and carry `reset`.
    Inputs are policy input parts before normalization. Targets are in action space: means
    for continuous parts, indices for categorical ones.
    """

    inputs: Mapping[str, Any]
    targets: Mapping[str, Any]
    reset: Any = None
    """`[time, batch]` flags, true where a step starts an episode. None for samples."""
