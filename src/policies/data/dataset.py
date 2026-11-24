"""The dataset contract."""

from collections.abc import Iterator
from typing import Protocol

from policies.data.batch import Batch


class Dataset(Protocol):
    """Batches for one epoch per iteration. Iterating again starts a new epoch."""

    def __iter__(self) -> Iterator[Batch]: ...
