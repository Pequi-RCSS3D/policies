"""A dataset held in memory as one stream of recorded steps."""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from array_api_compat import array_namespace, device

from policies.data.batch import Batch


def _take(value: Any, indices: np.ndarray) -> Any:
    """Rows of `value` at `indices`, reshaped to the shape of `indices`."""
    xp = array_namespace(value)
    flat = xp.asarray(indices.reshape(-1), device=device(value))
    return xp.reshape(xp.take(value, flat, axis=0), (*indices.shape, *value.shape[1:]))


class InMemoryDataset:
    """Steps `[steps, ...]` per part, episodes back to back, in NumPy or PyTorch arrays.

    Without `sequence_length`, yields shuffled samples. With it, yields `[time, batch]`
    windows cut at a random offset each epoch. Windows may cross episodes, and `reset`
    marks where one starts. Their first step always resets.
    """

    def __init__(
        self,
        inputs: Mapping[str, Any],
        targets: Mapping[str, Any],
        *,
        reset: Any = None,
        batch_size: int = 256,
        sequence_length: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """`reset` is `[steps]`, true at each episode's first step. `batch_size` counts
        samples, or windows when `sequence_length` is set."""
        self.inputs, self.targets = dict(inputs), dict(targets)
        lengths = {len(value) for value in [*self.inputs.values(), *self.targets.values()]}
        if len(lengths) != 1:
            raise ValueError(f"every part needs the same number of steps, got {sorted(lengths)}")
        self.steps = lengths.pop()
        if sequence_length is not None and not 0 < sequence_length <= self.steps:
            raise ValueError(f"sequence_length must be in 1..{self.steps}, got {sequence_length}")
        self.reset = np.zeros(self.steps, bool) if reset is None else np.asarray(reset, bool)
        self.batch_size, self.sequence_length = batch_size, sequence_length
        self.shuffle = shuffle
        self.generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Batches per epoch."""
        units = self.steps if self.sequence_length is None else self.steps // self.sequence_length
        return -(-units // self.batch_size)

    def _order(self, count: int) -> np.ndarray:
        return self.generator.permutation(count) if self.shuffle else np.arange(count)

    def _batch(self, indices: np.ndarray, reset: np.ndarray | None) -> Batch:
        return Batch(
            {name: _take(value, indices) for name, value in self.inputs.items()},
            {name: _take(value, indices) for name, value in self.targets.items()},
            reset,
        )

    def __iter__(self) -> Iterator[Batch]:
        length = self.sequence_length
        if length is None:
            order = self._order(self.steps)
            for start in range(0, self.steps, self.batch_size):
                yield self._batch(order[start : start + self.batch_size], None)
            return
        offset = int(self.generator.integers(self.steps % length + 1)) if self.shuffle else 0
        windows = offset + length * self._order((self.steps - offset) // length)
        for start in range(0, len(windows), self.batch_size):
            # `[time, batch]` step indices of the chosen windows.
            indices = windows[start : start + self.batch_size] + np.arange(length)[:, None]
            reset = self.reset[indices]
            reset[0] = True
            yield self._batch(indices, reset)
