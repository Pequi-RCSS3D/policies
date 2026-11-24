from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from policies.data import InMemoryDataset

BACKENDS: dict[str, Callable[[np.ndarray], Any]] = {
    "numpy": lambda value: value,
    "torch": torch.from_numpy,
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_samples_cover_every_step_once(backend: str) -> None:
    to = BACKENDS[backend]
    steps = np.arange(10, dtype=np.float32)
    dataset = InMemoryDataset({"x": to(steps[:, None])}, {"y": to(steps)}, batch_size=4)
    batches = list(dataset)
    assert len(batches) == len(dataset) == 3
    assert all(batch.reset is None for batch in batches)
    seen = np.concatenate([np.asarray(batch.targets["y"]) for batch in batches])
    assert sorted(seen) == list(steps)
    assert all(np.array_equal(b.inputs["x"][:, 0], b.targets["y"]) for b in batches)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sequences_are_windows_with_resets(backend: str) -> None:
    to = BACKENDS[backend]
    steps = np.arange(23, dtype=np.float32)
    reset = steps % 7 == 0
    dataset = InMemoryDataset(
        {"x": to(steps[:, None])}, {"y": to(steps)}, reset=reset, batch_size=2, sequence_length=5
    )
    for _ in range(3):
        batches = list(dataset)
        assert len(batches) == len(dataset) == 2
        for batch in batches:
            y = np.asarray(batch.targets["y"])
            assert batch.inputs["x"].shape == (5, *y.shape[1:], 1)
            assert (np.diff(y, axis=0) == 1).all()
            assert batch.reset[0].all()
            np.testing.assert_array_equal(batch.reset[1:], y[1:] % 7 == 0)


def test_parts_need_equal_lengths() -> None:
    with pytest.raises(ValueError, match="same number"):
        InMemoryDataset({"x": np.zeros(3)}, {"y": np.zeros(4)})
