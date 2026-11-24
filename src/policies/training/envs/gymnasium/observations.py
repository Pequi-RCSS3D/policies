"""Batched Gymnasium observations as named parts."""

from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector.utils import concatenate, create_empty_array

from policies.training.envs.base import Observation
from policies.training.envs.gymnasium.spaces import leaves

SUPPORTED = (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary)


class Observations:
    """One part per Dict entry, or one part named `name`. Floating values become float32.

    Tensors share memory with the vector env's arrays, so keep its default `copy=True`.
    """

    def __init__(self, space: spaces.Space[Any], name: str) -> None:
        self.space = space
        self.names = tuple(leaves(space, name, SUPPORTED))
        self.single = not isinstance(space, spaces.Dict)

    def __call__(self, observation: Any) -> Observation:
        values = {self.names[0]: observation} if self.single else observation
        parts = {name: torch.as_tensor(values[name]) for name in self.names}
        return {
            name: value.float() if value.is_floating_point() else value
            for name, value in parts.items()
        }

    def replace(
        self, parts: Observation, items: np.ndarray[Any, Any], mask: np.ndarray[Any, Any]
    ) -> Observation:
        """A copy of `parts` with the rows in `mask` taken from `items`, single observations."""
        index = np.flatnonzero(mask)
        empty = create_empty_array(self.space, len(index))
        assert empty is not None  # None only for spaces `leaves` refuses
        batch = concatenate(self.space, items[index], empty)
        rows, replaced = torch.as_tensor(index), self(batch)
        out = {name: value.clone() for name, value in parts.items()}
        for name, value in out.items():
            value[rows] = replaced[name].to(value.dtype)
        return out
