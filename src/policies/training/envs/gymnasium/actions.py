"""Output parts as a batched Gymnasium action."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from torch import Tensor

from policies.training.envs.gymnasium.spaces import leaves

SUPPORTED = (spaces.Box, spaces.Discrete)


class Actions:
    """Box parts take continuous outputs, Discrete parts take categorical indices."""

    def __init__(self, space: spaces.Space[Any], num_envs: int, name: str) -> None:
        self.leaves = leaves(space, name, SUPPORTED)
        self.single = not isinstance(space, spaces.Dict)
        self.num_envs = num_envs

    def _convert(self, value: Tensor, leaf: spaces.Space[Any]) -> np.ndarray[Any, Any]:
        array = value.detach().cpu().numpy()
        if isinstance(leaf, spaces.Discrete):
            indices: np.ndarray[Any, Any] = array.reshape(self.num_envs) + int(leaf.start)
            return indices.astype(leaf.dtype)
        assert leaf.shape is not None  # Box spaces always have a shape
        return array.reshape(self.num_envs, *leaf.shape).astype(leaf.dtype)

    def __call__(self, outputs: Mapping[str, Tensor]) -> Any:
        actions = {name: self._convert(outputs[name], leaf) for name, leaf in self.leaves.items()}
        return next(iter(actions.values())) if self.single else actions
