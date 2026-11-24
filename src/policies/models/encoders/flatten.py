"""Flattens a part into a vector."""

import math

from torch import Tensor

from policies.models.encoders.base import Encoder
from policies.models.state import State
from policies.spec.model.encoders import FlattenConfig


class Flatten(Encoder):
    def __init__(self, config: FlattenConfig, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.ndim = len(shape)
        self.out_size = math.prod(shape)
        self.state_shapes = {}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        return x.reshape(*x.shape[: x.dim() - self.ndim], self.out_size).float(), {}
