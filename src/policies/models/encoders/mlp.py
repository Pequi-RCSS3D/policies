"""Flattens a part, then applies an MLP."""

from torch import Tensor

from policies.models.encoders.base import Encoder
from policies.models.encoders.flatten import Flatten
from policies.models.layers import mlp
from policies.models.state import State
from policies.spec.model.encoders import FlattenConfig, MLPEncoderConfig


class MLPEncoder(Encoder):
    def __init__(self, config: MLPEncoderConfig, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.flatten = Flatten(FlattenConfig(), shape)
        self.net = mlp([self.flatten.out_size, *config.hidden], config.activation)
        self.out_size = config.hidden[-1]
        self.state_shapes = {}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        flat, _ = self.flatten(x, {})
        return self.net(flat), {}
