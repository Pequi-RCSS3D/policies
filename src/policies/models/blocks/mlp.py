"""A feedforward block."""

from torch import Tensor

from policies.models.blocks.base import Block
from policies.models.layers import mlp
from policies.models.state import State
from policies.spec.model.blocks import MLPBlockConfig


class MLPBlock(Block):
    def __init__(self, config: MLPBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.net = mlp([sum(in_sizes), *config.hidden], config.activation)
        self.out_size = config.hidden[-1] if config.hidden else sum(in_sizes)
        self.state_shapes = {}

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        return self.net(x), {}
