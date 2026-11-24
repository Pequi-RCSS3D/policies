"""Hidden layers, then a linear output with small initial values."""

from torch import Tensor, nn

from policies.models.heads.base import Head
from policies.models.layers import mlp, small_linear
from policies.spec.model.heads import MLPHeadConfig


class MLPHead(Head):
    def __init__(self, config: MLPHeadConfig, in_size: int, out_size: int) -> None:
        super().__init__()
        self.output = nn.Sequential(
            mlp([in_size, *config.hidden], config.activation),
            small_linear(
                config.hidden[-1] if config.hidden else in_size, out_size, config.initial_scale
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        output: Tensor = self.output(x)
        return output
