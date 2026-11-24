"""A linear head with small initial outputs."""

from torch import nn

from policies.models.layers import small_linear
from policies.spec.model.heads import LinearHeadConfig


def linear_head(config: LinearHeadConfig, in_size: int, out_size: int) -> nn.Linear:
    return small_linear(in_size, out_size, config.initial_scale)
