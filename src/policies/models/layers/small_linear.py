"""A linear layer with small initial outputs, for heads."""

from torch import nn


def small_linear(in_size: int, out_size: int, initial_scale: float) -> nn.Linear:
    """Orthogonal weights of gain `initial_scale` and zero bias."""
    layer = nn.Linear(in_size, out_size)
    nn.init.orthogonal_(layer.weight, gain=initial_scale)
    nn.init.zeros_(layer.bias)
    return layer
