"""Multilayer perceptrons."""

from torch import nn

from policies.spec.model.activations import activations


def mlp(sizes: list[int], activation: str = "elu", last_activation: bool = True) -> nn.Sequential:
    """Linear layers of the given sizes, with an activation after each (optionally not the last)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if last_activation or i < len(sizes) - 2:
            layers.append(activations.build({"type": activation}))
    return nn.Sequential(*layers)
