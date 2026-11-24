"""Built-in activation factories. The registry lives in `policies.spec.model.activations`."""

from typing import Any

from torch import nn

from policies.spec.model.activations import ActivationConfig


def elu(config: ActivationConfig, **_: Any) -> nn.Module:
    return nn.ELU()


def relu(config: ActivationConfig, **_: Any) -> nn.Module:
    return nn.ReLU()


def gelu(config: ActivationConfig, **_: Any) -> nn.Module:
    return nn.GELU()


def silu(config: ActivationConfig, **_: Any) -> nn.Module:
    return nn.SiLU()


def tanh(config: ActivationConfig, **_: Any) -> nn.Module:
    return nn.Tanh()
