"""The activation registry, importable without torch. Register more with `activations.register`."""

from typing import TYPE_CHECKING

from policies.core.config import Config
from policies.core.registry import Registry

if TYPE_CHECKING:
    from torch import nn

activations: "Registry[nn.Module]" = Registry("activation")


class ActivationConfig(Config):
    pass


for _name in ("elu", "relu", "gelu", "silu", "tanh"):
    activations.register_lazy(_name, ActivationConfig, f"policies.models.activations:{_name}")
