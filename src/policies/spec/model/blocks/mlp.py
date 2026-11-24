"""Feedforward block options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.blocks.registry import blocks


class MLPBlockConfig(Config):
    hidden: list[int] = Field(default_factory=lambda: [256, 128])
    """Layer sizes. Empty passes the input through."""
    activation: str = "elu"


blocks.register_lazy("mlp", MLPBlockConfig, "policies.models.blocks.mlp:MLPBlock")
