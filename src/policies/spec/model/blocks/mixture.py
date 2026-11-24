"""Mixture of experts block options."""

from typing import Self

from pydantic import Field, model_validator

from policies.core.config import Config
from policies.spec.model.blocks.registry import BlockConfig, blocks


class MixtureBlockConfig(Config):
    experts: int = Field(4, ge=1)
    expert: BlockConfig = Field(default_factory=lambda: {"type": "mlp"})
    """Config of each expert, any block."""
    gate: list[int] = Field(default_factory=list)
    """Hidden sizes of the gate. Empty makes it linear."""
    top_k: int | None = Field(None, ge=1)
    """Experts mixed per sample, with renormalized weights. None mixes all."""
    activation: str = "elu"
    """Of the gate's hidden layers."""

    @model_validator(mode="after")
    def _top_k_within_experts(self) -> Self:
        if self.top_k is not None and self.top_k > self.experts:
            raise ValueError(f"top_k {self.top_k} exceeds {self.experts} experts")
        return self


blocks.register_lazy("mixture", MixtureBlockConfig, "policies.models.blocks.mixture:MixtureBlock")
