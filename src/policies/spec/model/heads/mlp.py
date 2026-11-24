"""MLP head options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.heads.registry import heads


class MLPHeadConfig(Config):
    hidden: list[int] = Field(default_factory=lambda: [64])
    activation: str = "elu"
    initial_scale: float = Field(0.01, gt=0)
    """Of the output layer."""


heads.register_lazy("mlp", MLPHeadConfig, "policies.models.heads.mlp:MLPHead")
