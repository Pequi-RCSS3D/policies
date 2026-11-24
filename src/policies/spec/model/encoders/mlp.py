"""MLP encoder options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.encoders.registry import encoders


class MLPEncoderConfig(Config):
    hidden: list[int] = Field(min_length=1)
    activation: str = "elu"


encoders.register_lazy("mlp", MLPEncoderConfig, "policies.models.encoders.mlp:MLPEncoder")
