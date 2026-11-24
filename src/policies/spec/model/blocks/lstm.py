"""Long short-term memory block options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.blocks.registry import blocks


class LSTMBlockConfig(Config):
    hidden_size: int = Field(128, ge=1)
    layers: int = Field(1, ge=1)
    after: list[int] = Field(default_factory=list)
    """MLP sizes after the recurrence."""
    activation: str = "elu"


blocks.register_lazy("lstm", LSTMBlockConfig, "policies.models.blocks.lstm:LSTMBlock")
