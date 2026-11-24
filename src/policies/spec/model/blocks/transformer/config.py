"""Transformer block options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.blocks.registry import blocks
from policies.spec.model.blocks.transformer.pooling import Pooling
from policies.spec.model.blocks.transformer.tokens import Tokens


class TransformerBlockConfig(Config):
    width: int = Field(64, ge=1)
    heads: int = Field(4, ge=1)
    layers: int = Field(2, ge=1)
    feedforward: int = Field(128, ge=1)
    """Hidden size of each layer's feedforward."""
    activation: str = "gelu"
    tokens: Tokens = Tokens.PARTS
    memory: int = Field(0, ge=0)
    """Past steps attended to, kept as recurrent state. 0 makes the block stateless."""
    causal: bool = False
    """Whether tokens only see their own and earlier steps."""
    positions: bool = True
    """Whether to add learned embeddings per (step, token)."""
    pooling: Pooling = Pooling.MEAN
    after: list[int] = Field(default_factory=list)
    """MLP sizes after pooling."""


blocks.register_lazy(
    "transformer", TransformerBlockConfig, "policies.models.blocks.transformer:TransformerBlock"
)
