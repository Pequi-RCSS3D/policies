"""Config of the `parts` model: which encoders, block and heads it is built from."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.blocks import BlockConfig
from policies.spec.model.encoders import EncoderConfig
from policies.spec.model.heads import HeadConfig


class PartsModelConfig(Config):
    """One encoder per input part, a block, one head per output part."""

    encoder: EncoderConfig = Field(default_factory=lambda: {"type": "flatten"})
    """For input parts not in `encoders`."""
    encoders: dict[str, EncoderConfig] = Field(default_factory=dict)
    block: BlockConfig = Field(default_factory=lambda: {"type": "mlp"})
    head: HeadConfig = Field(default_factory=lambda: {"type": "linear"})
    """For output parts not in `heads`."""
    heads: dict[str, HeadConfig] = Field(default_factory=dict)
