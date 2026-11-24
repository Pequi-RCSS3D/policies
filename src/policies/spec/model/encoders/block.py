"""Block encoder options."""

from policies.core.config import Config
from policies.spec.model.blocks import BlockConfig
from policies.spec.model.encoders.registry import encoders


class BlockEncoderConfig(Config):
    block: BlockConfig
    """Config of the block applied to the flattened part, e.g. `{type: gru, hidden_size: 32}`."""


encoders.register_lazy("block", BlockEncoderConfig, "policies.models.encoders.block:BlockEncoder")
