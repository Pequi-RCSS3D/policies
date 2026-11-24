"""Sequential block options."""

from pydantic import Field

from policies.core.config import Config
from policies.spec.model.blocks.registry import BlockConfig, blocks


class SequentialBlockConfig(Config):
    blocks: list[BlockConfig] = Field(min_length=1)
    """Configs of the blocks, in order, e.g. `[{type: mlp}, {type: gru}, {type: mlp}]`."""


blocks.register_lazy(
    "sequential", SequentialBlockConfig, "policies.models.blocks.sequential:SequentialBlock"
)
