"""Flattens a part, then applies any block, e.g. a recurrent per-part encoder."""

from torch import Tensor

from policies.models.encoders.base import Encoder
from policies.models.encoders.flatten import Flatten
from policies.models.state import State, namespaced, run_scoped, run_scoped_sequence
from policies.spec.model.blocks import blocks
from policies.spec.model.encoders import BlockEncoderConfig, FlattenConfig


class BlockEncoder(Encoder):
    """State is the block's, under `block.`."""

    def __init__(self, config: BlockEncoderConfig, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.flatten = Flatten(FlattenConfig(), shape)
        self.block = blocks.build(config.block, in_sizes=(self.flatten.out_size,))
        self.out_size = self.block.out_size
        self.state_shapes = namespaced("block", self.block.state_shapes)

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        flat, _ = self.flatten(x, {})
        return run_scoped(self.block, "block", flat, state)

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        flat, _ = self.flatten(x, {})
        return run_scoped_sequence(self.block, "block", flat, state, reset)
