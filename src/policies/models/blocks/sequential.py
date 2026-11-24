"""Blocks in order, each fed the previous one's output."""

from typing import cast

from torch import Tensor, nn

from policies.models.blocks.base import Block
from policies.models.state import State, namespaced, run_scoped, run_scoped_sequence
from policies.spec.model.blocks import SequentialBlockConfig, blocks


class SequentialBlock(Block):
    """The first block sees the encoded parts, the rest one vector. State is `blocks.<i>.<name>`."""

    def __init__(self, config: SequentialBlockConfig, in_sizes: tuple[int, ...]) -> None:
        super().__init__()
        built: list[Block] = []
        for block_config in config.blocks:
            built.append(blocks.build(block_config, in_sizes=in_sizes))
            in_sizes = (built[-1].out_size,)
        self.blocks = nn.ModuleList(built)
        self.prefixes = [f"blocks.{i}" for i in range(len(built))]
        self.out_size = built[-1].out_size
        self.state_shapes = {
            name: shape
            for prefix, block in zip(self.prefixes, built, strict=True)
            for name, shape in namespaced(prefix, block.state_shapes).items()
        }

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        next_state: State = {}
        for prefix, block in zip(self.prefixes, self.blocks, strict=True):
            x, inner = run_scoped(cast(Block, block), prefix, x, state)
            next_state |= inner
        return x, next_state

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        for prefix, block in zip(self.prefixes, self.blocks, strict=True):
            x = run_scoped_sequence(cast(Block, block), prefix, x, state, reset)
        return x
