"""Encoders per input part, one block, heads per output part."""

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn

from policies.models.base import Model
from policies.models.blocks import blocks
from policies.models.encoders import Encoder, encoders
from policies.models.heads import heads
from policies.models.state import State, namespaced, run_scoped, run_scoped_sequence
from policies.spec.model import PartsModelConfig


class PartsModel(Model):
    """The encoded parts, concatenated in input order, feed the block, which feeds every head.

    State is `encoders.<part>.<name>` and `block.<name>`.
    """

    def __init__(
        self,
        spec: PartsModelConfig,
        in_shapes: Mapping[str, tuple[int, ...]],
        out_sizes: Mapping[str, int],
    ) -> None:
        super().__init__()
        self.in_names = list(in_shapes)
        self.out_names = list(out_sizes)
        built: dict[str, Encoder] = {
            name: encoders.build(spec.encoders.get(name, spec.encoder), shape=shape)
            for name, shape in in_shapes.items()
        }
        self.encoders = nn.ModuleDict(built)
        self.prefixes = {name: f"encoders.{name}" for name in self.in_names}
        in_sizes = tuple(encoder.out_size for encoder in built.values())
        self.block = blocks.build(spec.block, in_sizes=in_sizes)
        self.heads = nn.ModuleDict(
            {
                name: heads.build(
                    spec.heads.get(name, spec.head),
                    in_size=self.block.out_size,
                    out_size=size,
                )
                for name, size in out_sizes.items()
            }
        )
        self.state_shapes = {
            **{
                name: shape
                for part, encoder in built.items()
                for name, shape in namespaced(self.prefixes[part], encoder.state_shapes).items()
            },
            **namespaced("block", self.block.state_shapes),
        }

    def forward(self, parts: Mapping[str, Tensor], state: State) -> tuple[dict[str, Tensor], State]:
        encoded: list[Tensor] = []
        next_state: State = {}
        for name in self.in_names:
            y, inner = run_scoped(
                cast(Encoder, self.encoders[name]), self.prefixes[name], parts[name], state
            )
            encoded.append(y)
            next_state |= inner
        y, inner = run_scoped(self.block, "block", torch.cat(encoded, dim=-1), state)
        next_state |= inner
        return {name: self.heads[name](y) for name in self.out_names}, next_state

    def forward_sequence(
        self, parts: Mapping[str, Tensor], state: State, reset: Tensor
    ) -> dict[str, Tensor]:
        """Each component runs the whole sequence, natively where it can."""
        encoded = [
            run_scoped_sequence(
                cast(Encoder, self.encoders[name]), self.prefixes[name], parts[name], state, reset
            )
            for name in self.in_names
        ]
        y = run_scoped_sequence(self.block, "block", torch.cat(encoded, dim=-1), state, reset)
        return {name: self.heads[name](y) for name in self.out_names}
