"""The model contract. The registry lives in `policies.spec.model`."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from policies.models.sequence import restart
from policies.models.state import State


class Model(nn.Module):
    """Built from `(config, in_shapes, out_sizes)`: input part shapes and output part sizes.

    `forward(parts, state) -> (raw outputs, next state)`, pure: all state goes in and out.
    Parts are `(*batch, *shape)` and outputs `(*batch, size)`. `state_shapes` holds the full,
    exported names. Auxiliary losses are collected with `AuxiliaryLosses(model)`.
    Training runs `forward_sequence`, which may be overridden to run sequences natively.
    """

    state_shapes: dict[str, tuple[int, ...]]

    def initial_state(self, batch: tuple[int, ...], device: Any = None) -> State:
        return {k: torch.zeros(*batch, *s, device=device) for k, s in self.state_shapes.items()}

    def forward(self, parts: Mapping[str, Tensor], state: State) -> tuple[dict[str, Tensor], State]:
        raise NotImplementedError

    def forward_sequence(
        self, parts: Mapping[str, Tensor], state: State, reset: Tensor
    ) -> dict[str, Tensor]:
        """Parts (time, *batch, *shape) from `state` (*batch, ...) to outputs (time, *batch, size).

        `reset` (time, *batch) zeroes the state before a step. Must match stepping `forward`.
        The default steps through time, or runs at once if stateless.
        """
        raw: dict[str, Tensor]
        if not self.state_shapes:
            raw, _ = self(parts, {})
            return raw
        outputs = []
        for t in range(reset.shape[0]):
            raw, state = self(
                {name: value[t] for name, value in parts.items()}, restart(state, reset[t])
            )
            outputs.append(raw)
        return {name: torch.stack([output[name] for output in outputs]) for name in outputs[0]}
