"""The value function: its own inputs, normalization and model."""

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from policies.io.inputs import Inputs
from policies.learning.normalizer import Normalizer
from policies.learning.rollout import Steps
from policies.learning.sequences import group, unroll
from policies.models import Model, State, models

VALUE = "value"
"""The critic model's one output part, of size 1."""


class Critic(nn.Module):
    """Values per sample from critic input parts, which may be privileged."""

    def __init__(
        self, inputs: Inputs, model: Mapping[str, Any], normalizer_clip: float = 5.0
    ) -> None:
        super().__init__()
        self.inputs = inputs.compile()
        """Critic parts from an observation and bookkeeping slots."""
        normalized = {part.name: part.shape for part in inputs.parts if part.normalize}
        self.normalizer = Normalizer(normalized, normalizer_clip)
        shapes = {part.name: part.shape for part in inputs.parts}
        self.model: Model = models.build(model, in_shapes=shapes, out_sizes={VALUE: 1})

    def forward(self, parts: Mapping[str, Tensor], state: State) -> tuple[Tensor, State]:
        out, state = self.model(self.normalizer(parts), state)
        return out[VALUE][..., 0], state

    def evaluate(self, minibatch: Steps) -> Tensor:
        """Flat values of recorded steps `[batch]` or sequences `[time, batch]`."""
        parts, state = group(minibatch, "critic"), group(minibatch, "critic_state")
        if minibatch.batch_dims == 1:
            values: Tensor
            values, _ = self(parts, state)
            return values
        raw = unroll(self.model, self.normalizer, parts, state, minibatch["dones"])
        return raw[VALUE][..., 0]
