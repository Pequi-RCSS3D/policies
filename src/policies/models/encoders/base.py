"""The encoder contract. The registry and configs live in `policies.spec.model.encoders`."""

from torch import Tensor, nn

from policies.models.sequence import step_through
from policies.models.state import State


class Encoder(nn.Module):
    """Built from `(config, shape)`, the shape of one input part per sample.

    `forward(x, state)` takes the part, of shape `(*batch, *shape)`, and returns
    `(*batch, out_size)` and the next state. State follows the block contract: bare names in
    `state_shapes`, namespaced by the model, empty for stateless. So does `forward_sequence`.
    """

    out_size: int
    state_shapes: dict[str, tuple[int, ...]]

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        raise NotImplementedError

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        """`x` (time, *batch, *shape) to (time, *batch, out_size), as `Block.forward_sequence`."""
        return step_through(self, x, state, reset)
