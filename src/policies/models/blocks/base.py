"""The block contract. The registry and configs live in `policies.spec.model.blocks`."""

from torch import Tensor, nn

from policies.models.sequence import step_through
from policies.models.state import State


class Block(nn.Module):
    """Built from `(config, in_sizes)`, one size per encoded input part.

    `forward(x, state)` takes the parts' concatenation, of shape `(*batch, sum(in_sizes))`, and
    returns `(*batch, out_size)` and the next state. `state_shapes` maps bare names, e.g. `h`, to
    per-sample shapes, and the container namespaces them. Empty means stateless. Auxiliary losses
    go through `policies.models.auxiliary.record`. `forward_sequence` is optional, see there.
    """

    out_size: int
    state_shapes: dict[str, tuple[int, ...]]

    def forward(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        raise NotImplementedError

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor:
        """`x` (time, *batch, sum(in_sizes)) from `state` to outputs (time, *batch, out_size).

        `reset` (time, *batch) zeroes the state before a step. Must match stepping `forward`.
        Override to run the sequence natively. The default steps through time.
        """
        return step_through(self, x, state, reset)
