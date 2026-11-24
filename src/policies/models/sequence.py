"""Running stateful components through `[time, batch]` sequences that restart at resets.

`reset` (time, *batch) marks the steps whose state restarts from zero, as at an episode's first
step. Components may override `forward_sequence` to run the whole sequence natively. The
default steps through time with `forward`.
"""

import torch
from torch import Tensor

from policies.models.state import State, Stateful


def restart(state: State, reset: Tensor) -> State:
    """`state` (*batch, ...) zeroed where `reset` (*batch) is set."""
    return {
        name: value.masked_fill(reset.reshape(*reset.shape, *[1] * (value.dim() - reset.dim())), 0)
        for name, value in state.items()
    }


def step_through(component: Stateful, x: Tensor, state: State, reset: Tensor) -> Tensor:
    """`x` (time, *batch, size) to outputs (time, *batch, out), one `forward` per step.

    Stateless components run on the whole sequence at once.
    """
    if not component.state_shapes:
        return component(x, {})[0]
    outputs = []
    for t in range(x.shape[0]):
        y, state = component(x[t], restart(state, reset[t]))
        outputs.append(y)
    return torch.stack(outputs)
