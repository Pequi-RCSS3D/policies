"""Recurrent state passed explicitly in and out of models.

Components name their state bare, e.g. `h`. Containers prefix their children's names with the
child's module path, e.g. `block.h` or `block.experts.0.h`, so names never collide.
"""

from collections.abc import Mapping
from typing import Protocol

from torch import Tensor

type State = dict[str, Tensor]


class Stateful(Protocol):
    """Anything with the block and encoder state contract."""

    state_shapes: dict[str, tuple[int, ...]]

    def __call__(self, x: Tensor, state: State, /) -> tuple[Tensor, State]: ...

    def forward_sequence(self, x: Tensor, state: State, reset: Tensor) -> Tensor: ...


def namespaced[T](prefix: str, values: Mapping[str, T]) -> dict[str, T]:
    """`values` with each name prefixed by `prefix.`."""
    return {f"{prefix}.{name}": value for name, value in values.items()}


def scoped(component: Stateful, prefix: str, state: State) -> State:
    """`component`'s share of `state`, under `prefix`, with bare names."""
    return {name: state[f"{prefix}.{name}"] for name in component.state_shapes}


def run_scoped(component: Stateful, prefix: str, x: Tensor, state: State) -> tuple[Tensor, State]:
    """Run `component` on its share of `state`, under `prefix`, and namespace its next state."""
    y, inner = component(x, scoped(component, prefix, state))
    return y, namespaced(prefix, inner)


def run_scoped_sequence(
    component: Stateful, prefix: str, x: Tensor, state: State, reset: Tensor
) -> Tensor:
    """`forward_sequence` of `component` from its share of `state`, under `prefix`."""
    return component.forward_sequence(x, scoped(component, prefix, state), reset)
