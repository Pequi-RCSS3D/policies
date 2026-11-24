"""Running models through `[time, batch]` sequences, and grouping recorded keys."""

from collections.abc import Mapping

import torch
from torch import Tensor

from policies.learning.normalizer import Normalizer
from policies.models import Model, State


def mask(x: Tensor, keep: Tensor) -> Tensor:
    """`x` times `keep`, broadcast over its trailing dimensions."""
    return x * keep.reshape(-1, *[1] * (x.dim() - 1))


def group(batch: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Entries under `prefix/`, without the prefix."""
    start = prefix + "/"
    return {k[len(start) :]: v for k, v in batch.items() if k.startswith(start)}


def unroll(
    model: Model,
    normalizer: Normalizer,
    parts: Mapping[str, Tensor],
    state: State,
    dones: Tensor,
) -> dict[str, Tensor]:
    """Run a model through `[time, batch]` sequences from their first state. Outputs are flat.

    The state resets after every step that ended an episode. Runs `model.forward_sequence`,
    natively where the model's components support it.
    """
    initial = {name: value[0] for name, value in state.items()}
    reset = torch.cat([torch.zeros_like(dones[:1]), dones[:-1]]).bool()
    raw = model.forward_sequence(normalizer(parts), initial, reset)
    return {name: value.flatten(0, 1) for name, value in raw.items()}
