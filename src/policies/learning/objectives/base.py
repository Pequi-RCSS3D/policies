"""The objective contract, its contexts, the registry and shared losses."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol

import torch
from torch import Tensor, nn

from policies.core.registry import Registry
from policies.learning.distributions import PartsDistribution


@dataclass
class ActContext:
    """What objectives see when the learner acts, to record extra data per step."""

    observation: Mapping[str, Tensor]
    actions: Mapping[str, Tensor]
    reset: Tensor


@dataclass
class UpdateContext:
    batch: Mapping[str, Tensor]
    """Flat steps: `actions/<part>`, `log_prob`, `rewards`, `dones` and, with a critic,
    `values`, `returns` and `advantages`."""
    distribution: PartsDistribution
    log_prob: Tensor
    values: Tensor | None = None
    """The critic's current values. None without a critic."""
    reference: PartsDistribution | None = None
    recorded: dict[str, Tensor] = field(default_factory=dict)
    """What this objective recorded while acting."""


class Objective(Protocol):
    hyperparameters: dict[str, Any]
    """Tunable values by name, including `weight`."""
    needs_reference: bool
    """Gets `UpdateContext.reference`, a frozen copy of the actor taken at the first update."""
    needs_values: bool
    """Needs the critic: values, returns and advantages. Without one, there is no critic."""

    def parameters(self) -> Iterable[nn.Parameter]:
        """Trained, clipped and checkpointed along with the policy."""
        ...

    def record(self, context: ActContext) -> dict[str, Tensor]: ...
    def loss(self, context: UpdateContext) -> tuple[Tensor, dict[str, Tensor]]: ...


objectives: Registry[Objective] = Registry("objective")
"""Factories are called as `factory(config, teachers=..., device=...)`."""

type ObjectiveConfig = Annotated[dict[str, Any], objectives]
"""Any objective's config, validated to its canonical form."""


class BaseObjective:
    """Defaults: no reference, no critic, no parameters, nothing recorded."""

    hyperparameters: dict[str, Any]
    needs_reference = False
    needs_values = False

    def parameters(self) -> Iterable[nn.Parameter]:
        return ()

    def record(self, context: ActContext) -> dict[str, Tensor]:
        return {}


def imitation_loss(distribution: PartsDistribution, targets: Mapping[str, Tensor]) -> Tensor:
    """Each part's own imitation loss against its target actions, averaged over samples."""
    total = torch.zeros((), device=next(iter(targets.values())).device)
    for name, part in distribution.parts.items():
        if name in targets:
            total = total + part.imitation(targets[name]).mean()
    return total
