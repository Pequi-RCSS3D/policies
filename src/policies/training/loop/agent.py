"""What the trainer needs from whatever acts and learns. `Learner` is one."""

from collections.abc import Mapping
from typing import Any, Protocol

import torch
from torch import Tensor

from policies.training.control import Tunable


class Agent(Protocol):
    @property
    def iteration(self) -> int:
        """Updates done so far."""
        ...

    @property
    def device(self) -> torch.device: ...

    @property
    def ready(self) -> bool:
        """True once `update` should run."""
        ...

    def act(self, observation: Mapping[str, Tensor]) -> Mapping[str, Tensor]: ...

    def observe(
        self,
        next_observation: Mapping[str, Tensor],
        rewards: Tensor,
        dones: Tensor,
        truncated: Tensor | None = None,
        final_observation: Mapping[str, Tensor] | None = None,
    ) -> None:
        """`final_observation` ended the step before the reset, to bootstrap truncated episodes."""
        ...

    def update(self) -> Mapping[str, float]:
        """One update. Returns its metrics."""
        ...

    def tunables(self) -> Mapping[str, Tunable]:
        """Values that may change while running, by dotted path."""
        ...

    def set(self, path: str, value: Tunable) -> None:
        """Set a tunable value, already checked against its current type and bounds."""
        ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
