"""The environment contract."""

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor

type Observation = dict[str, Tensor]


type Step = tuple[Observation, Tensor, Tensor, dict[str, Any]]


TRUNCATED = "truncated"
"""Optional `info` key: a bool mask of environments cut off, e.g. by a time limit, to bootstrap."""

FINAL_OBSERVATION = "final_observation"
"""Optional `info` key: the observation that ended the step, before the reset.

Rows of environments that are not done equal the returned observation.
"""


class Environment(Protocol):
    """Batched and auto-resetting.

    On `done`, the returned observation starts the next episode. `info` may carry
    `TRUNCATED` and `FINAL_OBSERVATION`. The trainer passes outputs on the device of the
    observations.
    """

    num_envs: int

    def reset(self) -> Observation: ...
    def step(self, outputs: Mapping[str, Tensor]) -> Step: ...
