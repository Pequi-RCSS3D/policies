"""The teacher contract."""

from collections.abc import Callable, Mapping

from torch import Tensor

type Teacher = Callable[[Mapping[str, Tensor], Tensor], Mapping[str, Tensor]]
"""`(observation, reset) -> outputs`, batched over environments.

`reset` flags environments whose episode starts with this observation. Outputs are in action
space, e.g. the mode of each part's distribution. A teacher with an `artifact_id` attribute runs
that artifact, which the student's export records in `parents`.
"""
