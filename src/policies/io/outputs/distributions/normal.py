"""A diagonal normal with a learned, state-independent standard deviation."""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.outputs.distributions.base import distributions
from policies.io.outputs.distributions.events import event_sum, independent, unflatten

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


class NormalConfig(Config):
    initial_std: float | None = Field(default=None, gt=0)
    """Defaults to the learner's `initial_std`."""


class Normal:
    """Raw output is the mean. The log-std is one learned parameter per action element."""

    dtype = DType.FLOAT32

    def __init__(
        self, config: NormalConfig, shape: tuple[int, ...], initial_std: float = 1.0, **_: Any
    ) -> None:
        self.shape = shape
        self.raw_size = math.prod(shape)
        std = initial_std if config.initial_std is None else config.initial_std
        self.learned: Mapping[str, NDArray[np.float32]] = {
            "log_std": np.full(shape, math.log(std), np.float32)
        }

    def mode(self, raw: Any, xp: Any) -> Any:
        return unflatten(raw, self.shape, xp)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Normal as TorchNormal

        mean = raw.reshape(*raw.shape[:-1], *self.shape)
        return independent(
            TorchNormal(mean, learned["log_std"].exp().expand_as(mean)), len(self.shape)
        )

    def imitation(self, distribution: "TorchDistribution", target: "Tensor") -> "Tensor":
        """Squared error of the mean. Likelihood would also shrink the standard deviation."""
        return event_sum((distribution.mean - target) ** 2, len(self.shape))

    def smoothness(self, distribution: "TorchDistribution", previous: "Tensor") -> "Tensor":
        return event_sum((distribution.mean - previous) ** 2, len(self.shape))


distributions.register("normal", NormalConfig)(Normal)
