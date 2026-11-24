"""Independent yes-or-no decisions."""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.outputs.distributions.base import distributions
from policies.io.outputs.distributions.events import independent, unflatten

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


class BernoulliConfig(Config):
    pass


class Bernoulli:
    """Raw output is one logit per action element. Actions are 0 or 1."""

    dtype = DType.FLOAT32
    learned: Mapping[str, NDArray[np.float32]] = {}

    def __init__(self, config: BernoulliConfig, shape: tuple[int, ...], **_: Any) -> None:
        self.shape = shape
        self.raw_size = math.prod(shape)

    def mode(self, raw: Any, xp: Any) -> Any:
        return xp.astype(unflatten(raw, self.shape, xp) > 0, raw.dtype)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Bernoulli as TorchBernoulli

        logits = raw.reshape(*raw.shape[:-1], *self.shape)
        return independent(TorchBernoulli(logits=logits), len(self.shape))

    def imitation(self, distribution: "TorchDistribution", target: "Tensor") -> "Tensor":
        return -distribution.log_prob(target.float())

    def smoothness(self, distribution: "TorchDistribution", previous: "Tensor") -> None:
        return None


distributions.register("bernoulli", BernoulliConfig)(Bernoulli)
