"""Several independent choices, each from its own number of options."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from array_api_compat import array_namespace
from numpy.typing import NDArray
from pydantic import Field

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.outputs.distributions.base import distributions
from policies.io.outputs.distributions.constant import Constant
from policies.io.outputs.distributions.events import independent, unflatten

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution

MASKED = -1e9
"""Logit of options past an element's count. Finite, so KL and entropy stay finite."""


class MultiCategoricalConfig(Config):
    choices: tuple[Annotated[int, Field(ge=1)], ...] = Field(min_length=1)
    """Number of options per element."""


class MultiCategorical:
    """Raw output is logits, padded to the largest count per element. Actions are indices."""

    dtype = DType.INT64
    learned: Mapping[str, NDArray[np.float32]] = {}

    def __init__(
        self, config: MultiCategoricalConfig, shape: tuple[int, ...] | None = None, **_: Any
    ) -> None:
        self.shape = (len(config.choices),)
        if shape is not None and shape != self.shape:
            raise ValueError(f"{len(config.choices)} choice counts for actions of shape {shape}")
        self.options = max(config.choices)
        self.raw_size = len(config.choices) * self.options
        counts = np.asarray(config.choices)[:, None]
        padding = np.where(np.arange(self.options) < counts, 0.0, MASKED).astype(np.float32)
        self.mask = Constant(padding) if (padding != 0).any() else None

    def _logits(self, raw: Any, xp: Any) -> Any:
        logits = unflatten(raw, (*self.shape, self.options), xp)
        return logits if self.mask is None else logits + self.mask.like(logits)

    def mode(self, raw: Any, xp: Any) -> Any:
        return xp.astype(xp.argmax(self._logits(raw, xp), axis=-1), xp.int64)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Categorical as TorchCategorical

        return independent(TorchCategorical(logits=self._logits(raw, array_namespace(raw))), 1)

    def imitation(self, distribution: "TorchDistribution", target: "Tensor") -> "Tensor":
        return -distribution.log_prob(target.long())

    def smoothness(self, distribution: "TorchDistribution", previous: "Tensor") -> None:
        return None


distributions.register("multi_categorical", MultiCategoricalConfig)(MultiCategorical)
