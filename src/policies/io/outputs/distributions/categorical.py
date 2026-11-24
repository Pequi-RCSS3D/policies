"""One of several choices."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from policies.core.config import Config
from policies.core.parts import DType
from policies.io.outputs.distributions.base import distributions

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


class CategoricalConfig(Config):
    choices: int = Field(ge=1)


class Categorical:
    """Raw output is logits. Actions are indices."""

    shape: tuple[int, ...] = ()
    dtype = DType.INT64
    learned: Mapping[str, NDArray[np.float32]] = {}

    def __init__(self, config: CategoricalConfig, **_: Any) -> None:
        self.raw_size = config.choices

    def mode(self, raw: Any, xp: Any) -> Any:
        return xp.astype(xp.argmax(raw, axis=-1), xp.int64)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Categorical as TorchCategorical

        return TorchCategorical(logits=raw)

    def imitation(self, distribution: "TorchDistribution", target: "Tensor") -> "Tensor":
        return -distribution.log_prob(target.long())

    def smoothness(self, distribution: "TorchDistribution", previous: "Tensor") -> None:
        return None


distributions.register("categorical", CategoricalConfig)(Categorical)
