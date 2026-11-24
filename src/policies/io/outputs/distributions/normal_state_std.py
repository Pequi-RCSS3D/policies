"""A diagonal normal whose standard deviation the model outputs too."""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from policies.core.config import Config
from policies.io.outputs.distributions.base import distributions
from policies.io.outputs.distributions.events import independent, unflatten
from policies.io.outputs.distributions.normal import Normal, NormalConfig

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


class NormalStateStdConfig(Config):
    initial_std: float | None = Field(default=None, gt=0)
    """Standard deviation at a zero raw output. Defaults to the learner's `initial_std`."""
    min_std: float = Field(default=1e-4, ge=0)


class NormalStateStd(Normal):
    """Raw output is the mean, then the standard deviation before a biased softplus."""

    def __init__(
        self,
        config: NormalStateStdConfig,
        shape: tuple[int, ...],
        initial_std: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__(NormalConfig(initial_std=config.initial_std), shape, initial_std)
        std = initial_std if config.initial_std is None else config.initial_std
        if std <= config.min_std:
            raise ValueError(f"initial_std {std} must exceed min_std {config.min_std}")
        self.min_std = config.min_std
        # Inverse softplus, so a zero raw output gives the initial standard deviation.
        self.bias = math.log(math.expm1(std - config.min_std))
        self.size = self.raw_size
        self.raw_size = 2 * self.size
        self.learned = {}

    def mode(self, raw: Any, xp: Any) -> Any:
        return unflatten(raw[..., : self.size], self.shape, xp)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Normal as TorchNormal
        from torch.nn.functional import softplus

        mean = raw[..., : self.size].reshape(*raw.shape[:-1], *self.shape)
        std = softplus(raw[..., self.size :] + self.bias) + self.min_std
        return independent(TorchNormal(mean, std.reshape(mean.shape)), len(self.shape))


distributions.register("normal_state_std", NormalStateStdConfig)(NormalStateStd)
