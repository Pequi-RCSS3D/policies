"""Generalized state-dependent exploration (gSDE, Raffin, Kober and Stulp, CoRL 2021).

As Stable-Baselines3's `StateDependentNoiseDistribution`, with the features as model outputs.
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import Field

from policies.core.config import Config
from policies.io.outputs.distributions.base import Noise, distributions
from policies.io.outputs.distributions.events import independent, unflatten
from policies.io.outputs.distributions.normal import Normal, NormalConfig

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution

VARIANCE_EPSILON = 1e-6
"""Added to the variance, as Stable-Baselines3 does."""


class GSDEConfig(Config):
    latent: int = Field(default=64, ge=1)
    """Size of the features `z` the model outputs beside the mean."""
    sample_every: int = Field(default=4, ge=1)
    """Steps each environment holds its noise matrix. Episode starts redraw it too."""
    full_std: bool = True
    """A standard deviation per feature and action element, else one per feature."""
    learn_features: bool = False
    """Let log-prob gradients reach the features. Detached by default, as in Stable-Baselines3."""
    initial_std: float | None = Field(default=None, gt=0)
    """Of each noise matrix entry. Defaults to the learner's `initial_std`."""


class GSDE(Normal):
    """Raw output is the mean, then features `z`. Training adds `z @ noise`, a matrix of
    `N(0, std**2)` entries per environment, so the action std per element is
    `sqrt(sum_i z_i**2 * std_ij**2)`. The mode is the mean.
    """

    def __init__(
        self, config: GSDEConfig, shape: tuple[int, ...], initial_std: float = 1.0, **_: Any
    ) -> None:
        super().__init__(NormalConfig(initial_std=config.initial_std), shape, initial_std)
        std = initial_std if config.initial_std is None else config.initial_std
        self.config = config
        self.size = self.raw_size
        self.raw_size = self.size + config.latent
        std_shape = (config.latent, *shape) if config.full_std else (config.latent,)
        self.learned = {"log_std": np.full(std_shape, math.log(std), np.float32)}

    def mode(self, raw: Any, xp: Any) -> Any:
        return unflatten(raw[..., : self.size], self.shape, xp)

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        from torch.distributions import Normal as TorchNormal

        features = raw[..., self.size :]
        if not self.config.learn_features:
            features = features.detach()
        variance = features.square() @ (2 * learned["log_std"]).exp().reshape(
            self.config.latent, -1
        )
        batch = raw.shape[:-1]
        std = (variance + VARIANCE_EPSILON).sqrt().expand(*batch, self.size)
        shape = (*batch, *self.shape)
        return independent(
            TorchNormal(raw[..., : self.size].reshape(shape), std.reshape(shape)), len(self.shape)
        )

    def noise(self, batch: int, device: Any) -> Noise:
        from policies.io.outputs.distributions.gsde_noise import GSDENoise

        return GSDENoise(self.shape, self.config.latent, self.config.sample_every, batch, device)


distributions.register("gsde", GSDEConfig)(GSDE)
