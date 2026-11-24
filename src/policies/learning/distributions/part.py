"""One output part's training distribution, built from its raw output."""

from collections.abc import Mapping

import array_api_compat.torch as torch_xp
from torch import Tensor
from torch.distributions import kl_divergence

from policies.io.outputs import Distribution


class PartDistribution:
    """A part's distribution, its raw output and the torch distribution they make."""

    def __init__(
        self, definition: Distribution, raw: Tensor, learned: Mapping[str, Tensor]
    ) -> None:
        self.definition, self.raw = definition, raw
        self.torch = definition.torch_distribution(raw, learned)

    def sample(self) -> Tensor:
        return self.torch.sample()

    def mode(self) -> Tensor:
        """The same function inference runs on NumPy."""
        mode: Tensor = self.definition.mode(self.raw, torch_xp)
        return mode

    def log_prob(self, action: Tensor) -> Tensor:
        return self.torch.log_prob(action)

    def entropy(self) -> Tensor:
        return self.torch.entropy()

    def kl(self, other: "PartDistribution") -> Tensor:
        return kl_divergence(self.torch, other.torch)

    def imitation(self, target: Tensor) -> Tensor:
        return self.definition.imitation(self.torch, target)

    def smoothness(self, previous: Tensor) -> Tensor | None:
        return self.definition.smoothness(self.torch, previous)
