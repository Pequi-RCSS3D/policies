"""Independent distributions per output part."""

from collections.abc import Mapping

import torch
from torch import Tensor

from policies.io.outputs import Noise
from policies.learning.distributions.part import PartDistribution


class PartsDistribution:
    """Log-probs, entropies and KLs sum over parts, per sample."""

    def __init__(self, parts: Mapping[str, PartDistribution]) -> None:
        self.parts = dict(parts)

    def sample(self, noise: Mapping[str, Noise] | None = None) -> dict[str, Tensor]:
        """Parts in `noise` sample from their noise state, others independently."""
        noise = noise or {}
        return {name: part.sample(noise.get(name)) for name, part in self.parts.items()}

    def mode(self) -> dict[str, Tensor]:
        return {name: part.mode() for name, part in self.parts.items()}

    def log_prob(self, actions: Mapping[str, Tensor]) -> Tensor:
        return torch.stack([part.log_prob(actions[name]) for name, part in self.parts.items()]).sum(
            0
        )

    def entropy(self) -> Tensor:
        return torch.stack([part.entropy() for part in self.parts.values()]).sum(0)

    def kl(self, other: "PartsDistribution") -> Tensor:
        return torch.stack([part.kl(other.parts[name]) for name, part in self.parts.items()]).sum(0)
