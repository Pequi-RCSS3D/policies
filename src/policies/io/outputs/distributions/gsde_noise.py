"""Per-environment gSDE noise matrices, held for several steps."""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


class GSDENoise:
    """Unit-normal matrices `(batch, latent, size)`, scaled by the current std when sampling.

    Each environment redraws its matrix every `sample_every` steps and at each episode start.
    """

    def __init__(
        self, shape: tuple[int, ...], latent: int, sample_every: int, batch: int, device: Any
    ) -> None:
        import torch

        self.shape, self.sample_every = shape, sample_every
        self.size = math.prod(shape)
        self.unit = torch.zeros(batch, latent, self.size, device=device)
        self.age = torch.zeros(batch, dtype=torch.long, device=device)
        """Steps since each matrix was drawn, modulo `sample_every`. Zero redraws."""

    def sample(self, raw: "Tensor", learned: Mapping[str, "Tensor"]) -> "Tensor":
        import torch

        # Drawing every step and selecting avoids a host sync on which environments are due.
        due = (self.age == 0)[:, None, None]
        self.unit = torch.where(due, torch.randn_like(self.unit), self.unit)
        self.age = (self.age + 1) % self.sample_every
        std = learned["log_std"].exp().reshape(self.unit.shape[1], -1)
        features = raw[..., self.size :].unsqueeze(-2)
        action = raw[..., : self.size] + (features @ (std * self.unit)).squeeze(-2)
        return action.reshape(*raw.shape[:-1], *self.shape)

    def reset(self, done: "Tensor") -> None:
        self.age = self.age.masked_fill(done, 0)
