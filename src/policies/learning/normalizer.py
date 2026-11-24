"""Running mean and variance per part, on the training device."""

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn


class Normalizer(nn.Module):
    """Standardizes each part by its running statistics, then clips to `[-clip, clip]`."""

    def __init__(
        self, shapes: Mapping[str, tuple[int, ...]], clip: float = 5.0, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.shapes, self.clip, self.eps = dict(shapes), clip, eps
        for name, shape in self.shapes.items():
            self.register_buffer(f"{name}_mean", torch.zeros(shape))
            self.register_buffer(f"{name}_var", torch.ones(shape))
            self.register_buffer(f"{name}_count", torch.zeros(()))

    def _get(self, name: str, what: str) -> Tensor:
        return cast(Tensor, getattr(self, f"{name}_{what}"))

    @torch.no_grad()
    def update(self, parts: Mapping[str, Tensor]) -> None:
        for name, shape in self.shapes.items():
            x = parts[name].float().reshape(-1, *shape)
            n = x.shape[0]
            mean, var = self._get(name, "mean"), self._get(name, "var")
            count = self._get(name, "count")
            batch_mean, batch_var = x.mean(0), x.var(0, unbiased=False)
            total = count + n
            delta = batch_mean - mean
            mean.add_(delta * n / total)
            var.copy_((var * count + batch_var * n + delta**2 * count * n / total) / total)
            count.fill_(total)

    def forward(self, parts: Mapping[str, Tensor]) -> dict[str, Tensor]:
        out = dict(parts)
        for name in self.shapes:
            x = parts[name].float()
            z = (x - self._get(name, "mean")) / torch.sqrt(self._get(name, "var") + self.eps)
            out[name] = z.clamp(-self.clip, self.clip)
        return out
