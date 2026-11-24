"""Metric sums kept on the device, averaged with one sync."""

from collections.abc import Mapping

import torch
from torch import Tensor


class Metrics:
    def __init__(self) -> None:
        self.sums: dict[str, Tensor] = {}
        self.count = 0

    def add(self, values: Mapping[str, Tensor], prefix: str = "") -> None:
        for key, value in values.items():
            name = prefix + key
            self.sums[name] = self.sums[name] + value if name in self.sums else value

    def step(self) -> None:
        """Count one sample of every metric."""
        self.count += 1

    def averages(self) -> dict[str, float]:
        if not self.sums:
            return {}
        averages: list[float] = torch.stack(list(self.sums.values())).div(self.count).tolist()
        return dict(zip(self.sums, averages, strict=True))
