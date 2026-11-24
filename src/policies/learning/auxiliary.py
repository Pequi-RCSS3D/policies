"""Weights the auxiliary losses that models record while learning."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from policies.learning.metrics import Metrics
from policies.models import AuxiliaryLosses


class Auxiliary:
    """Collects what `models` record, named `<key>.<module path>.<loss>`, e.g.
    `actor.block.balance`. Each loss is weighted by its last segment, unweighted ones are only
    reported. Copies of the models, like a reference actor, are not covered.
    """

    def __init__(self, models: Mapping[str, nn.Module], weights: Mapping[str, float]) -> None:
        self.collection = AuxiliaryLosses(nn.ModuleDict(dict(models)))
        """Open it with `with` around the forward passes to collect."""
        self.weights = dict(weights)

    def loss(self, metrics: Metrics) -> Tensor | None:
        """The weighted sum of what was collected, reported as `auxiliary/<name>`."""
        losses = self.collection.losses()
        metrics.add({name: value.detach() for name, value in losses.items()}, "auxiliary/")
        weighted = [
            self.weights[name.rsplit(".", 1)[-1]] * value
            for name, value in losses.items()
            if name.rsplit(".", 1)[-1] in self.weights
        ]
        return torch.stack(weighted).sum() if weighted else None
