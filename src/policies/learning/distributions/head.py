"""Learned distribution parameters, and distributions built from raw outputs."""

from collections.abc import Mapping
from typing import cast

import torch
from torch import Tensor, nn

from policies.io.outputs import Distribution
from policies.learning.distributions.part import PartDistribution
from policies.learning.distributions.parts import PartsDistribution


class DistributionHead(nn.Module):
    """Holds each distribution's `learned` parameters, e.g. a state-independent log-std."""

    def __init__(self, distributions: Mapping[str, Distribution]) -> None:
        super().__init__()
        self.definitions = dict(distributions)
        self.learned = nn.ModuleDict(
            {
                name: nn.ParameterDict(
                    {
                        key: nn.Parameter(torch.tensor(value))
                        for key, value in definition.learned.items()
                    }
                )
                for name, definition in distributions.items()
            }
        )

    def forward(self, raw: Mapping[str, Tensor]) -> PartsDistribution:
        return PartsDistribution(
            {
                name: PartDistribution(
                    definition, raw[name], dict(cast(nn.ParameterDict, self.learned[name]))
                )
                for name, definition in self.definitions.items()
            }
        )
