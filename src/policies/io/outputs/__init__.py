"""Outputs composed of named parts."""

from policies.io.outputs.distributions import Distribution, DistributionConfig, distributions
from policies.io.outputs.kinds import Kind, KindConfig, KindWithExtras, kinds
from policies.io.outputs.spec import OutputDecoder, OutputPart, Outputs

__all__ = [
    "Distribution",
    "DistributionConfig",
    "Kind",
    "KindConfig",
    "KindWithExtras",
    "OutputDecoder",
    "OutputPart",
    "Outputs",
    "distributions",
    "kinds",
]
