"""Training distributions over output parts, from each part's registered distribution."""

from policies.learning.distributions.head import DistributionHead
from policies.learning.distributions.part import PartDistribution
from policies.learning.distributions.parts import PartsDistribution

__all__ = ["DistributionHead", "PartDistribution", "PartsDistribution"]
