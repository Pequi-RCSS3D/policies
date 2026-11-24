"""Training distributions of output parts. Register more with `distributions.register`."""

from policies.io.outputs.distributions.base import Distribution, DistributionConfig, distributions
from policies.io.outputs.distributions.bernoulli import Bernoulli
from policies.io.outputs.distributions.categorical import Categorical
from policies.io.outputs.distributions.constant import Constant
from policies.io.outputs.distributions.events import event_sum, independent, unflatten
from policies.io.outputs.distributions.multi_categorical import MultiCategorical
from policies.io.outputs.distributions.normal import Normal
from policies.io.outputs.distributions.normal_state_std import NormalStateStd

__all__ = [
    "Bernoulli",
    "Categorical",
    "Constant",
    "Distribution",
    "DistributionConfig",
    "MultiCategorical",
    "Normal",
    "NormalStateStd",
    "distributions",
    "event_sum",
    "independent",
    "unflatten",
]
