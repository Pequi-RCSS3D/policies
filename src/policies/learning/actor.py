"""The acting part of a policy: normalization, model and distribution head."""

from collections.abc import Mapping
from typing import Any

from torch import Tensor, nn

from policies.io.outputs import OutputDecoder
from policies.learning.distributions import DistributionHead, PartsDistribution
from policies.learning.normalizer import Normalizer
from policies.learning.rollout import Steps
from policies.learning.sequences import group, unroll
from policies.models import Model, State, models
from policies.spec import PolicySpec


class Actor(nn.Module):
    """Normalization, model and distribution head: everything export needs.

    `initial_std` is the default for distributions that learn a standard deviation.
    """

    def __init__(
        self, spec: PolicySpec, initial_std: float = 1.0, normalizer_clip: float = 5.0
    ) -> None:
        super().__init__()
        self.spec = spec
        self.settings: dict[str, Any] = {
            "initial_std": initial_std,
            "normalizer_clip": normalizer_clip,
        }
        """Constructor arguments besides `spec`. Saved with the weights to rebuild exactly."""
        self.decoder = OutputDecoder(spec.outputs, initial_std=initial_std)
        normalized = {part.name: part.shape for part in spec.inputs.parts if part.normalize}
        self.normalizer = Normalizer(normalized, clip=normalizer_clip)
        shapes = {part.name: part.shape for part in spec.inputs.parts}
        distributions = self.decoder.distributions
        sizes = {name: distribution.raw_size for name, distribution in distributions.items()}
        self.model: Model = models.build(spec.model, in_shapes=shapes, out_sizes=sizes)
        self.head = DistributionHead(distributions)

    def raw(self, parts: Mapping[str, Tensor], state: State) -> tuple[dict[str, Tensor], State]:
        """Raw outputs: what the exported graph computes."""
        raw: dict[str, Tensor]
        raw, state = self.model(self.normalizer(parts), state)
        return raw, state

    def forward(self, parts: Mapping[str, Tensor], state: State) -> tuple[PartsDistribution, State]:
        raw, state = self.raw(parts, state)
        return self.head(raw), state

    def evaluate(self, minibatch: Steps) -> PartsDistribution:
        """Flat distributions of recorded steps `[batch]` or sequences `[time, batch]`."""
        parts, state = group(minibatch, "actor"), group(minibatch, "actor_state")
        distribution: PartsDistribution
        if minibatch.batch_dims == 1:
            distribution, _ = self(parts, state)
        else:
            distribution = self.head(
                unroll(self.model, self.normalizer, parts, state, minibatch["dones"])
            )
        return distribution
