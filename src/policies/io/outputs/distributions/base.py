"""The distribution contract and registry."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from policies.core.parts import DType
from policies.core.registry import Registry

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


class Distribution(Protocol):
    """How an output part's actions are distributed, given its raw model output.

    The model produces `raw_size` values per sample. Actions have `shape` and `dtype` per
    sample, which is also what bookkeeping keeps. `learned` holds initial values of
    parameters trained alongside the model. `mode` runs on any array namespace `xp`, the
    other methods are for training and take torch tensors.
    """

    @property
    def raw_size(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def learned(self) -> Mapping[str, NDArray[np.float32]]: ...

    def mode(self, raw: Any, xp: Any) -> Any:
        """The most likely action."""
        ...

    def torch_distribution(
        self, raw: "Tensor", learned: Mapping[str, "Tensor"]
    ) -> "TorchDistribution":
        """Samples, log-probs, entropy and KL. Its event shape is `shape`."""
        ...

    def imitation(self, distribution: "TorchDistribution", target: "Tensor") -> "Tensor":
        """Per-sample loss for matching target actions."""
        ...

    def smoothness(self, distribution: "TorchDistribution", previous: "Tensor") -> "Tensor | None":
        """Per-sample penalty for changing from the previous action, or None if not applicable."""
        ...


class Noise(Protocol):
    """Exploration noise that persists across steps, one state per environment."""

    def sample(self, raw: "Tensor", learned: Mapping[str, "Tensor"]) -> "Tensor":
        """This step's actions `(batch, *shape)` from raw outputs `(batch, raw_size)`."""
        ...

    def reset(self, done: "Tensor") -> None:
        """Restarts the noise of environments where `done`, of shape `(batch,)`, holds."""
        ...


@runtime_checkable
class DistributionWithNoise(Protocol):
    """A distribution whose training samples come from stateful `Noise`, not independent draws.

    Log-probs, entropy and KL still come from `torch_distribution`, the per-step marginal.
    """

    def noise(self, batch: int, device: Any) -> Noise:
        """Fresh noise state for `batch` environments."""
        ...


distributions: Registry[Distribution] = Registry("distribution")

type DistributionConfig = Annotated[dict[str, Any], distributions]
"""A distribution config, validated and stored canonical."""
