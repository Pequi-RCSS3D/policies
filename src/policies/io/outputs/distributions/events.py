"""Helpers between flat raw outputs and event-shaped actions."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor
    from torch.distributions import Distribution as TorchDistribution


def unflatten(raw: Any, shape: tuple[int, ...], xp: Any) -> Any:
    """`(..., size)` to `(..., *shape)`."""
    return xp.reshape(raw, (*raw.shape[:-1], *shape))


def event_sum(x: "Tensor", ndim: int) -> "Tensor":
    """Sums the last `ndim` dimensions, none for scalar events."""
    return x.reshape(*x.shape[: x.dim() - ndim], -1).sum(-1)


def independent(base: "TorchDistribution", ndim: int) -> "TorchDistribution":
    """Treats the last `ndim` batch dimensions as one event, so log-probs sum over them."""
    from torch.distributions import Independent

    return Independent(base, ndim) if ndim else base
