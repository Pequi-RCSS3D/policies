"""Array helpers for trackers."""

from typing import Any

from policies.core.parts import DType
from policies.io.bookkeeping.base import TrackerContext
from policies.io.bookkeeping.errors import UnknownSourceError


def zeros(context: TrackerContext, shape: tuple[int, ...], dtype: str = DType.FLOAT32) -> Any:
    """Zeros for every sample of the batch."""
    xp = context.xp
    return xp.zeros((*context.batch, *shape), dtype=getattr(xp, dtype), device=context.device)


def select(context: TrackerContext, mask: Any, new: Any, old: Any) -> Any:
    """Per-sample choice of `new` where `mask` is set. `mask` has the batch shape."""
    xp = context.xp
    extra = len(old.shape) - len(context.batch)
    return xp.where(xp.reshape(mask, (*mask.shape, *(1,) * extra)), new, old)


def source_shape(context: TrackerContext, source: str) -> tuple[int, ...]:
    """Raises `UnknownSourceError` for a source of unknown shape."""
    if source not in context.shapes:
        raise UnknownSourceError(source, context.shapes)
    return context.shapes[source]


def source_dtype(context: TrackerContext, source: str) -> str:
    return context.dtypes.get(source, DType.FLOAT32)
