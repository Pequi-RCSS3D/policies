"""Rounding modes."""

from collections.abc import Callable
from enum import StrEnum

import torch
from torch import Tensor


class Rounding(StrEnum):
    NEAREST = "nearest"
    """Halves go to the even neighbor."""
    TRUNCATE = "truncate"
    """Toward zero."""
    FLOOR = "floor"
    CEIL = "ceil"


ROUNDERS: dict[Rounding, Callable[[Tensor], Tensor]] = {
    Rounding.NEAREST: torch.round,
    Rounding.TRUNCATE: torch.trunc,
    Rounding.FLOOR: torch.floor,
    Rounding.CEIL: torch.ceil,
}
