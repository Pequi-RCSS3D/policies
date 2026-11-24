"""Auxiliary losses that components record during forward, for the learner to add.

Recording is opt-in, like Flax's `sow`: `record` is a no-op unless an `AuxiliaryLosses` over
the module is open, so acting and export pay nothing and the forward signature stays pure.
"""

from contextvars import ContextVar, Token
from types import TracebackType
from typing import Self

import torch
from torch import Tensor, nn

_open: ContextVar[tuple["AuxiliaryLosses", ...]] = ContextVar("auxiliary_losses", default=())


class AuxiliaryLosses:
    """Collects the losses `root`'s modules record while open.

    Names are `<module path>.<name>`, e.g. `block.balance`. A name recorded several times,
    as when unrolling through time, averages. Values are unweighted: the learner owns the weights.
    Reusable: each `with` starts empty.
    """

    def __init__(self, root: nn.Module) -> None:
        self._paths = {id(module): path for path, module in root.named_modules()}
        self._values: dict[str, list[Tensor]] = {}
        self._token: Token[tuple[AuxiliaryLosses, ...]] | None = None

    def __enter__(self) -> Self:
        self._values = {}
        self._token = _open.set((*_open.get(), self))
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _open.reset(self._token)
            self._token = None

    def losses(self) -> dict[str, Tensor]:
        return {name: torch.stack(values).mean() for name, values in self._values.items()}

    def covers(self, module: nn.Module) -> bool:
        return id(module) in self._paths

    def add(self, module: nn.Module, name: str, value: Tensor) -> None:
        """Keep `value` if `module` is under the root. `record` calls this."""
        path = self._paths.get(id(module))
        if path is not None:
            self._values.setdefault(f"{path}.{name}" if path else name, []).append(value)


def recording(module: nn.Module) -> bool:
    """Whether an open collection would keep `module`'s losses, to skip computing them if not."""
    return any(collection.covers(module) for collection in _open.get())


def record(module: nn.Module, name: str, value: Tensor) -> None:
    """Record scalar loss `name` of `module` in every open collection that covers it."""
    for collection in _open.get():
        collection.add(module, name, value)
