"""One optimizer over every trained parameter, its learning rate schedule and clipping."""

from collections.abc import Mapping, Sequence
from typing import Any

from torch import Tensor, nn

from policies.learning.optimizers import optimizers
from policies.learning.schedules import schedules


class Optimization:
    def __init__(
        self,
        parameters: Sequence[nn.Parameter],
        optimizer: Mapping[str, Any],
        schedule: Mapping[str, Any],
        max_grad_norm: float,
        **schedule_context: Any,
    ) -> None:
        self.parameters = list(parameters)
        self.optimizer = optimizers.build(optimizer, params=self.parameters)
        self.lr = float(self.optimizer.param_groups[0]["lr"])
        self.schedule = schedules.build(schedule, base_lr=self.lr, **schedule_context)
        self.max_grad_norm = max_grad_norm

    def step(self, loss: Tensor) -> Tensor:
        """Backpropagate, clip and step. Returns the gradient norm before clipping."""
        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]  # torch Tensor.backward is unannotated
        norm = nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        return norm

    def set_lr(self, lr: float) -> None:
        self.lr = lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def schedule_step(self, iteration: int, metrics: dict[str, float]) -> None:
        self.set_lr(self.schedule.step(self.lr, iteration, metrics))

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict(), "lr": self.lr}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.set_lr(state["lr"])
