"""Adapts the learning rate to keep each update's KL near a target."""

from collections.abc import Mapping
from typing import Any

from policies.core.config import Config
from policies.learning.schedules.base import schedules

FOLLOWED = "ppo"
"""The objective type whose KL is followed when no metric is given."""


class AdaptiveKLConfig(Config):
    target_kl: float = 0.01
    factor: float = 1.5
    min_lr: float = 1e-5
    max_lr: float = 1e-2
    metric: str | None = None
    """The update metric to follow. None follows `<name>/kl` of the one `ppo` objective."""


class AdaptiveKL:
    """Lowers the learning rate when the policy moves too far per update, raises it otherwise."""

    def __init__(
        self, config: AdaptiveKLConfig, objectives: Mapping[str, str] | None = None, **_: Any
    ) -> None:
        """`objectives` maps instance names to types."""
        self.config = config
        self.metric = config.metric or self._followed(objectives or {})

    @staticmethod
    def _followed(objectives: Mapping[str, str]) -> str:
        names = [name for name, kind in objectives.items() if kind == FOLLOWED]
        if len(names) != 1:
            raise ValueError(
                f"adaptive_kl follows the one {FOLLOWED!r} objective, found {names}. Set `metric`."
            )
        return f"{names[0]}/kl"

    def step(self, lr: float, iteration: int, metrics: dict[str, float]) -> float:
        kl = metrics.get(self.metric)
        if kl is None:
            return lr
        if kl > 2 * self.config.target_kl:
            lr /= self.config.factor
        elif kl < self.config.target_kl / 2:
            lr *= self.config.factor
        return min(max(lr, self.config.min_lr), self.config.max_lr)


schedules.register("adaptive_kl", AdaptiveKLConfig)(AdaptiveKL)
