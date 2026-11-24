"""The learning-rate schedule contract and registry."""

from typing import Annotated, Any, Protocol

from policies.core.registry import Registry


class Schedule(Protocol):
    def step(self, lr: float, iteration: int, metrics: dict[str, float]) -> float: ...


schedules: Registry[Schedule] = Registry("schedule")

type ScheduleConfig = Annotated[dict[str, Any], schedules]
"""Any schedule's config, validated to its canonical form."""

DEFAULT_SCHEDULE = "constant"
"""The schedule of a config without `type`."""
