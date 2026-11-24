"""Values a running job lets you change, with type and bound checks."""

from collections.abc import Collection, Mapping

from policies.training.control.messages import ControlError, Tunable

type Bounds = tuple[float | None, float | None]


def _kind(value: Tunable) -> str:
    return type(value).__name__


def _coerce(path: str, current: Tunable, new: Tunable) -> Tunable:
    if isinstance(current, bool):
        if isinstance(new, bool):
            return new
    elif isinstance(current, int):
        if isinstance(new, int) and not isinstance(new, bool):
            return new
    elif isinstance(current, float):
        if isinstance(new, int | float) and not isinstance(new, bool):
            return float(new)
    elif isinstance(current, str) and isinstance(new, str):
        return new
    raise ControlError(f"{path} is {_kind(current)}, got {_kind(new)} {new!r}")


def check_path(path: str, tunable: Collection[str]) -> None:
    if path not in tunable:
        known = ", ".join(sorted(tunable)) or "none"
        raise ControlError(f"{path} is not tunable. Tunable: {known}")


def check_bounds(path: str, value: Tunable, bounds: Bounds) -> None:
    """Numbers only. Other values have no bounds."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        return
    low, high = bounds
    if low is not None and value < low:
        raise ControlError(f"{path}={value!r} is below the bound {low!r}")
    if high is not None and value > high:
        raise ControlError(f"{path}={value!r} is above the bound {high!r}")


class Tunables:
    """Current values keyed by dotted path, with optional (low, high) bounds."""

    def __init__(
        self, values: Mapping[str, Tunable], bounds: Mapping[str, Bounds] | None = None
    ) -> None:
        self._values = dict(values)
        self._bounds = dict(bounds or {})
        unknown = sorted(set(self._bounds) - set(self._values))
        if unknown:
            raise ValueError(f"bounds for unknown paths: {', '.join(unknown)}")

    @property
    def values(self) -> dict[str, Tunable]:
        return dict(self._values)

    def apply(self, changes: Mapping[str, Tunable]) -> dict[str, tuple[Tunable, Tunable]]:
        """Validate every change, then apply all of them. Returns `{path: (old, new)}`."""
        accepted: dict[str, tuple[Tunable, Tunable]] = {}
        for path, new in changes.items():
            check_path(path, self._values)
            old = self._values[path]
            value = _coerce(path, old, new)
            check_bounds(path, value, self._bounds.get(path, (None, None)))
            accepted[path] = (old, value)
        for path, (_, value) in accepted.items():
            self._values[path] = value
        return accepted
