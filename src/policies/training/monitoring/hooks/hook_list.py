"""Fans events out to hooks."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from policies.core.logs import get_logger
from policies.training.monitoring.events import Event
from policies.training.monitoring.hooks.base import FunctionHook, Hook, hooks

log = get_logger("policies.training")


class HookList:
    """Fans events out to hooks. A failing hook is logged once and disabled."""

    def __init__(self, hooks: Mapping[str, Hook] | Sequence[Hook] = ()) -> None:
        if isinstance(hooks, Mapping):
            self._hooks = dict(hooks)
        else:
            self._hooks = {}
            for hook in hooks:
                name = type(hook).__name__
                if name in self._hooks:
                    name = f"{name}#{len(self._hooks)}"
                self._hooks[name] = hook

    @property
    def names(self) -> list[str]:
        return list(self._hooks)

    def add(self, name: str, function: Callable[[Event], None], every: int = 1) -> None:
        """Attach any function as a hook, e.g. to forward metrics to your own dashboard."""
        if name in self._hooks:
            raise ValueError(f"hook {name!r} already exists")
        self._hooks[name] = FunctionHook(function, every)

    def emit(self, event: Event) -> None:
        # A hook may emit in turn, so iterate over a copy and skip hooks disabled meanwhile.
        for name, hook in list(self._hooks.items()):
            if name not in self._hooks:
                continue
            if event.traits.throttled and event.step % max(hook.every, 1) != 0:
                continue
            try:
                hook(event)
            except Exception as error:
                self._hooks.pop(name, None)
                log.warning("hook disabled", hook=name, during=event.kind, error=repr(error))

    def close(self) -> None:
        for name, hook in self._hooks.items():
            close = getattr(hook, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as error:
                    log.warning("hook failed to close", hook=name, error=repr(error))


def build_hooks(
    configs: Iterable[Mapping[str, Any]], run_directory: Path, **context: Any
) -> HookList:
    """Build each `{type: ...}` config with `run_directory` and `context`."""
    built: dict[str, Hook] = {}
    for config in configs:
        name = str(config.get("type"))
        if name in built:
            name = f"{name}#{len(built)}"
        built[name] = hooks.build(config, run_directory=Path(run_directory), **context)
    return HookList(built)
