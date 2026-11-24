"""Every registry, discovered through one Python entry-point group.

A distribution that defines a registry names it, so tools can list it without importing
the others:

    [project.entry-points."policies.registries"]
    formations = "my_package.formations:formations"
"""

from functools import cached_property
from importlib import metadata
from typing import Any

from policies.core.registry.base import Registry
from policies.core.registry.errors import PluginError, unknown


class Catalog:
    """Registries by name. Each is imported on first use, with its built-ins."""

    def __init__(self, group: str) -> None:
        self.group = group

    @cached_property
    def _entry_points(self) -> dict[str, metadata.EntryPoint]:
        return {point.name: point for point in metadata.entry_points(group=self.group)}

    def names(self) -> list[str]:
        return sorted(self._entry_points)

    def __getitem__(self, name: str) -> Registry[Any]:
        point = self._entry_points.get(name)
        if point is None:
            raise unknown("registry", name, self.names())
        registry = point.load()
        if not isinstance(registry, Registry):
            raise PluginError(f"{name} = {point.value!r} in {self.group!r} is not a Registry")
        return registry


registries = Catalog("policies.registries")
