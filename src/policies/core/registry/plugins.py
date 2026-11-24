"""Installed plugins, discovered through one Python entry-point group.

A distribution adds components to any registry by naming a module that registers them:

    [project.entry-points."policies.plugins"]
    my_package = "my_package.components"
"""

from functools import cached_property
from importlib import metadata

from policies.core.logs import get_logger
from policies.core.registry.errors import PluginError
from policies.core.registry.origin import Origin

log = get_logger(__name__)


def _describe(point: metadata.EntryPoint) -> str:
    source = f"{point.dist.name} {point.dist.version}" if point.dist else "an unknown distribution"
    return f"entry point {point.name!r} = {point.value!r} from {source}"


class PluginGroup:
    """An entry-point group whose modules register components when imported. Loads once."""

    def __init__(self, group: str) -> None:
        self.group = group
        self._loaded = False
        self.loading = False
        self._failure: PluginError | None = None
        self._origins: dict[str, Origin | None] = {}

    @cached_property
    def _entry_points(self) -> list[metadata.EntryPoint]:
        # Sorted so that `replace=True` overrides resolve the same way on every machine.
        return sorted(metadata.entry_points(group=self.group), key=lambda point: point.name)

    def load(self) -> None:
        """Import every entry point. Later calls do nothing, or re-raise the failure."""
        if self._failure is not None:
            raise self._failure
        if self._loaded:
            return
        # Set first, so a plugin that looks up names while importing does not recurse.
        self._loaded = True
        self.loading = True
        failures: list[tuple[metadata.EntryPoint, Exception]] = []
        try:
            for point in self._entry_points:
                try:
                    point.load()
                except Exception as error:  # any plugin bug, reported below with its source
                    failures.append((point, error))
                else:
                    log.debug("plugin loaded", group=self.group, entry_point=_describe(point))
        finally:
            self.loading = False
        if failures:
            lines = "".join(
                f"\n  {_describe(point)}: {type(error).__name__}: {error}"
                for point, error in failures
            )
            self._failure = PluginError(f"plugins in {self.group!r} failed to load:{lines}")
            raise self._failure from failures[0][1]

    def origin(self, module: str) -> Origin | None:
        """The distribution that ships `module`, or None for unpackaged code."""
        package = module.partition(".")[0]
        if package not in self._origins:
            self._origins[package] = self._find(package)
        return self._origins[package]

    def _find(self, package: str) -> Origin | None:
        # Entry points first, since editable installs hide their packages from file listings.
        for point in self._entry_points:
            if point.dist is not None and point.module.partition(".")[0] == package:
                return Origin(point.dist.name, point.dist.version)
        try:
            distribution = metadata.distribution(package)
        except metadata.PackageNotFoundError:
            return None
        return Origin(distribution.name, distribution.version)


installed_plugins = PluginGroup("policies.plugins")
