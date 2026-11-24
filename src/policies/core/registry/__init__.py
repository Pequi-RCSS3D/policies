"""Building components by name."""

from policies.core.registry.base import Registry
from policies.core.registry.catalog import Catalog, registries
from policies.core.registry.entry import Entry
from policies.core.registry.errors import (
    DuplicateComponentError,
    PluginError,
    UnknownComponentError,
)
from policies.core.registry.origin import Origin
from policies.core.registry.plugins import PluginGroup, installed_plugins
from policies.core.registry.usage import Use, used_components

__all__ = [
    "Catalog",
    "DuplicateComponentError",
    "Entry",
    "Origin",
    "PluginError",
    "PluginGroup",
    "Registry",
    "UnknownComponentError",
    "Use",
    "installed_plugins",
    "registries",
    "used_components",
]
