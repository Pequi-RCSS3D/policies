"""The distributions, with versions, that provide the components a spec uses."""

from collections.abc import Mapping
from importlib import metadata

from pydantic import BaseModel

from policies.core.logs import get_logger
from policies.core.registry import installed_plugins, used_components

log = get_logger(__name__)


def requirements(config: BaseModel) -> dict[str, str]:
    """`{distribution: version}` by name, including the one that provides this package."""
    origins = {installed_plugins.origin(__name__)}
    for registry, name in set(used_components(config)):
        origin = registry.origin(name)
        if origin is None:
            log.warning("component has no distribution to require", kind=registry.kind, name=name)
        origins.add(origin)
    return dict(sorted((origin.distribution, origin.version) for origin in origins if origin))


def installed(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def unmet(requirements: Mapping[str, str]) -> dict[str, str | None]:
    """Required distributions installed at another version, or None when missing."""
    found = {name: installed(name) for name in requirements}
    return {name: version for name, version in found.items() if version != requirements[name]}
