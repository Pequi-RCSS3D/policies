"""A registered component, as listed."""

from dataclasses import dataclass

from policies.core.config import Config
from policies.core.registry.origin import Origin


@dataclass(frozen=True, slots=True)
class Entry:
    """A component's name, config class and the distribution that registered it."""

    name: str
    config: type[Config]
    origin: Origin | None
