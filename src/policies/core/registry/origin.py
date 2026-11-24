"""Where a registered component comes from."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Origin:
    """The installed distribution that registered a component."""

    distribution: str
    version: str

    def __str__(self) -> str:
        return f"{self.distribution} {self.version}"
