"""One of several named choices."""

from typing import Any

from pydantic import Field

from policies.core.config import Config
from policies.io.outputs.kinds.base import kinds


class CategoricalConfig(Config):
    choices: tuple[str, ...] = Field(min_length=1)


class Categorical:
    """Decodes to the index of the choice."""

    shape: tuple[int, ...] = ()

    def __init__(self, config: CategoricalConfig, **_: Any) -> None:
        self.choices = config.choices
        self.distribution = {"type": "categorical", "choices": len(config.choices)}

    def decode(self, action: Any, xp: Any) -> Any:
        return action


kinds.register("categorical", CategoricalConfig)(Categorical)
