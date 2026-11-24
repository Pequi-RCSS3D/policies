"""One named choice per element, each from its own options."""

from typing import Annotated, Any

from pydantic import Field

from policies.core.config import Config
from policies.io.outputs.kinds.base import kinds


class MultiCategoricalConfig(Config):
    choices: tuple[Annotated[tuple[str, ...], Field(min_length=1)], ...] = Field(min_length=1)


class MultiCategorical:
    """Decodes to the index of each element's choice."""

    def __init__(self, config: MultiCategoricalConfig, **_: Any) -> None:
        self.choices = config.choices
        self.shape = (len(config.choices),)
        counts = [len(options) for options in config.choices]
        self.distribution = {"type": "multi_categorical", "choices": counts}

    def decode(self, action: Any, xp: Any) -> Any:
        return action


kinds.register("multi_categorical", MultiCategoricalConfig)(MultiCategorical)
