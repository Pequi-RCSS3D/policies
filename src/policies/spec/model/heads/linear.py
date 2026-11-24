"""Linear head options."""

from policies.core.config import Config
from policies.spec.model.heads.registry import heads


class LinearHeadConfig(Config):
    initial_scale: float = 0.01


heads.register_lazy("linear", LinearHeadConfig, "policies.models.heads.linear:linear_head")
