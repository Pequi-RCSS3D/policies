"""Head configs, registered with factories imported on first build."""

from policies.spec.model.heads.linear import LinearHeadConfig
from policies.spec.model.heads.mlp import MLPHeadConfig
from policies.spec.model.heads.registry import HeadConfig, heads

__all__ = ["HeadConfig", "LinearHeadConfig", "MLPHeadConfig", "heads"]
