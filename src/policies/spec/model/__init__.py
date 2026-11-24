"""Model configs, the model registry and the configs of model components. Needs no torch."""

from policies.spec.model.activations import ActivationConfig, activations
from policies.spec.model.blocks import (
    BlockConfig,
    GRUBlockConfig,
    LSTMBlockConfig,
    MixtureBlockConfig,
    MLPBlockConfig,
    Pooling,
    SequentialBlockConfig,
    Tokens,
    TransformerBlockConfig,
    blocks,
)
from policies.spec.model.encoders import (
    BlockEncoderConfig,
    EncoderConfig,
    FlattenConfig,
    MLPEncoderConfig,
    encoders,
)
from policies.spec.model.heads import HeadConfig, LinearHeadConfig, MLPHeadConfig, heads
from policies.spec.model.parts import PartsModelConfig
from policies.spec.model.registry import (
    DEFAULT_MODEL,
    ModelConfig,
    default_model_type,
    models,
)

__all__ = [
    "DEFAULT_MODEL",
    "ActivationConfig",
    "BlockConfig",
    "BlockEncoderConfig",
    "EncoderConfig",
    "FlattenConfig",
    "GRUBlockConfig",
    "HeadConfig",
    "LSTMBlockConfig",
    "LinearHeadConfig",
    "MLPBlockConfig",
    "MLPEncoderConfig",
    "MLPHeadConfig",
    "MixtureBlockConfig",
    "ModelConfig",
    "PartsModelConfig",
    "Pooling",
    "SequentialBlockConfig",
    "Tokens",
    "TransformerBlockConfig",
    "activations",
    "blocks",
    "default_model_type",
    "encoders",
    "heads",
    "models",
]
