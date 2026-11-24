"""Encoder configs, registered with factories imported on first build."""

from policies.spec.model.encoders.block import BlockEncoderConfig
from policies.spec.model.encoders.flatten import FlattenConfig
from policies.spec.model.encoders.mlp import MLPEncoderConfig
from policies.spec.model.encoders.registry import EncoderConfig, encoders

__all__ = [
    "BlockEncoderConfig",
    "EncoderConfig",
    "FlattenConfig",
    "MLPEncoderConfig",
    "encoders",
]
