"""The shared computation between encoders and heads. Register more with `blocks.register`."""

from policies.models.blocks.base import Block
from policies.models.blocks.gru import GRUBlock
from policies.models.blocks.lstm import LSTMBlock
from policies.models.blocks.mixture import MixtureBlock
from policies.models.blocks.mlp import MLPBlock
from policies.models.blocks.sequential import SequentialBlock
from policies.models.blocks.transformer import TransformerBlock
from policies.spec.model.blocks import blocks

__all__ = [
    "Block",
    "GRUBlock",
    "LSTMBlock",
    "MLPBlock",
    "MixtureBlock",
    "SequentialBlock",
    "TransformerBlock",
    "blocks",
]
