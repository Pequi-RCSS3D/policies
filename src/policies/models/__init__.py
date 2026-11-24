"""Models built from parts. Requires the torch extra."""

from policies.models.auxiliary import AuxiliaryLosses, record, recording
from policies.models.base import Model
from policies.models.blocks import Block, blocks
from policies.models.encoders import Encoder, encoders
from policies.models.heads import Head, heads
from policies.models.layers import mlp
from policies.models.parts import PartsModel
from policies.models.state import State, Stateful, namespaced, run_scoped
from policies.spec.model import models

__all__ = [
    "AuxiliaryLosses",
    "Block",
    "Encoder",
    "Head",
    "Model",
    "PartsModel",
    "State",
    "Stateful",
    "blocks",
    "encoders",
    "heads",
    "mlp",
    "models",
    "namespaced",
    "record",
    "recording",
    "run_scoped",
]
