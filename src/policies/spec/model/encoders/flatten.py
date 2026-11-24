"""Flatten encoder options."""

from policies.core.config import Config
from policies.spec.model.encoders.registry import encoders


class FlattenConfig(Config):
    pass


encoders.register_lazy("flatten", FlattenConfig, "policies.models.encoders.flatten:Flatten")
