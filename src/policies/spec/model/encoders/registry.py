"""The encoder registry, importable without torch."""

from typing import TYPE_CHECKING, Annotated, Any

from policies.core.registry import Registry

if TYPE_CHECKING:
    from policies.models.encoders import Encoder

encoders: "Registry[Encoder]" = Registry("encoder")

type EncoderConfig = Annotated[dict[str, Any], encoders]
"""Any encoder's config, validated to its canonical form."""
