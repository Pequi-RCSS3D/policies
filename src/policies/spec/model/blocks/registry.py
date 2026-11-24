"""The block registry, importable without torch."""

from typing import TYPE_CHECKING, Annotated, Any

from policies.core.registry import Registry

if TYPE_CHECKING:
    from policies.models.blocks import Block

blocks: "Registry[Block]" = Registry("block")

type BlockConfig = Annotated[dict[str, Any], blocks]
"""Any block's config, validated to its canonical form."""
