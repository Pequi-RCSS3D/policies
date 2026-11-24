"""The head registry, importable without torch."""

from typing import TYPE_CHECKING, Annotated, Any

from policies.core.registry import Registry

if TYPE_CHECKING:
    from torch import nn

heads: "Registry[nn.Module]" = Registry("head")

type HeadConfig = Annotated[dict[str, Any], heads]
"""Any head's config, validated to its canonical form."""
