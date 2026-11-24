"""One head per output part. Register more with `heads.register`."""

from policies.models.heads.base import Head
from policies.models.heads.linear import linear_head
from policies.models.heads.mlp import MLPHead
from policies.spec.model.heads import heads

__all__ = ["Head", "MLPHead", "heads", "linear_head"]
