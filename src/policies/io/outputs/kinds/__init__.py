"""Output kinds. Register more with `kinds.register`."""

from policies.io.outputs.kinds.base import Kind, KindConfig, KindWithExtras, kinds
from policies.io.outputs.kinds.binary import Binary
from policies.io.outputs.kinds.categorical import Categorical
from policies.io.outputs.kinds.continuous import Continuous
from policies.io.outputs.kinds.multi_categorical import MultiCategorical
from policies.io.outputs.kinds.scalar import Scalar
from policies.io.outputs.kinds.shaped import ShapedConfig

__all__ = [
    "Binary",
    "Categorical",
    "Continuous",
    "Kind",
    "KindConfig",
    "KindWithExtras",
    "MultiCategorical",
    "Scalar",
    "ShapedConfig",
    "kinds",
]
