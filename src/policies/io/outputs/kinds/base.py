"""The output kind contract and registry."""

from collections.abc import Mapping
from typing import Annotated, Any, Protocol, runtime_checkable

from policies.core.registry import Registry


class Kind(Protocol):
    """`shape` is the action shape per sample. `distribution` is the default distribution
    config, which a part may override. `decode` turns actions into output values.

    A kind may also be a `KindWithExtras`.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def distribution(self) -> Mapping[str, Any]: ...

    def decode(self, action: Any, xp: Any) -> Any: ...


@runtime_checkable
class KindWithExtras(Protocol):
    """A kind with per-step data next to its decoded value, such as controller gains."""

    def extras(self, action: Any, xp: Any) -> Mapping[str, Any]:
        """Named values for this step's `action`, beside what `decode` returns."""
        ...


kinds: Registry[Kind] = Registry("output kind")

type KindConfig = Annotated[dict[str, Any], kinds]
"""An output kind config, validated and stored canonical."""
