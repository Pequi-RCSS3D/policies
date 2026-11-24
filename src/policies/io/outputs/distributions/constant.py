"""A NumPy constant on the namespace, device and dtype of other arrays, cached."""

from typing import Any

from array_api_compat import array_namespace, device
from numpy.typing import NDArray


class Constant:
    def __init__(self, value: NDArray[Any]) -> None:
        self.value = value
        self._cache: dict[tuple[str, str], Any] = {}

    def like(self, other: Any) -> Any:
        """The constant as an array of `other`'s kind, converted once per device and dtype."""
        key = (str(device(other)), str(other.dtype))
        if key not in self._cache:
            xp = array_namespace(other)
            self._cache[key] = xp.asarray(self.value, dtype=other.dtype, device=device(other))
        return self._cache[key]
