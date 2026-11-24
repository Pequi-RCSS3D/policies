"""What `validate_artifact` checked."""

from dataclasses import dataclass
from enum import StrEnum


class ParitySkip(StrEnum):
    """Why ONNX Runtime was not checked against PyTorch."""

    NO_WEIGHTS = "no weights.pt"
    NO_TORCH = "torch is not installed"


@dataclass(frozen=True)
class Validation:
    skipped: ParitySkip | None = None
    """Why parity did not run. None when it ran and passed."""

    @property
    def parity(self) -> bool:
        """True when ONNX Runtime was checked against PyTorch."""
        return self.skipped is None
