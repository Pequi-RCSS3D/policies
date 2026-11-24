"""What a Legendre memory holds after a reset."""

from enum import StrEnum

import numpy as np


class LegendreReset(StrEnum):
    ZERO = "zero"
    """Forgets the window, as if the source had been zero."""
    HOLD = "hold"
    """Fills the window with the next input, as if it had been constant."""

    def start(self, order: int) -> np.ndarray:
        """Coefficients after a reset, per unit of the next input."""
        start = np.zeros(order)
        start[0] = 1.0 if self is LegendreReset.HOLD else 0.0
        return start
