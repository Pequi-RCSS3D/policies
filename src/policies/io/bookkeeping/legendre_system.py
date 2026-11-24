"""The continuous linear system of a Legendre memory."""

import numpy as np


def legendre_system(window: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """`(A, B)` of `dc/dt = A c + B x`, the LMU system (Voelker et al. 2019), as HiPPO-LegT.

    `c` projects the last `window` seconds of `x` onto the first `order` Legendre polynomials.
    """
    i = np.arange(order)[:, None]
    j = np.arange(order)[None, :]
    a = np.where(i < j, -1.0, (-1.0) ** (i - j + 1)) * (2 * i + 1) / window
    b = (2 * np.arange(order) + 1) * (-1.0) ** np.arange(order) / window
    return a, b
