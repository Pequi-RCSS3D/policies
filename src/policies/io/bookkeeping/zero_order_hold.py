"""Zero-order hold discretization of a continuous linear system."""

import math

import numpy as np

TERMS = 20
"""Taylor terms of the matrix exponential, past float64 precision once the norm is at most 1/2."""


def _exponential(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential by scaling and squaring a Taylor series."""
    norm = float(np.linalg.norm(matrix, 1))
    squarings = max(0, math.ceil(math.log2(norm * 2))) if norm > 0 else 0
    scaled = matrix / 2**squarings
    term = result = np.eye(len(matrix))
    for k in range(1, TERMS):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def zero_order_hold(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """`(A_d, B_d)` of `c <- A_d c + B_d x` for `dc/dt = A c + B x` with `x` held over `dt`."""
    n = len(a)
    augmented = np.zeros((n + 1, n + 1))
    augmented[:n, :n] = a * dt
    augmented[:n, n] = b * dt
    exponential = _exponential(augmented)
    return exponential[:n, :n], exponential[:n, n]
