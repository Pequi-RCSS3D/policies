"""Observation shapes and dtypes, from observations or from recorded input parts."""

from collections.abc import Mapping
from typing import Any

from policies.data import Batch
from policies.io.inputs import Inputs


def dtype_name(value: Any) -> str:
    """`float32` for NumPy and PyTorch arrays alike."""
    return str(value.dtype).removeprefix("torch.")


def observed(
    observation: Mapping[str, Any], leading: int = 1
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Per-sample shapes and dtypes, past `leading` batch dimensions."""
    shapes = {name: tuple(value.shape[leading:]) for name, value in observation.items()}
    return shapes, {name: dtype_name(value) for name, value in observation.items()}


def copied_observations(
    inputs: Inputs, batch: Batch
) -> tuple[dict[str, tuple[int, ...]], dict[str, str]]:
    """Shapes and dtypes of the sources that untransformed input parts copy, by source.

    Sources that are bookkeeping slots are included, and dropped again at export.
    """
    copied = {
        part.source: batch.inputs[part.name]
        for part in inputs.parts
        if not part.transforms and part.name in batch.inputs
    }
    return observed(copied, 1 if batch.reset is None else 2)
