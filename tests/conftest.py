import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

type Make = Callable[..., Any]


def _numpy(shape: tuple[int, ...], dtype: str = "float32", fill: float = 0.0) -> Any:
    return np.full(shape, fill, dtype=dtype)


def _torch(shape: tuple[int, ...], dtype: str = "float32", fill: float = 0.0) -> Any:
    import torch

    return torch.full(shape, fill, dtype=getattr(torch, dtype))


@pytest.fixture(params=["numpy", "torch"])
def make(request: pytest.FixtureRequest) -> Make:
    """Array factory for each backend. The torch one is skipped without the extra."""
    if request.param == "torch":
        pytest.importorskip("torch")
        return _torch
    return _numpy


TORCH_ONLY = ["learning/*", "models/*"]
"""Test modules that need the torch extra, skipped at collection without it."""

collect_ignore_glob = [] if importlib.util.find_spec("torch") else TORCH_ONLY
