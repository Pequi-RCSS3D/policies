"""Picks the training device."""

import torch

AUTO = "auto"


def resolve_device(device: str) -> torch.device:
    """`auto` is CUDA when available, else the CPU. Anything else is a torch device string."""
    if device == AUTO:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
