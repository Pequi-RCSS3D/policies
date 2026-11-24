"""Check an artifact: consistent metadata and, with PyTorch installed, ONNX parity."""

import importlib.util
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from policies.deploy.inference.policy import Policy, PolicyLoadError
from policies.deploy.inference.validation import ParitySkip, Validation


def validate_artifact(path: str | Path, tolerance: float = 1e-4, samples: int = 8) -> Validation:
    """Raise `PolicyLoadError` on any inconsistency. Returns whether parity ran."""
    path = Path(path)
    policy = Policy.load(path)
    manifest_file = path / "manifest.json" if path.is_dir() else None
    if manifest_file and manifest_file.exists():
        from policies.deploy.registry import read_manifest

        manifest = read_manifest(path)
        if manifest.fingerprint != policy.fingerprint:
            raise PolicyLoadError("manifest fingerprint differs from the ONNX metadata")
    weights = path / "weights.pt"
    if not (path.is_dir() and weights.is_file()):
        return Validation(ParitySkip.NO_WEIGHTS)
    if importlib.util.find_spec("torch") is None:
        return Validation(ParitySkip.NO_TORCH)
    _check_parity(policy, weights, tolerance, samples)
    return Validation()


def _compare(
    onnx: Mapping[str, np.ndarray], torch: Mapping[str, np.ndarray], tolerance: float, what: str
) -> None:
    for name, value in onnx.items():
        difference = float(np.abs(value - torch[name]).max())
        if difference > tolerance:
            raise PolicyLoadError(f"{what} {name!r} differs from PyTorch by {difference:.2e}")


def _sample(generator: np.random.Generator, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    """Random values of a dtype. Integers are 0 or 1, valid for any index-like input."""
    if np.dtype(dtype).kind == "f":
        return generator.standard_normal(shape).astype(dtype)
    return generator.integers(0, 2, shape).astype(dtype)


def _check_parity(policy: Policy, weights: Path, tolerance: float, samples: int) -> None:
    """One batch of `samples` with random state, then the single-sample path of `Policy`."""
    import torch

    from policies.learning import Actor

    saved = torch.load(weights, map_location="cpu", weights_only=True)
    actor = Actor(policy.spec, **saved["settings"])
    actor.load_state_dict(saved["actor"])
    actor.eval()

    def run_torch(
        parts: Mapping[str, np.ndarray], state: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        with torch.no_grad():
            raw, after = actor.raw(
                {name: torch.from_numpy(value) for name, value in parts.items()},
                {name: torch.from_numpy(value) for name, value in state.items()},
            )
        return (
            {name: value.numpy() for name, value in raw.items()},
            {name: value.numpy() for name, value in after.items()},
        )

    generator = np.random.default_rng(0)
    parts = {
        part.name: _sample(generator, (samples, *part.shape), part.dtype)
        for part in policy.spec.inputs.parts
    }
    recurrent = {
        name: _sample(generator, (samples, *shape), policy.state_dtypes[name])
        for name, shape in policy.state_shapes.items()
    }
    onnx_raw, onnx_state = policy.run(parts, recurrent)
    torch_raw, torch_state = run_torch(parts, recurrent)
    _compare(onnx_raw, torch_raw, tolerance, "batched output")
    _compare(onnx_state, torch_state, tolerance, "batched state")
    policy.reset()
    first = {name: value[:1] for name, value in parts.items()}
    onnx_raw, _ = policy._run({name: value[0] for name, value in first.items()})
    torch_raw, _ = run_torch(first, policy.initial_state(1))
    _compare(onnx_raw, {name: value[0] for name, value in torch_raw.items()}, tolerance, "output")
