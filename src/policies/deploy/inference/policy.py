"""Run an exported policy with NumPy and ONNX Runtime only."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import array_api_compat.numpy as numpy_xp
import numpy as np
import onnxruntime
from pydantic import ValidationError

from policies.core.logs import get_logger
from policies.deploy.inference.metadata import (
    FORMAT_VERSION,
    MetadataKey,
    input_tensor,
    next_state_tensor,
    output_tensor,
    state_tensor,
)
from policies.deploy.inference.requirements import unmet
from policies.deploy.inference.sources import MissingObservationsError, tracker_sources
from policies.io.bookkeeping import Compose, build_bookkeeping
from policies.io.outputs import Kind, OutputDecoder
from policies.spec import PolicySpec

log = get_logger(__name__)


class PolicyLoadError(ValueError):
    """The artifact is missing, malformed or inconsistent."""


@dataclass(frozen=True)
class PolicyStep:
    """Everything one `Policy.act` produced."""

    outputs: dict[str, Any]
    """Decoded values by part, what `Policy.step` returns."""
    actions: dict[str, Any]
    """Actions by part: the mode of each part's distribution, before decoding."""
    extras: dict[str, dict[str, Any]]
    """Per-step extras by part, from kinds that have them. See `KindWithExtras`."""


def _onnx_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    return path / "policy.onnx" if path.is_dir() else path


def _spec(file: Path, data: Any, required: Mapping[str, str]) -> PolicySpec:
    """Validated, naming what to install when components are unknown.

    Other versions than at export only warn, since the fingerprint guards behavior.
    """
    differing = unmet(required)
    try:
        spec = PolicySpec.model_validate(data)
    except ValidationError as error:
        if not differing:
            raise PolicyLoadError(f"{file} has an invalid spec: {error}") from None
        hints = ", ".join(
            f"{name}=={required[name]} ({f'found {version}' if version else 'not installed'})"
            for name, version in differing.items()
        )
        raise PolicyLoadError(
            f"{file} uses components that are not installed. Install the versions it was "
            f"exported with: {hints}.\n{error}"
        ) from None
    for name, version in differing.items():
        log.warning(
            "installed version differs from export",
            distribution=name,
            exported=required[name],
            installed=version,
        )
    return spec


def _unique(names: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


class Policy:
    """One agent at a time: `reset()`, then `step(observation)` each cycle."""

    def __init__(self, path: str | Path, threads: int = 1) -> None:
        file = _onnx_path(path)
        if not file.exists():
            raise PolicyLoadError(f"no policy at {file}")
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = threads
        self.session = onnxruntime.InferenceSession(
            str(file), options, providers=["CPUExecutionProvider"]
        )
        metadata_map = self.session.get_modelmeta().custom_metadata_map
        # Checked first, since other formats may lack keys.
        found = json.loads(metadata_map.get(MetadataKey.FORMAT, json.dumps(FORMAT_VERSION)))
        if found != FORMAT_VERSION:
            raise PolicyLoadError(f"{file} has format {found}, expected {FORMAT_VERSION}")
        # Exported before requirements were recorded. Readers need not know them to run.
        metadata_map.setdefault(MetadataKey.REQUIREMENTS, "{}")
        try:
            values = {key: json.loads(metadata_map[key]) for key in MetadataKey}
        except KeyError as missing:
            raise PolicyLoadError(f"{file} lacks policy metadata {missing}") from None
        self.requirements: dict[str, str] = values[MetadataKey.REQUIREMENTS]
        self.spec = _spec(file, values[MetadataKey.SPEC], self.requirements)
        self.fingerprint: str = values[MetadataKey.FINGERPRINT]
        if self.spec.fingerprint() != self.fingerprint:
            raise PolicyLoadError(f"{file}: embedded fingerprint does not match its spec")
        self.observation_shapes = {
            k: tuple(v) for k, v in values[MetadataKey.OBSERVATION_SHAPES].items()
        }
        self.observation_dtypes: dict[str, str] = values[MetadataKey.OBSERVATION_DTYPES]
        self.state_shapes = {k: tuple(v) for k, v in values[MetadataKey.STATE_SHAPES].items()}
        self.state_dtypes: dict[str, str] = values[MetadataKey.STATE_DTYPES]
        self.input_dtypes = {part.name: np.dtype(part.dtype) for part in self.spec.inputs.parts}
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.inputs = self.spec.inputs.compile()
        self.decoder = OutputDecoder(self.spec.outputs)
        try:
            tracked = tracker_sources(
                self.spec.bookkeeping,
                self.decoder,
                self.observation_shapes,
                self.observation_dtypes,
            )
        except MissingObservationsError as error:
            raise PolicyLoadError(f"{file}: {error}. Re-export it.") from None
        self.bookkeeping: Compose
        self.state: dict[str, np.ndarray] = {}
        self.reset()
        slots = self.bookkeeping.slots
        self.observation_keys = _unique(
            [part.source for part in self.spec.inputs.parts if part.source not in slots]
            + list(tracked)
        )
        """Exactly the observations `step` reads: input part sources that are not bookkeeping
        slots, then those trackers read."""

    @classmethod
    def load(cls, path: str | Path, threads: int = 1) -> "Policy":
        return cls(path, threads)

    @property
    def kinds(self) -> dict[str, Kind]:
        """The output kind of each part, as built for decoding."""
        return self.decoder.kinds

    @property
    def kind_types(self) -> dict[str, str]:
        """The registered kind name of each output part."""
        return {part.name: part.kind_type for part in self.spec.outputs.parts}

    def initial_state(self, batch: int) -> dict[str, np.ndarray]:
        return {
            name: np.zeros((batch, *shape), self.state_dtypes[name])
            for name, shape in self.state_shapes.items()
        }

    def initial_bookkeeping(self, batch: tuple[int, ...]) -> Compose:
        """Fresh bookkeeping for observations and actions with `batch` in front."""
        return build_bookkeeping(
            self.spec.bookkeeping,
            shapes={**self.observation_shapes, **self.decoder.shapes},
            dtypes={**self.observation_dtypes, **self.decoder.dtypes},
            batch=batch,
            xp=numpy_xp,
        )

    def reset(self) -> None:
        """Start a new episode: clears bookkeeping and recurrent state."""
        self.bookkeeping = self.initial_bookkeeping(())
        self.state = self.initial_state(1)

    def run(
        self, parts: Mapping[str, Any], state: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Raw outputs and the next state for a batch of input parts. No bookkeeping."""
        feed = {
            input_tensor(name): np.asarray(value, self.input_dtypes[name])
            for name, value in parts.items()
        }
        feed.update({state_tensor(name): value for name, value in state.items()})
        values = dict(zip(self.output_names, self.session.run(None, feed), strict=True))
        raw = {name: values[output_tensor(name)] for name in self.spec.outputs.names}
        return raw, {name: values[next_state_tensor(name)] for name in self.state_shapes}

    def _run(self, parts: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """One sample through `run` with the policy's own state."""
        raw, state = self.run(
            {name: np.asarray(value)[None] for name, value in parts.items()}, self.state
        )
        return {name: value[0] for name, value in raw.items()}, state

    def _actions(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """The mode of each part for one cycle, advancing bookkeeping and state."""
        slots = self.bookkeeping.observe(observation)
        raw, self.state = self._run(self.inputs(observation, slots))
        actions = self.decoder.mode(raw)
        self.bookkeeping.commit(actions)
        return actions

    def step(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Decoded outputs for one cycle."""
        return self.decoder.decode(self._actions(observation))

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        """Like `step`, also returning actions and the extras of kinds that have them."""
        actions = self._actions(observation)
        return PolicyStep(self.decoder.decode(actions), actions, self.decoder.extras(actions))
