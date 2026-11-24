"""Export an actor to ONNX, one tensor per part, with its spec embedded."""

import contextlib
import copy
import io
import json
import logging
import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import onnx
import torch
from torch import Tensor, nn

from policies.deploy.export.canonical import canonical_spec
from policies.deploy.export.observations import observations
from policies.deploy.export.source import Exportable
from policies.deploy.inference.metadata import (
    FORMAT_VERSION,
    MetadataKey,
    input_tensor,
    next_state_tensor,
    output_tensor,
    state_tensor,
)
from policies.deploy.inference.requirements import requirements
from policies.learning import Actor


class ExportModule(nn.Module):
    """Positional wrapper: input parts and state in, raw outputs and state out."""

    def __init__(self, actor: Actor, input_names: list[str], state_names: list[str]) -> None:
        super().__init__()
        self.actor = actor
        self.input_names, self.state_names = input_names, state_names

    def forward(self, *tensors: Tensor) -> tuple[Tensor, ...]:
        count = len(self.input_names)
        parts = dict(zip(self.input_names, tensors[:count], strict=True))
        state = dict(zip(self.state_names, tensors[count:], strict=True))
        raw, state = self.actor.raw(parts, state)
        return (*raw.values(), *(state[name] for name in self.state_names))


QUIET_LOGGERS = ("torch.onnx", "torch.utils.flop_counter")
"""Torch loggers that warn about optional packages (torchvision, triton) on every export."""


@contextlib.contextmanager
def _quiet_torch() -> Iterator[None]:
    loggers = [logging.getLogger(name) for name in QUIET_LOGGERS]
    levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        for logger, level in zip(loggers, levels, strict=True):
            logger.setLevel(level)


def _dtype(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def export_onnx(
    source: Exportable, path: str | Path, required: Mapping[str, str] | None = None
) -> Path:
    """Write `policy.onnx`. Raw outputs are what each part's distribution is built from.

    `required` are the spec's `requirements`, computed here when not given.
    """
    path = Path(path)
    # Traces a copy in eval mode, so the caller's actor keeps its device and training mode.
    actor = copy.deepcopy(source.actor)
    spec = actor.spec
    input_names = list(spec.inputs.names)
    # A batch of 1 would specialize the batch axis to a constant, so trace with 2.
    state = actor.model.initial_state((2,))
    state_names = list(state)
    # Before tracing, so an artifact that could not load is never written.
    values = metadata(source, {name: _dtype(value) for name, value in state.items()}, required)
    module = ExportModule(actor, input_names, state_names).cpu().eval()
    example = [
        torch.zeros(2, *part.shape, dtype=getattr(torch, part.dtype)) for part in spec.inputs.parts
    ]
    example += [state[name] for name in state_names]
    # Only the first tensor names the axis. Export unifies the rest, and naming them all
    # warns that the shared name is redundant.
    batch = [torch.export.Dim("batch"), *[torch.export.Dim.DYNAMIC] * (len(example) - 1)]
    output_names = [
        *(output_tensor(name) for name in spec.outputs.names),
        *(next_state_tensor(name) for name in state_names),
    ]
    # The exporter prints progress with emoji, which breaks non-UTF-8 consoles.
    with (
        torch.no_grad(),
        contextlib.redirect_stdout(io.StringIO()),
        warnings.catch_warnings(),
        _quiet_torch(),
    ):
        # Tracing torch's own recurrent layers emits these. Neither changes the graph.
        warnings.filterwarnings("ignore", "The tensor attributes .* were assigned during export")
        warnings.filterwarnings("ignore", "_check_is_size will be removed", FutureWarning)
        torch.onnx.export(
            module,
            tuple(example),
            str(path),
            input_names=[*map(input_tensor, input_names), *map(state_tensor, state_names)],
            output_names=output_names,
            dynamic_shapes=(tuple({0: axis} for axis in batch),),
            dynamo=True,
            external_data=False,
            verbose=False,
        )
    model = onnx.load(str(path))
    _set_metadata(model, values)
    onnx.save(model, str(path))
    return path


def metadata(
    source: Exportable, state_dtypes: Mapping[str, str], required: Mapping[str, str] | None = None
) -> dict[str, str]:
    """ONNX metadata entries. Only observations that bookkeeping reads are kept."""
    spec = canonical_spec(source.actor.spec)
    observation_shapes, observation_dtypes = observations(source)
    values: Mapping[MetadataKey, Any] = {
        MetadataKey.FORMAT: FORMAT_VERSION,
        MetadataKey.SPEC: spec.model_dump(mode="json"),
        MetadataKey.FINGERPRINT: spec.fingerprint(),
        MetadataKey.OBSERVATION_SHAPES: observation_shapes,
        MetadataKey.OBSERVATION_DTYPES: observation_dtypes,
        MetadataKey.STATE_SHAPES: source.actor.model.state_shapes,
        MetadataKey.STATE_DTYPES: dict(state_dtypes),
        MetadataKey.REQUIREMENTS: dict(requirements(spec) if required is None else required),
    }
    return {key.value: json.dumps(value) for key, value in values.items()}


def _set_metadata(model: onnx.ModelProto, values: Mapping[str, str]) -> None:
    for key, value in values.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
