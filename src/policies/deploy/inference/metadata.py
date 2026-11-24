"""Metadata keys and tensor names shared by export and inference."""

from enum import StrEnum

FORMAT_VERSION = "2"


class MetadataKey(StrEnum):
    """Keys of the ONNX metadata entries."""

    FORMAT = "policies.format"
    SPEC = "policies.spec"
    FINGERPRINT = "policies.fingerprint"
    OBSERVATION_SHAPES = "policies.observation_shapes"
    OBSERVATION_DTYPES = "policies.observation_dtypes"
    STATE_SHAPES = "policies.state_shapes"
    STATE_DTYPES = "policies.state_dtypes"
    REQUIREMENTS = "policies.requirements"


def input_tensor(part: str) -> str:
    return f"input.{part}"


def output_tensor(part: str) -> str:
    return f"output.{part}"


def state_tensor(name: str) -> str:
    return f"state.{name}"


def next_state_tensor(name: str) -> str:
    return f"state.{name}.next"
