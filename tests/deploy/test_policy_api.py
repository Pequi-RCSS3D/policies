import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest

from policies.core.config import Config
from policies.io.bookkeeping import TrackerContext, trackers

torch = pytest.importorskip("torch")

from policies.deploy.export import ExportSource, export_artifact  # noqa: E402  # needs the torch extra
from policies.deploy.inference import (  # noqa: E402  # same
    ParitySkip,
    Policy,
    PolicyLoadError,
    PolicyStep,
    Validation,
    validate_artifact,
)
from policies.deploy.inference.metadata import MetadataKey  # noqa: E402  # same
from policies.learning import Actor, Learner  # noqa: E402  # same
from policies.spec import PolicySpec  # noqa: E402  # same
from tests.io.test_outputs import GAINED, gained  # noqa: E402  # same

SPEC = PolicySpec.model_validate(
    {
        "name": "keys",
        "inputs": {
            "parts": [
                {"name": "state", "source": "state", "shape": [2]},
                {"name": "past", "source": "history.position", "shape": [3, 1]},
                {"name": "previous", "source": "last.force", "shape": [2]},
            ]
        },
        "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 2}}]},
        "bookkeeping": [
            {"type": "history", "source": "position", "length": 3},
            {"type": "last", "source": "force"},
        ],
    }
)


def test_observation_keys_are_what_the_policy_reads(tmp_path: Path) -> None:
    learner = Learner(SPEC, num_envs=2)
    # `extra` stands for observations only the critic or objectives read.
    learner.act(
        {"state": torch.zeros(2, 2), "position": torch.zeros(2, 1), "extra": torch.zeros(2, 5)}
    )
    export_artifact(learner, tmp_path)
    policy = Policy.load(tmp_path)
    assert policy.observation_keys == ("state", "position")
    assert policy.observation_shapes == {"position": (1,)}
    assert policy.observation_dtypes == {"position": "float32"}


def test_export_refuses_unknown_observations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"unknown shape: \['position'\]. Act once"):
        export_artifact(ExportSource(Actor(SPEC)), tmp_path)
    assert not (tmp_path / "policy.onnx").exists()


def test_loading_needs_the_observations_bookkeeping_reads(tmp_path: Path) -> None:
    export_artifact(ExportSource(Actor(SPEC), {"position": (1,)}), tmp_path)
    model = onnx.load(str(tmp_path / "policy.onnx"))
    for entry in model.metadata_props:
        if entry.key == MetadataKey.OBSERVATION_SHAPES:
            entry.value = json.dumps({})
    onnx.save(model, str(tmp_path / "policy.onnx"))
    with pytest.raises(PolicyLoadError, match=r"unknown shape: \['position'\]"):
        Policy.load(tmp_path)


class PeakConfig(Config):
    source: str


class Peak:
    """The largest value of an observation. Declares its source without needing its shape."""

    def __init__(self, config: PeakConfig, context: TrackerContext) -> None:
        self.context, self.source = context, config.source
        self.sources = (config.source,)
        self.slots = {f"peak.{config.source}": (1,)}

    def reset(self, mask: Any = None) -> None:
        pass

    def observe(self, observation: Mapping[str, Any], update: bool = True) -> dict[str, Any]:
        value = self.context.xp.max(observation[self.source], axis=-1, keepdims=True)
        return {f"peak.{self.source}": value}

    def commit(self, outputs: Mapping[str, Any]) -> None:
        pass


def test_declared_sources_reach_the_artifact(tmp_path: Path) -> None:
    with trackers.scoped():
        trackers.register("tests.peak", PeakConfig)(Peak)
        spec = PolicySpec.model_validate(
            {
                **SPEC.model_dump(),
                "inputs": {"parts": [{"name": "peak", "source": "peak.contact", "shape": [1]}]},
                "bookkeeping": [{"type": "tests.peak", "source": "contact"}],
            }
        )
        with pytest.raises(ValueError, match=r"unknown shape: \['contact'\]"):
            export_artifact(ExportSource(Actor(spec)), tmp_path)
        learner = Learner(spec, num_envs=2)
        learner.act({"contact": torch.zeros(2, 3), "extra": torch.zeros(2, 1)})
        export_artifact(learner, tmp_path)
        policy = Policy.load(tmp_path)
        assert policy.observation_keys == ("contact",)
        assert policy.observation_shapes == {"contact": (3,)}
        assert set(policy.step({"contact": np.ones(3, np.float32)})) == {"force"}


def test_kinds_and_their_extras(tmp_path: Path) -> None:
    with gained():
        spec = PolicySpec.model_validate(
            {
                "name": "gains",
                "inputs": {"parts": [{"name": "state", "source": "state", "shape": [2]}]},
                "outputs": {
                    "parts": [
                        {"name": "joints", "kind": GAINED},
                        {"name": "mode", "kind": {"type": "categorical", "choices": ["a", "b"]}},
                    ]
                },
            }
        )
        export_artifact(ExportSource(Actor(spec)), tmp_path)
        policy = Policy.load(tmp_path)
        assert policy.kind_types == {"joints": "tests.gained", "mode": "categorical"}
        assert policy.kinds["mode"].shape == ()
        step = policy.act({"state": np.zeros(2, np.float32)})
    assert isinstance(step, PolicyStep)
    assert set(step.outputs) == set(step.actions) == {"joints", "mode"}
    assert list(step.extras) == ["joints"]
    np.testing.assert_array_equal(step.extras["joints"]["kp"], [40.0, 40.0])
    assert set(policy.step({"state": np.zeros(2, np.float32)})) == {"joints", "mode"}


def test_validate_reports_whether_parity_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    export_artifact(ExportSource(Actor(SPEC), {"position": (1,)}), tmp_path)
    assert validate_artifact(tmp_path) == Validation()
    assert validate_artifact(tmp_path).parity
    assert validate_artifact(tmp_path / "policy.onnx").skipped == ParitySkip.NO_WEIGHTS
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    result = validate_artifact(tmp_path)
    assert result.skipped == ParitySkip.NO_TORCH
    assert not result.parity
