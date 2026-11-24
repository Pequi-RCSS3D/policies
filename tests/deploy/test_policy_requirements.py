import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import structlog

pytest.importorskip("torch")
import onnx

from policies.core.config import Config
from policies.deploy.export import ExportSource, export_artifact
from policies.deploy.inference import Policy, PolicyLoadError
from policies.deploy.inference.metadata import MetadataKey
from policies.io.inputs import Transform, transforms
from policies.learning import Actor
from tests.deploy.test_requirements import POLICIES, spec


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    export_artifact(ExportSource(Actor(spec({"type": "scale", "factor": 2.0}))), tmp_path)
    return tmp_path


def _rewrite(directory: Path, changes: Mapping[MetadataKey, Any]) -> None:
    """Replaces ONNX metadata entries. None removes one."""
    path = str(directory / "policy.onnx")
    model = onnx.load(path)
    values = {entry.key: entry.value for entry in model.metadata_props}
    for key, value in changes.items():
        if value is None:
            del values[key]
        else:
            values[key] = json.dumps(value)
    del model.metadata_props[:]
    for key, text in values.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, text
    onnx.save(model, path)


def test_export_records_requirements(artifact: Path) -> None:
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["requirements"] == {"policies": POLICIES}
    assert Policy.load(artifact).requirements == {"policies": POLICIES}


def test_missing_plugin_names_the_distribution(artifact: Path) -> None:
    data = Policy.load(artifact).spec.model_dump(mode="json")
    data["inputs"]["parts"][0]["transforms"] = [{"type": "acme.mirror"}]
    required = {"acme": "1.2.0", "policies": POLICIES}
    _rewrite(artifact, {MetadataKey.SPEC: data, MetadataKey.REQUIREMENTS: required})
    with pytest.raises(
        PolicyLoadError, match=r"exported with: acme==1\.2\.0 \(not installed\)"
    ) as e:
        Policy.load(artifact)
    assert "unknown transform 'acme.mirror'" in str(e.value)
    assert str(artifact.resolve()) in str(e.value)


def test_missing_policy_shows_the_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PolicyLoadError, match="no policy at") as caught:
        Policy.load("missing")
    assert str((tmp_path / "missing").resolve()) in str(caught.value)


def test_other_version_warns(artifact: Path) -> None:
    _rewrite(artifact, {MetadataKey.REQUIREMENTS: {"policies": "0.0.1"}})
    with structlog.testing.capture_logs() as logs:
        Policy.load(artifact)
    assert logs == [
        {
            "event": "installed version differs from export",
            "distribution": "policies",
            "exported": "0.0.1",
            "installed": POLICIES,
            "log_level": "warning",
        }
    ]


def test_artifacts_without_requirements_load(artifact: Path) -> None:
    _rewrite(artifact, {MetadataKey.REQUIREMENTS: None})
    assert Policy.load(artifact).requirements == {}


class DoubleConfig(Config):
    pass


def double(config: DoubleConfig, **_: Any) -> Transform:
    return lambda x, xp: 2 * x


def test_unpackaged_component_warns_once(tmp_path: Path) -> None:
    with transforms.scoped():
        transforms.register("tests.double", DoubleConfig)(double)
        with structlog.testing.capture_logs() as logs:
            export_artifact(ExportSource(Actor(spec({"type": "tests.double"}))), tmp_path)
    events = [log["event"] for log in logs]
    assert events.count("component has no distribution to require") == 1
