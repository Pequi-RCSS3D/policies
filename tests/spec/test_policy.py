import json
import subprocess
import sys
from typing import Any

from policies.core.config import Config
from policies.spec import PolicySpec
from policies.spec.model import blocks


def make_spec(**fields: Any) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "walk",
            "inputs": {"parts": [{"name": "gyro", "source": "gyro", "shape": [3]}]},
            "outputs": {"parts": [{"name": "action", "kind": {"type": "continuous", "size": 2}}]},
            **fields,
        }
    )


BLOCK = {
    "type": "mixture",
    "experts": 4,
    "expert": {"type": "gru", "hidden_size": 128, "layers": 1, "after": [], "activation": "elu"},
    "gate": [],
    "top_k": None,
    "activation": "elu",
}
SPELLED = {
    "bookkeeping": [{"type": "timer", "name": "episode", "dt": 1.0}],
    "model": {
        "encoder": {"type": "flatten"},
        "block": BLOCK,
        "head": {"type": "linear", "initial_scale": 0.01},
    },
}
SHORT = {
    "bookkeeping": [{"type": "timer", "name": "episode"}],
    "model": {"block": {"type": "mixture", "expert": {"type": "gru"}}},
}


def test_spelled_out_defaults_do_not_change_the_fingerprint() -> None:
    assert make_spec(**SHORT).fingerprint() == make_spec(**SPELLED).fingerprint()
    assert make_spec().fingerprint() == make_spec(model={"block": {"type": "mlp"}}).fingerprint()


def test_settings_change_the_fingerprint() -> None:
    wide = {"model": {"block": {"type": "mlp", "hidden": [512]}}}
    assert make_spec().fingerprint() != make_spec(**wide).fingerprint()


def test_canonical_spec_validates_to_itself() -> None:
    spec = make_spec(**SHORT)
    assert PolicySpec.model_validate(spec.model_dump(mode="json")) == spec


class Narrow(Config):
    width: int = 1


class Wide(Config):
    width: int = 2


def test_changed_default_changes_the_fingerprint() -> None:
    with blocks.scoped():
        blocks.register_lazy("tests.block", Narrow, "tests:unused")
        before = make_spec(model={"block": {"type": "tests.block"}}).fingerprint()
        blocks.register_lazy("tests.block", Wide, "tests:unused", replace=True)
        after = make_spec(model={"block": {"type": "tests.block"}}).fingerprint()
    assert before != after
    assert "tests.block" not in blocks


LIGHT = """
import json, sys
sys.modules["torch"] = None  # makes importing torch fail
from policies.spec import PolicySpec
spec = PolicySpec.model_validate(json.loads(sys.argv[1]))
print(json.dumps([spec.fingerprint(), spec.model["block"]]))
"""


def test_canonical_without_torch() -> None:
    data = make_spec(**SHORT).model_dump(mode="json") | SHORT
    result = subprocess.run(
        [sys.executable, "-c", LIGHT, json.dumps(data)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    digest, block = json.loads(result.stdout)
    assert digest == make_spec(**SPELLED).fingerprint()
    assert block == BLOCK
