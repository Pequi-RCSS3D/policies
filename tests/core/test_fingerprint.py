import subprocess
import sys

import pytest

from policies.core.config import Config
from policies.core.fingerprint import fingerprint, short


class Part(Config):
    name: str
    size: int = 3


def test_key_order_does_not_matter() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_any_change_changes_it() -> None:
    assert fingerprint({"a": [1, 2]}) != fingerprint({"a": [2, 1]})
    assert fingerprint(Part(name="gyro")) != fingerprint(Part(name="gyro", size=4))


def test_config_and_its_dump_agree() -> None:
    assert fingerprint(Part(name="gyro")) == fingerprint({"name": "gyro", "size": 3})


def test_stable_across_processes() -> None:
    code = "from policies.core.fingerprint import fingerprint; print(fingerprint({'x': 1.5}))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == fingerprint({"x": 1.5})


def test_nan_is_rejected() -> None:
    with pytest.raises(ValueError):
        fingerprint({"x": float("nan")})


def test_short() -> None:
    assert short(fingerprint("x")) == fingerprint("x")[:12]


def test_models_inside_containers() -> None:
    assert fingerprint({"parts": [Part(name="gyro")]}) == fingerprint(
        {"parts": [{"name": "gyro", "size": 3}]}
    )
