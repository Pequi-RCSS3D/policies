from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from policies.core.fingerprint import fingerprint
from policies.core.parts import PartError, Parts, PartSpec

type Make = Callable[..., Any]

GYRO = PartSpec(name="gyro", shape=(3,))
COMMAND = PartSpec(name="command", shape=(3,), low=-1.0, high=1.0)
CONTACT = PartSpec(name="contact", shape=(2,), dtype="bool", privileged=True)
PARTS = Parts.of(GYRO, COMMAND, CONTACT)


def test_valid_batch(make: Make) -> None:
    values = {
        "gyro": make((8, 3)),
        "command": make((8, 3), fill=0.5),
        "contact": make((8, 2), dtype="bool"),
    }
    assert PARTS.check(values) == (8,)


def test_single_sample(make: Make) -> None:
    assert GYRO.check(make((3,))) == ()


def test_wrong_shape(make: Make) -> None:
    with pytest.raises(PartError, match=r"part 'gyro': expected shape \(\.\.\., 3\), got \(8, 2\)"):
        GYRO.check(make((8, 2)))


def test_wrong_dtype(make: Make) -> None:
    with pytest.raises(PartError, match="expected float32"):
        GYRO.check(make((3,), dtype="float64"))


def test_nan(make: Make) -> None:
    with pytest.raises(PartError, match="NaN"):
        GYRO.check(make((3,), fill=float("nan")))


def test_bounds(make: Make) -> None:
    with pytest.raises(PartError, match=r"above 1\.0"):
        COMMAND.check(make((3,), fill=2.0))


def test_mismatched_batch(make: Make) -> None:
    values = {"gyro": make((8, 3)), "command": make((4, 3)), "contact": make((8, 2), dtype="bool")}
    with pytest.raises(PartError, match="batch shape"):
        PARTS.check(values)


def test_missing_and_extra(make: Make) -> None:
    with pytest.raises(PartError, match=r"missing parts \['contact'\], unexpected parts \['vel'\]"):
        PARTS.check({"gyro": make((3,)), "command": make((3,)), "vel": make((3,))})


def test_lookup() -> None:
    assert PARTS["command"].size == 3
    assert "gyro" in PARTS
    with pytest.raises(KeyError, match="no part 'vel'"):
        PARTS["vel"]


def test_duplicate_names() -> None:
    with pytest.raises(ValidationError, match="duplicate part 'gyro'"):
        Parts.of(GYRO, GYRO)


def test_bad_specs() -> None:
    with pytest.raises(ValidationError, match=r"low 1\.0 is above high"):
        PartSpec(name="x", low=1.0, high=0.0)
    with pytest.raises(ValidationError, match="must be positive"):
        PartSpec(name="x", shape=(0,))
    with pytest.raises(ValidationError, match="name"):
        PartSpec(name="Bad Name")


def test_fingerprint_tracks_changes() -> None:
    assert fingerprint(PARTS) != fingerprint(Parts.of(COMMAND, GYRO, CONTACT))
    assert fingerprint(PARTS) != fingerprint(
        Parts.of(GYRO, COMMAND, CONTACT.model_copy(update={"privileged": False}))
    )


def test_array_api_integer_dtypes(make: Make) -> None:
    image = PartSpec(name="image", shape=(2, 2), dtype="uint8")
    assert image.check(make((4, 2, 2), dtype="uint8")) == (4,)
    with pytest.raises(PartError, match="expected uint8"):
        image.check(make((2, 2), dtype="int8"))
