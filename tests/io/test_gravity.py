import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import array_api_compat.numpy as numpy_xp
import numpy as np
import pytest
from array_api_compat import array_namespace

from policies.io.bookkeeping import Compose, build_bookkeeping

type Make = Callable[..., Any]

SHAPES = {"gyro": (3,), "accel": (3,)}
DT = 0.02
UP = np.array([0.0, 0.0, 9.81], np.float32)


def tracker(xp: Any = numpy_xp, batch: tuple[int, ...] = (2,), **options: Any) -> Compose:
    config = {"type": "gravity", "gyroscope": "gyro", "accelerometer": "accel", "dt": DT}
    return build_bookkeeping([{**config, **options}], shapes=SHAPES, batch=batch, xp=xp)


def tilted(angle: float) -> np.ndarray:
    """Body-frame gravity of a body rolled by `angle` about x."""
    return np.array([0.0, -math.sin(angle), -math.cos(angle)], np.float32)


def test_static_tilt_converges() -> None:
    bookkeeping = tracker(gain=2.0)
    still = np.zeros((2, 3), np.float32)
    first = bookkeeping.observe({"gyro": still, "accel": np.stack([UP, UP])})
    np.testing.assert_allclose(first["gravity.estimate"], [[0, 0, -1]] * 2, atol=1e-6)
    truth = np.stack([tilted(0.4), tilted(-1.2)])
    for _ in range(400):
        estimate = bookkeeping.observe({"gyro": still, "accel": -9.81 * truth})["gravity.estimate"]
    np.testing.assert_allclose(estimate, truth, atol=1e-4)


@pytest.mark.parametrize(("unit", "scale"), [("rad/s", 1.0), ("deg/s", 180 / math.pi)])
def test_rotation_tracks_the_gyroscope(unit: str, scale: float) -> None:
    bookkeeping = tracker(gain=0.0, gyroscope_unit=unit, batch=(2, 1))
    rate = np.array([[[2.0, 0, 0]], [[-5.0, 0, 0]]], np.float32)
    accel = np.broadcast_to(UP, (2, 1, 3))
    for _ in range(50):
        estimate = bookkeeping.observe({"gyro": rate * scale, "accel": accel})["gravity.estimate"]
    elapsed = 49 * DT
    np.testing.assert_allclose(estimate[:, 0], [tilted(2 * elapsed), tilted(-5 * elapsed)], 1e-4)


def test_rotation_with_correction_stays_on_truth() -> None:
    bookkeeping = tracker(batch=(1,))
    for t in range(100):
        truth = tilted(3 * t * DT)[None]
        rate = np.array([[3.0, 0, 0]], np.float32)
        estimate = bookkeeping.observe({"gyro": rate, "accel": -9.81 * truth})["gravity.estimate"]
        np.testing.assert_allclose(estimate, truth, atol=1e-4)


def test_reset_restarts_from_the_accelerometer() -> None:
    bookkeeping = tracker(gain=0.0)
    turning = np.array([[1.0, 0, 0]] * 2, np.float32)
    for _ in range(20):
        bookkeeping.observe({"gyro": turning, "accel": np.stack([UP, UP])})
    accel = -9.81 * np.stack([tilted(0.3), tilted(0.3)])
    peeked = bookkeeping.observe({"gyro": turning, "accel": accel}, update=False)
    bookkeeping.reset(np.array([True, False]))
    estimate = bookkeeping.observe({"gyro": turning, "accel": accel})["gravity.estimate"]
    np.testing.assert_allclose(estimate[0], tilted(0.3), atol=1e-6)
    np.testing.assert_allclose(estimate[1], peeked["gravity.estimate"][1], atol=1e-6)
    bookkeeping.reset()
    still = np.zeros((2, 3), np.float32)
    estimate = bookkeeping.observe({"gyro": still, "accel": still})["gravity.estimate"]
    np.testing.assert_allclose(estimate, [[0, 0, -1]] * 2)


def test_numpy_and_torch_agree() -> None:
    torch = pytest.importorskip("torch")
    import array_api_compat.torch as torch_xp

    generator = np.random.default_rng(0)
    numpy_bookkeeping, torch_bookkeeping = tracker(), tracker(torch_xp)
    for t in range(30):
        if t == 10:
            numpy_bookkeeping.reset(np.array([False, True]))
            torch_bookkeeping.reset(torch.tensor([False, True]))
        gyro = generator.normal(size=(2, 3)).astype(np.float32)
        accel = (UP + generator.normal(size=(2, 3))).astype(np.float32)
        expected = numpy_bookkeeping.observe({"gyro": gyro, "accel": accel})
        actual = torch_bookkeeping.observe(
            {"gyro": torch.from_numpy(gyro), "accel": torch.from_numpy(accel)}
        )
        np.testing.assert_allclose(
            expected["gravity.estimate"], actual["gravity.estimate"].numpy(), atol=1e-5
        )


def test_dtypes(make: Make) -> None:
    xp = array_namespace(make((1,)))
    wide = build_bookkeeping(
        [{"type": "gravity", "gyroscope": "gyro", "accelerometer": "accel", "dt": DT}],
        shapes=SHAPES,
        dtypes={"accel": "float64"},
        batch=(2,),
        xp=xp,
    )
    slots = wide.observe({"gyro": make((2, 3)), "accel": make((2, 3), "float64", 1.0)})
    assert str(slots["gravity.estimate"].dtype).endswith("float64")
    slots = tracker(xp).observe({"gyro": make((2, 3), "int32"), "accel": make((2, 3), fill=1.0)})
    assert str(slots["gravity.estimate"].dtype).endswith("float32")


def test_sources_and_shapes() -> None:
    bookkeeping = tracker(name="imu")
    assert bookkeeping.sources == ("gyro", "accel")
    assert bookkeeping.slots == {"gravity.imu": (3,)}
    with pytest.raises(ValueError, match="'gyro' must have shape"):
        build_bookkeeping(
            [{"type": "gravity", "gyroscope": "gyro", "accelerometer": "accel", "dt": DT}],
            shapes={"gyro": (2,), "accel": (3,)},
            batch=(1,),
            xp=numpy_xp,
        )


def test_policy_exports_and_steps(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from policies.deploy.export import export_artifact
    from policies.deploy.inference import Policy
    from policies.learning import Learner
    from policies.spec import PolicySpec

    spec = PolicySpec.model_validate(
        {
            "name": "upright",
            "inputs": {"parts": [{"name": "gravity", "source": "gravity.estimate", "shape": [3]}]},
            "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 2}}]},
            "bookkeeping": [
                {"type": "gravity", "gyroscope": "gyro", "accelerometer": "accel", "dt": DT}
            ],
        }
    )
    learner = Learner(spec, num_envs=2)
    learner.act({"gyro": torch.zeros(2, 3), "accel": torch.from_numpy(np.stack([UP, UP]))})
    export_artifact(learner, tmp_path)
    policy = Policy.load(tmp_path)
    assert policy.observation_keys == ("gyro", "accel")
    assert policy.observation_shapes == {"gyro": (3,), "accel": (3,)}
    accel = -9.81 * tilted(0.5)
    for _ in range(3):
        outputs = policy.step({"gyro": np.zeros(3, np.float32), "accel": accel})
    assert outputs["force"].shape == (2,)
    peeked = policy.bookkeeping.observe({"gyro": np.zeros(3, np.float32), "accel": accel}, False)
    np.testing.assert_allclose(peeked["gravity.estimate"], tilted(0.5), atol=1e-6)
