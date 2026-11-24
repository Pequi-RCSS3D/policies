import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import array_api_compat.numpy as numpy_xp
import numpy as np
import pytest
from array_api_compat import array_namespace
from numpy.polynomial import legendre

from policies.io.bookkeeping import Compose, build_bookkeeping
from policies.io.bookkeeping.legendre_system import legendre_system
from policies.io.bookkeeping.zero_order_hold import zero_order_hold

type Make = Callable[..., Any]

DT = 0.02
WINDOW = 0.5
SLOT = "legendre.memory"


def tracker(
    xp: Any = numpy_xp,
    batch: tuple[int, ...] = (2,),
    shape: tuple[int, ...] = (3,),
    **options: Any,
) -> Compose:
    config = {"type": "legendre", "source": "x", "window": WINDOW, "order": 6, "dt": DT}
    config |= {"name": "memory", **options}
    return build_bookkeeping([config], shapes={"x": shape}, batch=batch, xp=xp)


def smooth(t: np.ndarray) -> np.ndarray:
    """Two smooth features at time `t`, stacked last."""
    first = np.sin(2 * np.pi * t) + 0.5 * np.cos(2 * np.pi * 0.4 * t)
    second = np.cos(2 * np.pi * 1.5 * t + 1.0)
    return np.stack([first, second], axis=-1)


def reconstruction_error(order: int) -> tuple[float, float]:
    """Max and RMS error of the decoded window against the signal it saw."""
    bookkeeping = tracker(batch=(1,), shape=(2,), order=order)
    times = np.arange(150) * DT
    for value in smooth(times).astype(np.float32):
        coefficients = bookkeeping.observe({"x": value[None]})[SLOT][0]
    ago = np.linspace(0, 1, 201)
    decoded = legendre.legvander(2 * ago - 1, order - 1) @ coefficients.T
    # Held samples lag a smooth signal by half a step.
    truth = smooth(times[-1] - ago * WINDOW + DT / 2)
    error = decoded - truth
    return float(np.abs(error).max()), float(np.sqrt(np.mean(error**2)))


def test_constant_input_converges() -> None:
    bookkeeping = tracker(order=8)
    x = np.array([[1.0, -2.0, 0.5], [3.0, 0.0, -0.25]], np.float32)
    for _ in range(int(4 * WINDOW / DT)):
        memory = bookkeeping.observe({"x": x})[SLOT]
    expected = np.zeros((2, 3, 8), np.float32)
    expected[..., 0] = x
    np.testing.assert_allclose(memory, expected, atol=1e-5)


@pytest.mark.parametrize("order", [6, 7, 8])
def test_decoding_recovers_the_window(order: int) -> None:
    largest, root_mean_square = reconstruction_error(order)
    assert largest < 0.05
    assert root_mean_square < 0.01


def test_zero_reset_forgets_the_window() -> None:
    bookkeeping = tracker()
    x = np.ones((2, 3), np.float32)
    for _ in range(10):
        bookkeeping.observe({"x": x})
    peeked = bookkeeping.observe({"x": x}, update=False)[SLOT]
    bookkeeping.reset(np.array([True, False]))
    memory = bookkeeping.observe({"x": x})[SLOT]
    np.testing.assert_allclose(memory[0], tracker().observe({"x": x})[SLOT][0])
    np.testing.assert_allclose(memory[1], peeked[1])


def test_hold_reset_fills_the_window() -> None:
    bookkeeping = tracker(reset="hold")
    for _ in range(10):
        bookkeeping.observe({"x": np.ones((2, 3), np.float32)})
    bookkeeping.reset()
    x = np.array([[2.0, -1.0, 0.0]] * 2, np.float32)
    expected = np.zeros((2, 3, 6), np.float32)
    expected[..., 0] = x
    np.testing.assert_allclose(bookkeeping.observe({"x": x})[SLOT], expected, atol=1e-5)


def test_outputs_advance_on_commit() -> None:
    bookkeeping = tracker(reset="hold")
    assert not bookkeeping.observe({})[SLOT].any()
    x = np.full((2, 3), 2.0, np.float32)
    bookkeeping.commit({"x": x})
    np.testing.assert_allclose(bookkeeping.observe({})[SLOT][..., 0], x, atol=1e-5)
    bookkeeping.reset(np.array([True, False]))
    memory = bookkeeping.observe({})[SLOT]
    assert not memory[0].any()
    np.testing.assert_allclose(memory[1, :, 0], x[1], atol=1e-5)


def test_numpy_and_torch_agree() -> None:
    torch = pytest.importorskip("torch")
    import array_api_compat.torch as torch_xp

    generator = np.random.default_rng(0)
    for reset in ("zero", "hold"):
        numpy_bookkeeping = tracker(reset=reset)
        torch_bookkeeping = tracker(torch_xp, reset=reset)
        for t in range(40):
            if t == 15:
                numpy_bookkeeping.reset(np.array([False, True]))
                torch_bookkeeping.reset(torch.tensor([False, True]))
            x = generator.normal(size=(2, 3)).astype(np.float32)
            expected = numpy_bookkeeping.observe({"x": x})[SLOT]
            actual = torch_bookkeeping.observe({"x": torch.from_numpy(x)})[SLOT]
            np.testing.assert_allclose(expected, actual.numpy(), atol=1e-5)


def test_dtypes(make: Make) -> None:
    xp = array_namespace(make((1,)))
    config = {"type": "legendre", "source": "x", "window": 1.0, "order": 4, "dt": DT, "name": "x"}
    wide = build_bookkeeping(
        [config], shapes={"x": (3,)}, dtypes={"x": "float64"}, batch=(2,), xp=xp
    )
    memory = wide.observe({"x": make((2, 3), "float64", 1.0)})["legendre.x"]
    assert str(memory.dtype).endswith("float64")
    narrow = build_bookkeeping([config], shapes={"x": (3,)}, batch=(2,), xp=xp)
    memory = narrow.observe({"x": make((2, 3), "int32", 1.0)})["legendre.x"]
    assert str(memory.dtype).endswith("float32")


def test_sources_and_shapes() -> None:
    bookkeeping = tracker(batch=(4, 2), shape=(23,), order=8, name="joints")
    assert bookkeeping.sources == ("x",)
    assert bookkeeping.slots == {"legendre.joints": (23, 8)}
    memory = bookkeeping.observe({"x": np.zeros((4, 2, 23), np.float32)})["legendre.joints"]
    assert memory.shape == (4, 2, 23, 8)


def test_zero_order_hold_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    a, b = legendre_system(WINDOW, 8)
    transition, intake = zero_order_hold(a, b, DT)
    expected = torch.linalg.matrix_exp(torch.from_numpy(a * DT)).numpy()
    np.testing.assert_allclose(transition, expected, rtol=1e-12, atol=1e-12)
    # Held input integrates through the system: B_d = A^-1 (A_d - I) B.
    np.testing.assert_allclose(intake, np.linalg.solve(a, (transition - np.eye(8)) @ b), 1e-10)


def test_step_time_is_negligible() -> None:
    bookkeeping = tracker(batch=(), shape=(23,), order=8)
    x = np.random.default_rng(0).normal(size=23).astype(np.float32)
    for _ in range(100):
        bookkeeping.observe({"x": x})
    steps = 2000
    start = time.perf_counter()
    for _ in range(steps):
        bookkeeping.observe({"x": x})
    per_step = (time.perf_counter() - start) / steps
    assert per_step < 1e-3, f"{per_step * 1e6:.1f} us per step"


def test_policy_trains_exports_and_steps(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from policies.deploy.export import export_artifact
    from policies.deploy.inference import Policy
    from policies.learning import Learner, LearnerConfig, RolloutConfig
    from policies.spec import PolicySpec

    spec = PolicySpec.model_validate(
        {
            "name": "remember",
            "inputs": {
                "parts": [{"name": "memory", "source": "legendre.joints", "shape": [23, 8]}]
            },
            "outputs": {"parts": [{"name": "force", "kind": {"type": "continuous", "size": 2}}]},
            "bookkeeping": [
                {
                    "type": "legendre",
                    "source": "joints",
                    "window": WINDOW,
                    "order": 8,
                    "dt": DT,
                    "name": "joints",
                }
            ],
            "model": {"block": {"type": "mlp", "hidden": [16]}},
        }
    )
    torch.manual_seed(0)
    learner = Learner(spec, LearnerConfig(rollout=RolloutConfig(length=4)), num_envs=2)
    before = [parameter.clone() for parameter in learner.actor.parameters()]
    for _ in range(3):
        while not learner.ready:
            joints = torch.randn(2, 23)
            outputs = learner.act({"joints": joints})
            reward = -outputs["force"].square().sum(-1)
            learner.observe({"joints": joints}, reward, torch.tensor([False, True]))
        assert all(np.isfinite(value) for value in learner.update().values())
    after = list(learner.actor.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    export_artifact(learner, tmp_path)
    policy = Policy.load(tmp_path)
    assert policy.observation_keys == ("joints",)
    assert policy.observation_shapes == {"joints": (23,)}
    for _ in range(3):
        outputs = policy.step({"joints": np.ones(23, np.float32)})
    assert outputs["force"].shape == (2,)
