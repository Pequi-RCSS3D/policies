from collections.abc import Callable
from typing import Any, ClassVar, cast

import array_api_compat.numpy as numpy_xp
import numpy as np
import pytest
from array_api_compat import array_namespace

from policies.io.bookkeeping import (
    Compose,
    Tracker,
    TrackerContext,
    UnknownSourceError,
    build_bookkeeping,
    select,
    source_dtype,
    source_shape,
    zeros,
)

type Make = Callable[..., Any]

TRACKERS = (
    {"type": "history", "source": "position", "length": 3},
    {"type": "last", "source": "action"},
    {"type": "timer", "name": "skill", "dt": 0.02},
)
SHAPES = {"position": (2,), "action": (1,)}


def trajectory(steps: int, batch: int) -> list[np.ndarray]:
    return [np.full((batch, 2), float(t), np.float32) for t in range(steps)]


def run(
    bookkeeping: Tracker, xp: object, positions: list[np.ndarray], to: Callable[[np.ndarray], Any]
) -> list[dict[str, np.ndarray]]:
    results = []
    for t, position in enumerate(positions):
        slots = bookkeeping.observe({"position": to(position)})
        results.append(
            {k: np.asarray(v.cpu() if hasattr(v, "cpu") else v) for k, v in slots.items()}
        )
        bookkeeping.commit({"action": to(np.full((position.shape[0], 1), t, np.float32))})
    return results


def test_history_last_timer_on_numpy() -> None:
    bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(2,), xp=numpy_xp)
    assert bookkeeping.slots == {
        "history.position": (3, 2),
        "last.action": (1,),
        "timer.skill": (1,),
    }
    results = run(bookkeeping, numpy_xp, trajectory(4, 2), lambda x: x)
    np.testing.assert_array_equal(results[0]["history.position"][0, :, 0], [0, 0, 0])
    np.testing.assert_array_equal(results[3]["history.position"][0, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(results[3]["last.action"][0], [2])
    np.testing.assert_allclose(results[3]["timer.skill"][0], [0.06])


def test_reset_mask() -> None:
    bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(2,), xp=numpy_xp)
    run(bookkeeping, numpy_xp, trajectory(3, 2), lambda x: x)
    bookkeeping.reset(np.array([True, False]))
    slots = bookkeeping.observe({"position": np.full((2, 2), 9.0, np.float32)})
    np.testing.assert_array_equal(slots["history.position"][0, :, 0], [9, 9, 9])
    np.testing.assert_array_equal(slots["history.position"][1, :, 0], [1, 2, 9])
    assert float(slots["timer.skill"][0, 0]) == 0.0
    assert float(slots["last.action"][0, 0]) == 0.0


def test_peek_does_not_advance() -> None:
    bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(1,), xp=numpy_xp)
    bookkeeping.observe({"position": np.zeros((1, 2), np.float32)})
    bookkeeping.observe({"position": np.ones((1, 2), np.float32)}, update=False)
    slots = bookkeeping.observe({"position": np.full((1, 2), 2.0, np.float32)})
    np.testing.assert_array_equal(slots["history.position"][0, :, 0], [0, 0, 2])


def test_single_sample_batch() -> None:
    bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(), xp=numpy_xp)
    slots = bookkeeping.observe({"position": np.ones(2, np.float32)})
    assert slots["history.position"].shape == (3, 2)


def test_numpy_and_torch_agree() -> None:
    torch = pytest.importorskip("torch")
    import array_api_compat.torch as torch_xp

    positions = trajectory(5, 3)
    numpy_bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(3,), xp=numpy_xp)
    torch_bookkeeping = build_bookkeeping(TRACKERS, shapes=SHAPES, batch=(3,), xp=torch_xp)
    expected = run(numpy_bookkeeping, numpy_xp, positions, lambda x: x)
    actual = run(torch_bookkeeping, torch_xp, positions, torch.from_numpy)
    for left, right in zip(expected, actual, strict=True):
        for name in left:
            np.testing.assert_allclose(left[name], right[name], rtol=1e-6)


def test_slot_clash() -> None:
    with pytest.raises(ValueError, match="defined twice"):
        build_bookkeeping((*TRACKERS, TRACKERS[0]), shapes=SHAPES, batch=(1,), xp=numpy_xp)


def test_unknown_source() -> None:
    with pytest.raises(UnknownSourceError, match="'velocity' is not known") as error:
        build_bookkeeping(
            [{"type": "history", "source": "velocity", "length": 2}],
            shapes=SHAPES,
            batch=(1,),
            xp=numpy_xp,
        )
    assert error.value.source == "velocity"


def test_trackers_declare_their_sources() -> None:
    configs = (*TRACKERS, {"type": "last", "source": "position"})
    bookkeeping = build_bookkeeping(configs, shapes=SHAPES, batch=(1,), xp=numpy_xp)
    assert [part.sources for part in bookkeeping.parts] == [
        ("position",),
        ("action",),
        (),
        ("position",),
    ]
    assert bookkeeping.sources == ("position", "action")


class Undeclared:
    """A tracker without `sources`."""

    slots: ClassVar[dict[str, tuple[int, ...]]] = {}


def test_trackers_must_declare_sources() -> None:
    with pytest.raises(TypeError, match="Undeclared must declare `sources`"):
        Compose([cast(Tracker, Undeclared())])


def test_sources_keep_their_dtypes(make: Make) -> None:
    bookkeeping = build_bookkeeping(
        [{"type": "history", "source": "mode", "length": 2}, {"type": "last", "source": "skill"}],
        shapes={"mode": (1,), "skill": ()},
        dtypes={"mode": "int32", "skill": "int64"},
        batch=(2,),
        xp=array_namespace(make((1,))),
    )
    slots = bookkeeping.observe({"mode": make((2, 1), dtype="int32", fill=3)})
    assert str(slots["history.mode"].dtype).endswith("int32")
    assert str(slots["last.skill"].dtype).endswith("int64")
    bookkeeping.commit({"skill": make((2,), dtype="int64", fill=2)})
    assert int(bookkeeping.observe({"mode": make((2, 1), dtype="int32")})["last.skill"][0]) == 2


def test_helpers_build_a_plugin_tracker() -> None:
    context = TrackerContext({"flag": (2,)}, (3,), numpy_xp, {"flag": "bool"})
    value = zeros(context, source_shape(context, "flag"), source_dtype(context, "flag"))
    assert value.shape == (3, 2) and value.dtype == np.bool_
    chosen = select(context, np.array([True, False, True]), ~value, value)
    np.testing.assert_array_equal(chosen[:, 0], [True, False, True])
