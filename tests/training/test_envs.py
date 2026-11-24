from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import Tensor

from policies.core.config import Config
from policies.core.registry import UnknownComponentError
from policies.training.envs import (
    FINAL_OBSERVATION,
    ActionDelay,
    DroppedCycles,
    Environment,
    Observation,
    Quantize,
    Rounding,
    Step,
    Wrapper,
    wrap,
    wrappers,
)
from policies.training.envs.wrappers import Hold, Noise


class Record:
    """Observes the output it was given."""

    num_envs = 2

    def __init__(self) -> None:
        self.applied: list[float] = []

    def reset(self) -> Observation:
        return {"value": torch.full((2, 1), 0.129)}

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        self.applied.append(float(outputs["value"][0, 0]))
        done = torch.zeros(2, dtype=torch.bool)
        observation = {"value": outputs["value"].clone()}
        return observation, torch.zeros(2), done, {FINAL_OBSERVATION: observation}


def output(value: float) -> dict[str, Tensor]:
    return {"value": torch.full((2, 1), value)}


def test_action_delay_applies_late_and_clears_on_reset() -> None:
    env = Record()
    delayed = ActionDelay(env, cycles=1)
    delayed.reset()
    for value in (1.0, 2.0, 3.0):
        delayed.step(output(value))
    delayed.reset()
    delayed.step(output(4.0))
    assert env.applied == [1.0, 1.0, 2.0, 4.0]


def test_dropped_cycles_repeat_and_clear_on_reset() -> None:
    env = Record()
    dropped = DroppedCycles(env, rate=1.0)
    dropped.reset()
    dropped.step(output(1.0))
    dropped.step(output(2.0))
    dropped.reset()
    dropped.step(output(3.0))
    assert env.applied == [1.0, 1.0, 3.0]


VALUES = torch.tensor([0.129, 0.125, -0.129, -0.121])


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [
        (Rounding.NEAREST, [0.12, 0.12, -0.12, -0.12]),
        (Rounding.TRUNCATE, [0.12, 0.12, -0.12, -0.12]),
        (Rounding.FLOOR, [0.12, 0.12, -0.16, -0.16]),
        (Rounding.CEIL, [0.16, 0.16, -0.12, -0.12]),
    ],
)
def test_quantize_rounding(rounding: Rounding, expected: list[float]) -> None:
    quantize = Quantize(Record(), {"value": 0.04}, rounding)
    rounded = quantize.observation({"value": VALUES, "other": VALUES})
    torch.testing.assert_close(rounded["value"], torch.tensor(expected))
    assert rounded["other"] is VALUES


def test_quantize_nearest_differs_from_truncate() -> None:
    observation = {"value": torch.tensor([0.19, -0.19])}
    nearest = Quantize(Record(), {"value": 0.1}).observation(observation)
    truncated = Quantize(Record(), {"value": 0.1}, Rounding.TRUNCATE).observation(observation)
    torch.testing.assert_close(nearest["value"], torch.tensor([0.2, -0.2]))
    torch.testing.assert_close(truncated["value"], torch.tensor([0.1, -0.1]))


def test_quantize_truncates_each_part_to_its_decimals() -> None:
    values = torch.tensor([1.23456, -1.23456, 0.0049, -0.0049], dtype=torch.float64)
    steps = {"coarse": 0.01, "fine": 0.001}
    quantize = Quantize(Record(), steps, Rounding.TRUNCATE)
    rounded = quantize.observation({"coarse": values, "fine": values})
    torch.testing.assert_close(rounded["coarse"], torch.trunc(values * 100) / 100)
    torch.testing.assert_close(rounded["fine"], torch.trunc(values * 1000) / 1000)


def test_quantize_truncation_keeps_values_on_the_grid() -> None:
    grid = torch.tensor([1.0, -1.0, 0.5, 0.0, 0.25, 0.999])
    truncated = Quantize(Record(), {"value": 0.001}, Rounding.TRUNCATE).observation({"value": grid})
    assert torch.equal(truncated["value"], grid)


def test_quantize_applies_to_resets_steps_and_final_observations() -> None:
    quantize = Quantize(Record(), {"value": 0.01})
    torch.testing.assert_close(quantize.reset()["value"], torch.full((2, 1), 0.13))
    observation, _, _, info = quantize.step(output(0.456))
    torch.testing.assert_close(observation["value"], torch.full((2, 1), 0.46))
    torch.testing.assert_close(info[FINAL_OBSERVATION]["value"], torch.full((2, 1), 0.46))


class Offset(Wrapper):
    """Adds `amount` to every output."""

    def __init__(self, env: Environment, amount: float) -> None:
        super().__init__(env)
        self.amount = amount

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        return self.env.step({k: v + self.amount for k, v in outputs.items()})


class OffsetConfig(Config):
    amount: float


@wrappers.register("tests.offset", OffsetConfig)
def offset(config: OffsetConfig, env: Environment, **_: Any) -> Offset:
    return Offset(env, config.amount)


def test_wrap_applies_configs_in_order() -> None:
    env = Record()
    wrapped = wrap(
        env,
        [
            {"type": "quantize", "steps": {"value": 0.5}, "rounding": "floor"},
            {"type": "tests.offset", "amount": 1.0},
            {"type": "action_delay", "cycles": 1},
        ],
        seed=0,
    )
    assert isinstance(wrapped, ActionDelay)
    assert isinstance(wrapped.env, Offset) and isinstance(wrapped.env.env, Quantize)
    assert wrapped.env.env.env is env
    wrapped.reset()
    observation, *_ = wrapped.step(output(0.2))
    assert env.applied == [pytest.approx(1.2)]
    torch.testing.assert_close(observation["value"], torch.full((2, 1), 1.0))


def test_builtin_wrappers_are_registered() -> None:
    assert {"action_delay", "dropped_cycles", "hold", "noise", "quantize"} <= set(wrappers.names())


def test_dropped_cycles_seed_defaults_to_the_run_seed() -> None:
    built = wrap(Record(), [{"type": "dropped_cycles", "rate": 0.5}], seed=7)
    assert isinstance(built, DroppedCycles) and built.seed == 7
    built = wrap(Record(), [{"type": "dropped_cycles", "rate": 0.5, "seed": 1}], seed=7)
    assert isinstance(built, DroppedCycles) and built.seed == 1


def test_unknown_wrapper_is_an_error() -> None:
    with pytest.raises(UnknownComponentError, match="quantize"):
        wrap(Record(), [{"type": "quantise", "steps": {}}])


class Count:
    """Observes the step number. Environments in `ends` finish at the steps they map to."""

    def __init__(self, num_envs: int = 2, ends: Mapping[int, list[int]] | None = None) -> None:
        self.num_envs = num_envs
        self.ends = ends or {}
        self.t = 0

    def _observation(self) -> Observation:
        value = torch.full((self.num_envs, 1), float(self.t))
        return {"value": value, "other": value.clone()}

    def reset(self) -> Observation:
        self.t = 0
        return self._observation()

    def step(self, outputs: Mapping[str, Tensor]) -> Step:
        self.t += 1
        final = self._observation()
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        dones[self.ends.get(self.t, [])] = True
        observation = self._observation()
        observation["value"][dones] = -1.0
        return observation, torch.zeros(self.num_envs), dones, {FINAL_OBSERVATION: final}


def run(env: Environment, steps: int, part: str = "value") -> list[list[float]]:
    """The part's first column per environment, from the reset and each step."""
    values = [env.reset()[part][:, 0].tolist()]
    for _ in range(steps):
        observation, *_ = env.step({})
        values.append(observation[part][:, 0].tolist())
    return values


@pytest.mark.parametrize(("phase", "expected"), [(0, [0, 0, 2, 2, 4, 4]), (1, [0, 1, 1, 3, 3, 5])])
def test_hold_refreshes_every_n_steps(phase: int, expected: list[float]) -> None:
    held = Hold(Count(), ["value"], every=2, phase=phase)
    assert run(held, 5) == [[value] * 2 for value in expected]
    assert run(held, 2, "other") == [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]


def test_hold_refreshes_on_new_episodes_and_holds_final_observations() -> None:
    held = Hold(Count(ends={3: [0]}), ["value"], every=2)
    held.reset()
    infos = []
    observations = []
    for _ in range(4):
        observation, _, _, info = held.step({})
        observations.append(observation["value"][:, 0].tolist())
        infos.append(info[FINAL_OBSERVATION]["value"][:, 0].tolist())
    # Environment 0 starts over at step 3 and refreshes then. Its final observation is held.
    assert observations == [[0, 0], [2, 2], [-1, 2], [-1, 4]]
    assert infos == [[0, 0], [2, 2], [2, 2], [-1, 4]]


def test_hold_random_phase_differs_per_environment_and_follows_the_seed() -> None:
    def refreshes(seed: int) -> Tensor:
        values = torch.tensor(
            run(Hold(Count(num_envs=64), ["value"], 4, random_phase=True, seed=seed), 8)
        )
        return values[1:] != values[:-1]

    first = refreshes(0)
    assert torch.equal(first, refreshes(0))
    assert not torch.equal(first, refreshes(1))
    assert torch.all(first.sum(0) == 2)
    assert len({tuple(column.tolist()) for column in first.T}) == 4


def test_hold_and_noise_configs_default_to_the_run_seed() -> None:
    built = wrap(Count(), [{"type": "hold", "parts": ["value"], "every": 3}], seed=7)
    assert isinstance(built, Hold) and (built.every, built.seed) == (3, 7)
    built = wrap(Count(), [{"type": "noise", "scales": {"value": 0.1}, "seed": 1}], seed=7)
    assert isinstance(built, Noise) and built.seed == 1


def test_hold_rejects_a_zero_rate() -> None:
    with pytest.raises(ValueError, match="every"):
        Hold(Count(), ["value"], every=0)
    with pytest.raises(ValueError, match="every"):
        wrap(Count(), [{"type": "hold", "parts": ["value"], "every": 0}])


def test_noise_scales_each_part_and_follows_the_seed() -> None:
    env = Count(num_envs=4096)
    noisy = Noise(env, {"value": 0.5, "missing": 1.0}, seed=3)
    observation = noisy.reset()
    assert set(observation) == {"value", "other"}
    assert torch.equal(observation["other"], torch.zeros(4096, 1))
    assert abs(float(observation["value"].mean())) < 0.05
    assert abs(float(observation["value"].std()) - 0.5) < 0.05
    again = Noise(Count(num_envs=4096), {"value": 0.5}, seed=3).reset()
    assert torch.equal(observation["value"], again["value"])


def test_noise_draws_once_per_step_for_the_final_observation() -> None:
    noisy = Noise(Count(), {"value": 1.0})
    noisy.reset()
    observation, _, _, info = noisy.step({})
    torch.testing.assert_close(info[FINAL_OBSERVATION]["value"], observation["value"])
    assert not torch.equal(observation["value"], torch.ones(2, 1))
