import math
from collections.abc import Mapping

import pytest
import torch
from torch import Tensor

from policies.io.outputs import Noise, distributions
from policies.io.outputs.distributions import GSDE
from policies.learning import DistributionHead, Learner, LearnerConfig, RolloutConfig
from policies.spec import PolicySpec
from tests.envs import Reach
from tests.learning.test_learner import train

LATENT, SIZE = 5, 3


def build(**config: object) -> GSDE:
    built = distributions.build({"type": "gsde", "latent": LATENT, **config}, shape=(SIZE,))
    assert isinstance(built, GSDE)
    return built


def learned(definition: GSDE, seed: int = 0) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    log_std = torch.tensor(definition.learned["log_std"])
    return {"log_std": log_std + 0.3 * torch.randn(log_std.shape, generator=generator)}


@pytest.mark.parametrize("full_std", [True, False])
def test_log_prob_is_gaussian_with_feature_weighted_variance(full_std: bool) -> None:
    definition = build(full_std=full_std, initial_std=0.5)
    parameters = learned(definition)
    raw = torch.randn(7, definition.raw_size)
    action = torch.randn(7, SIZE)
    mean, features = raw[:, :SIZE], raw[:, SIZE:]
    std = parameters["log_std"].exp().reshape(LATENT, -1).expand(LATENT, SIZE)
    variance = (features[:, :, None] ** 2 * std[None] ** 2).sum(1) + 1e-6
    expected = (
        -((action - mean) ** 2) / (2 * variance) - 0.5 * torch.log(2 * math.pi * variance)
    ).sum(-1)
    distribution = definition.torch_distribution(raw, parameters)
    torch.testing.assert_close(distribution.log_prob(action), expected)
    torch.testing.assert_close(distribution.mean, mean)


def test_noise_matches_the_distribution_std() -> None:
    torch.manual_seed(0)
    definition = build()
    parameters = learned(definition)
    raw = torch.randn(1, definition.raw_size).expand(20000, -1)
    actions = definition.noise(20000, "cpu").sample(raw, parameters)
    distribution = definition.torch_distribution(raw[:1], parameters)
    torch.testing.assert_close(actions.mean(0), distribution.mean[0], atol=0.05, rtol=0)
    torch.testing.assert_close(actions.std(0), distribution.stddev[0], atol=0, rtol=0.03)


def steps(noise: Noise, raw: Tensor, parameters: Mapping[str, Tensor], count: int) -> list[Tensor]:
    return [noise.sample(raw, parameters) for _ in range(count)]


def test_noise_holds_between_redraws() -> None:
    definition = build(sample_every=3)
    parameters, raw = learned(definition), torch.randn(4, definition.raw_size)
    first, second, third, fourth = steps(definition.noise(4, "cpu"), raw, parameters, 4)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, third)
    assert (fourth - first).abs().amin(-1).gt(0).all()


def test_reset_redraws_only_done_environments() -> None:
    definition = build(sample_every=10)
    parameters, raw = learned(definition), torch.randn(2, definition.raw_size)
    noise = definition.noise(2, "cpu")
    (before,) = steps(noise, raw, parameters, 1)
    noise.reset(torch.tensor([True, False]))
    (after,) = steps(noise, raw, parameters, 1)
    assert (after[0] != before[0]).all()
    torch.testing.assert_close(after[1], before[1])


def test_noise_follows_the_current_std() -> None:
    definition = build(sample_every=10)
    parameters, raw = learned(definition), torch.randn(2, definition.raw_size)
    noise = definition.noise(2, "cpu")
    (before,) = steps(noise, raw, parameters, 1)
    doubled = {"log_std": parameters["log_std"] + math.log(2)}
    (after,) = steps(noise, raw, doubled, 1)
    mean = raw[:, :SIZE]
    torch.testing.assert_close(after - mean, 2 * (before - mean))


def test_features_are_detached_unless_learned() -> None:
    for learn, expected in ((False, False), (True, True)):
        definition = build(learn_features=learn)
        raw = torch.randn(3, definition.raw_size, requires_grad=True)
        log_prob = definition.torch_distribution(raw, learned(definition)).log_prob(
            torch.zeros(3, SIZE)
        )
        (grad,) = torch.autograd.grad(log_prob.sum(), raw)
        assert bool(grad[:, SIZE:].abs().sum() > 0) is expected


def test_head_builds_noise_only_for_noisy_parts() -> None:
    built = {"g": build(), "n": distributions.build({"type": "normal"}, shape=(2,))}
    noise = DistributionHead(built).noise(4, "cpu")
    assert list(noise) == ["g"]


def spec(initial_std: float, sample_every: int = 4, normalize: bool = True) -> PolicySpec:
    return PolicySpec.model_validate(
        {
            "name": "reach",
            "inputs": {
                "parts": [
                    {"name": "state", "source": "state", "shape": [2], "normalize": normalize},
                    {"name": "target", "source": "target", "shape": [1]},
                ]
            },
            "outputs": {
                "parts": [
                    {
                        "name": "force",
                        "kind": {"type": "continuous", "size": 1},
                        "distribution": {
                            "type": "gsde",
                            "latent": 16,
                            "sample_every": sample_every,
                            "initial_std": initial_std,
                        },
                    }
                ]
            },
            "model": {"block": {"type": "mlp", "hidden": [64, 64]}},
        }
    )


def test_learner_holds_noise_and_acts_on_the_mean_when_deployed() -> None:
    torch.manual_seed(0)
    learner = Learner(spec(1.0, sample_every=1000, normalize=False), num_envs=4)
    observation = Reach(4).reset()
    with torch.no_grad():
        parts = learner.actor_inputs(observation, learner._slots(observation, update=False))
        distribution, _ = learner.actor(parts, learner.actor_state)
    mean = distribution.mode()["force"]
    first, second = learner.act(observation)["force"], learner.act(observation)["force"]
    torch.testing.assert_close(first, second)
    assert (first != mean).all()
    learner.observe(observation, torch.zeros(4), torch.tensor([True, False, False, False]))
    third = learner.act(observation)["force"]
    assert third[0] != first[0]
    torch.testing.assert_close(third[1:], first[1:])
    learner.training = False
    torch.testing.assert_close(learner.act(observation)["force"], mean)


def test_ppo_learns_reach_with_gsde() -> None:
    # Features start near 0.02 in norm, so this gives an action std near 0.4.
    config = LearnerConfig(optimizer={"type": "adam", "lr": 3e-3}, rollout=RolloutConfig(length=40))
    learner = Learner(spec(20.0), config, num_envs=64)
    rewards = train(learner, Reach(64), 60)
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.3
