from typing import Any

import array_api_compat.numpy as numpy_xp
import numpy as np
import pytest
import torch

from policies.io.outputs import Distribution, distributions
from policies.learning import DistributionHead, PartDistribution, PartsDistribution

CASES: dict[str, tuple[dict[str, Any], tuple[int, ...]]] = {
    "normal": ({"type": "normal", "initial_std": 0.8}, (2, 3)),
    "normal_scalar": ({"type": "normal"}, ()),
    "normal_state_std": ({"type": "normal_state_std", "initial_std": 0.7}, (3,)),
    "gsde": ({"type": "gsde", "latent": 4, "initial_std": 0.6}, (2, 3)),
    "gsde_scalar": ({"type": "gsde", "latent": 3, "full_std": False}, ()),
    "categorical": ({"type": "categorical", "choices": 4}, ()),
    "bernoulli": ({"type": "bernoulli"}, (2, 2)),
    "multi_categorical": ({"type": "multi_categorical", "choices": [2, 3, 4]}, (3,)),
}
CONTINUOUS = {"normal", "normal_scalar", "normal_state_std", "gsde", "gsde_scalar"}
BATCH = 5


def definition(case: str) -> Distribution:
    config, shape = CASES[case]
    return distributions.build(config, shape=shape)


def part(case: str, seed: int = 0) -> PartDistribution:
    built = definition(case)
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(BATCH, built.raw_size, generator=generator) * 0.5
    distribution: PartsDistribution = DistributionHead({case: built})({case: raw.requires_grad_()})
    return distribution.parts[case]


@pytest.mark.parametrize("case", CASES)
def test_sample_log_prob_and_entropy_shapes(case: str) -> None:
    distribution = part(case)
    shape = CASES[case][1]
    sample = distribution.sample()
    assert sample.shape == (BATCH, *shape)
    assert sample.dtype == getattr(torch, distribution.definition.dtype)
    for value in (distribution.log_prob(sample), distribution.entropy()):
        assert value.shape == (BATCH,)
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("case", CASES)
def test_entropy_matches_samples(case: str) -> None:
    torch.manual_seed(0)
    distribution = part(case)
    samples = distribution.torch.sample((20000,))
    estimate = -distribution.log_prob(samples).mean(0)
    torch.testing.assert_close(estimate, distribution.entropy(), atol=0.05, rtol=0.02)


@pytest.mark.parametrize("case", CASES)
def test_kl_matches_samples(case: str) -> None:
    torch.manual_seed(0)
    p, q = part(case, seed=1), part(case, seed=2)
    torch.testing.assert_close(p.kl(p), torch.zeros(BATCH), atol=1e-5, rtol=0)
    samples = p.torch.sample((20000,))
    estimate = (p.log_prob(samples) - q.log_prob(samples)).mean(0)
    kl = p.kl(q)
    assert (kl >= 0).all()
    torch.testing.assert_close(estimate, kl, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("case", CASES)
def test_numpy_mode_matches_torch_and_is_most_likely(case: str) -> None:
    distribution = part(case)
    expected = distribution.mode()
    actual = distribution.definition.mode(distribution.raw.detach().numpy(), numpy_xp)
    assert actual.dtype == np.dtype(distribution.definition.dtype)
    np.testing.assert_array_equal(actual, expected.detach().numpy())
    samples = distribution.torch.sample((200,))
    most = distribution.log_prob(expected)
    assert (most >= distribution.log_prob(samples).max(0).values - 1e-5).all()


@pytest.mark.parametrize("case", CASES)
def test_imitation_and_smoothness(case: str) -> None:
    distribution = part(case)
    mode, sample = distribution.mode(), distribution.sample()
    best, other = distribution.imitation(mode), distribution.imitation(sample)
    assert best.shape == (BATCH,)
    assert (best <= other + 1e-5).all()
    best.sum().backward()  # type: ignore[no-untyped-call]  # torch Tensor.backward is unannotated
    assert distribution.raw.grad is not None
    penalty = distribution.smoothness(sample)
    if case in CONTINUOUS:
        assert penalty is not None and penalty.shape == (BATCH,)
        torch.testing.assert_close(distribution.smoothness(mode), torch.zeros(BATCH))
    else:
        assert penalty is None


def test_learned_std_is_a_parameter() -> None:
    parameters = dict(DistributionHead({"x": definition("normal")}).named_parameters())
    assert list(parameters) == ["learned.x.log_std"]
    torch.testing.assert_close(parameters["learned.x.log_std"].exp(), torch.full((2, 3), 0.8))


def test_multi_categorical_never_picks_padding() -> None:
    built = definition("multi_categorical")
    raw = np.zeros((1, built.raw_size), np.float32)
    raw[0, :] = np.tile([0, 0, 0, 9], 3)
    np.testing.assert_array_equal(built.mode(raw, numpy_xp), [[0, 0, 3]])


def test_multi_categorical_needs_matching_shape() -> None:
    with pytest.raises(ValueError, match="choice counts"):
        distributions.build({"type": "multi_categorical", "choices": [2, 3]}, shape=(3,))
