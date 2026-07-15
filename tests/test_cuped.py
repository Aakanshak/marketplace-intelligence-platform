import numpy as np
import pytest
from src.ab_test_cuped import cuped_adjust, required_sample_size, simulate_experiment


def test_cuped_preserves_mean_and_reduces_variance():
    rng = np.random.default_rng(7)
    x = rng.normal(size=10_000)
    y = .8 * x + rng.normal(size=10_000)
    adjusted, theta, reduction = cuped_adjust(y, x)
    assert adjusted.mean() == pytest.approx(y.mean())
    assert theta > 0
    assert reduction > .25


def test_cuped_rejects_bad_shapes():
    with pytest.raises(ValueError):
        cuped_adjust(np.ones(3), np.ones(4))


def test_power_and_simulation_are_reproducible():
    assert required_sample_size(.08, .05) > 1_000
    a, b = simulate_experiment(n=20_000), simulate_experiment(n=20_000)
    assert a["cuped"].p_value == b["cuped"].p_value
    assert a["variance_reduction"] > 0
