"""Reference tests for the Model Confidence Set."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import model_confidence_set as mcs


def _losses(rng, n, m, gap, best=0):
    losses = rng.standard_normal((n, m)) + gap
    losses[:, best] -= gap
    return losses


def test_one_clearly_best_model_is_alone_in_the_set():
    rng = np.random.default_rng(801)
    result = mcs.model_confidence_set(_losses(rng, 300, 4, 1.0, best=2), block_length=1, rng=rng, replications=999)
    assert result.included.tolist() == [2]
    assert result.p_values[2] == 1.0
    assert np.all(np.delete(result.p_values, 2) <= 0.05)


def test_p_values_are_monotone_in_elimination_order_and_the_survivor_has_one():
    rng = np.random.default_rng(802)
    result = mcs.model_confidence_set(rng.standard_normal((80, 6)), block_length=1, rng=rng, replications=499)
    ordered = result.p_values[result.elimination_order]
    assert np.all(np.diff(ordered) >= 0) and ordered[-1] == 1.0
    assert sorted(result.elimination_order.tolist()) == list(range(6))


@pytest.mark.parametrize("statistic", ["range", "max"])
def test_all_equal_models_are_retained_at_the_nominal_rate(statistic):
    rng = np.random.default_rng(803)
    runs = 120
    retained = [mcs.model_confidence_set(rng.standard_normal((100, 4)), block_length=1, rng=rng, replications=299, statistic=statistic).included.size == 4 for _ in range(runs)]
    assert np.mean(retained) >= 0.85


def test_two_models_range_and_max_are_identical():
    data = np.random.default_rng(804).standard_normal((120, 2)) + np.array([0.0, 0.3])
    a = mcs.model_confidence_set(data, block_length=1, rng=np.random.default_rng(1), replications=999, statistic="range")
    b = mcs.model_confidence_set(data, block_length=1, rng=np.random.default_rng(1), replications=999, statistic="max")
    assert a.p_values == pytest.approx(b.p_values)


def test_two_models_p_value_approaches_the_normal_tail():
    rng = np.random.default_rng(805)
    n = 2000
    data = rng.standard_normal((n, 2)) + np.array([0.0, 0.02])
    d = data[:, 1] - data[:, 0]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    result = mcs.model_confidence_set(data, block_length=1, rng=rng, replications=20_000)
    worse = int(d.mean() < 0)
    assert result.p_values[1 - worse] == pytest.approx(2 * stats.norm.sf(abs(t)), abs=0.02)


def test_identical_loss_columns_do_not_crash():
    data = np.column_stack([np.ones(30), np.ones(30), np.arange(30.0)])
    result = mcs.model_confidence_set(data, block_length=1, rng=np.random.default_rng(806), replications=199)
    assert result.p_values.max() == 1.0


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(807)
    with pytest.raises(ValueError):
        mcs.model_confidence_set(np.ones((5, 1)), block_length=1, rng=rng)
    with pytest.raises(ValueError):
        mcs.model_confidence_set(np.ones((5, 2)), block_length=1, rng=rng, statistic="minimum")
