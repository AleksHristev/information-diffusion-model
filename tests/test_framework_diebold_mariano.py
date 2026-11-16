"""Reference tests for the Diebold-Mariano test and its small-sample correction."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import diebold_mariano as dm


def test_hand_computed_statistic_horizon_one():
    result = dm.diebold_mariano([1.0, -1.0, 2.0, 0.0, 3.0])
    assert result.statistic == pytest.approx(1 / np.sqrt(0.4))
    assert result.p_value == pytest.approx(2 * stats.norm.sf(1 / np.sqrt(0.4)))


def test_hln_correction_at_horizon_one_is_the_one_sample_t_test():
    rng = np.random.default_rng(601)
    d = rng.standard_normal(21)
    corrected = dm.diebold_mariano(d, corrected=True)
    assert corrected.statistic == pytest.approx(stats.ttest_1samp(d, 0).statistic)
    assert corrected.p_value == pytest.approx(stats.ttest_1samp(d, 0).pvalue)


def test_hand_computed_with_correction_at_horizon_two():
    d = [1.0, 2.0, 3.0, 4.0]
    assert dm.diebold_mariano(d, horizon=2).statistic == pytest.approx(2.5 * np.sqrt(32 / 15))
    assert dm.diebold_mariano(d, horizon=2, corrected=True).statistic == pytest.approx(np.sqrt(5.0))


def test_p_values_one_sided_are_consistent():
    d = np.array([1.0, -1.0, 2.0, 0.0, 3.0])
    greater = dm.diebold_mariano(d, alternative="greater", corrected=True)
    less = dm.diebold_mariano(d, alternative="less", corrected=True)
    two_sided = dm.diebold_mariano(d, corrected=True)
    assert greater.p_value + less.p_value == pytest.approx(1.0)
    assert 2 * greater.p_value == pytest.approx(two_sided.p_value)


def test_undefined_variance_raises():
    with pytest.raises(ValueError):
        dm.diebold_mariano([1.0, -1.0, 2.0, 0.0, 3.0], horizon=2)
    with pytest.raises(ValueError):
        dm.diebold_mariano([2.0, 2.0, 2.0])


def test_size_large_sample_iid():
    rng = np.random.default_rng(602)
    d = rng.standard_normal((40_000, 800))
    p = dm.p_value_from_statistic(dm.dm_statistic(d), 800)
    assert np.mean(p <= 0.05) == pytest.approx(0.05, abs=0.006)


def test_vectorised_statistic_matches_the_single_series_result():
    rng = np.random.default_rng(603)
    d = rng.standard_normal((5, 30))
    assert dm.dm_statistic(d)[0] == pytest.approx(dm.diebold_mariano(d[0]).statistic)
