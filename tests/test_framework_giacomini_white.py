"""Reference tests for the Giacomini-White test of equal predictive ability and the directional decision rule."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import giacomini_white as gw


def test_hand_computed_case():
    result = gw.giacomini_white([1.0, 2.0, 3.0, 4.0, 5.0], null="student")
    t = 3 / np.sqrt(2.8 / 5)
    assert result.statistic == pytest.approx(t)
    assert result.long_run_variance == pytest.approx(2.8)
    assert result.p_value == pytest.approx(2 * stats.t.sf(t, 4))
    assert result.degrees_of_freedom == 4 and result.mean_differential == pytest.approx(3.0)


def test_one_sided_p_values():
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    greater = gw.giacomini_white(d, alternative="greater").p_value
    less = gw.giacomini_white(d, alternative="less").p_value
    two_sided = gw.giacomini_white(d).p_value
    assert greater + less == pytest.approx(1.0)
    assert 2 * greater == pytest.approx(two_sided)


def test_directional_rule():
    assert gw.supports_stated_sign(0.01, 0.3, 1)
    assert not gw.supports_stated_sign(0.01, -0.3, 1)
    assert not gw.supports_stated_sign(0.20, 0.3, 1)
    assert gw.supports_stated_sign(0.01, -0.3, -1)


def test_statistic_is_scale_invariant():
    rng = np.random.default_rng(702)
    d = rng.standard_normal(21)
    assert gw.giacomini_white(d * 1e-8).statistic == pytest.approx(gw.giacomini_white(d).statistic)


def test_large_sample_size_is_near_nominal():
    rng = np.random.default_rng(703)
    d = rng.standard_normal((40_000, 800))
    size = np.mean(gw.p_value_from_statistic(gw.statistic(d), 800) <= 0.05)
    assert size == pytest.approx(0.05, abs=0.006)


def test_vectorised_statistic_matches_the_single_series_result():
    rng = np.random.default_rng(704)
    d = rng.standard_normal((3, 21))
    assert gw.statistic(d)[1] == pytest.approx(gw.giacomini_white(d[1]).statistic)


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError):
        gw.giacomini_white([1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        gw.giacomini_white([1.0, 2.0])
    with pytest.raises(ValueError):
        gw.giacomini_white([1.0, 2.0, 3.0], alternative="sideways")
