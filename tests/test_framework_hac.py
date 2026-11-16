"""Reference tests for Newey-West HAC variances."""

import numpy as np
import pytest
from scipy.signal import lfilter

from information_diffusion.statistical_framework import hac

SERIES = np.arange(1.0, 6.0)


def test_bartlett_weights_match_the_definition():
    assert hac.bartlett_weights(3) == pytest.approx([0.75, 0.5, 0.25])


def test_mean_variance_lag_zero_is_gamma_zero_over_n():
    assert hac.variance_of_mean(SERIES, 0) == pytest.approx(0.4)


def test_mean_variance_lag_one_hand_computed():
    assert hac.long_run_variance(SERIES, 1) == pytest.approx(2.8)
    assert hac.variance_of_mean(SERIES, 1) == pytest.approx(0.56)


def test_mean_variance_lag_two_hand_computed():
    assert hac.variance_of_mean(SERIES, 2) == pytest.approx((2 + 2 * (2 / 3) * 0.8 + 2 * (1 / 3) * (-0.2)) / 5)


def test_regression_covariance_hand_computed():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 4.0])
    fit = hac.ols_newey_west(y, X, 1)
    assert fit.coefficients == pytest.approx([-2 / 3, 3 / 2])
    assert fit.covariance == pytest.approx(np.array([[5 / 81, -1 / 36], [-1 / 36, 1 / 72]]))
    assert fit.residuals == pytest.approx([1 / 6, -1 / 3, 1 / 6])


def test_regression_small_sample_correction_is_n_over_n_minus_k():
    X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = np.array([1.0, 2.0, 4.0])
    assert hac.ols_newey_west(y, X, 1, small_sample=True).covariance == pytest.approx(3 * hac.ols_newey_west(y, X, 1).covariance)


def test_ar1_long_run_variance_by_large_sample_simulation():
    rng = np.random.default_rng(101)
    x = lfilter([1.0], [1.0, -0.5], rng.standard_normal(600_000))
    assert hac.long_run_variance(x, 80) == pytest.approx(4.0, rel=0.05)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        hac.long_run_variance(SERIES, 5)
    with pytest.raises(ValueError):
        hac.ols_newey_west(np.ones(3), np.ones((3, 2)), 0)
