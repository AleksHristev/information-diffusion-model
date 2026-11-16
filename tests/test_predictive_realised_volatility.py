"""Checks of the realised-volatility regression: known coefficients and the floor on zero variance."""

import numpy as np
import pytest

from information_diffusion.predictive import realised_volatility_regression as rv


def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    alpha = np.array([-4.0, -3.0, -5.0])
    beta = np.array([0.35, 0.2, 0.15])
    series = rv.simulate(alpha, beta, 0.6, (1, 5, 20), 33, 180, rng)
    fit = rv.fit(series, (1, 5, 20))
    assert fit.beta == pytest.approx(beta, abs=0.03)
    assert fit.alpha == pytest.approx(alpha, abs=0.15)


def test_zero_realised_variance_is_floored_not_dropped():
    series = np.ones((1, 1, 8))
    series[0, 0, 3] = 0.0
    y, _, components = rv.lagged_design(series, lags=(1, 2))
    assert np.isfinite(y).all() and np.isfinite(components).all()
