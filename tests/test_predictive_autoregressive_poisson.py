"""Checks of the autoregressive Poisson model on the spline baseline: recovery, the rate and its lag alignment, simulation and refused shapes."""

import numpy as np
import pytest

from information_diffusion.epidemic.spline_baseline import SplineBaseline
from information_diffusion.predictive import autoregressive_poisson as ap
from information_diffusion.predictive.spline_poisson import step_quadrature

PHI = np.array([0.35, 0.2, 0.1])
GAMMA = (1.0 - PHI.sum()) * np.array([0.1, 0.2, 0.2, 0.3, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3])


@pytest.fixture(scope="module")
def baseline() -> SplineBaseline:
    return SplineBaseline(5400.0, 1800.0, n_events=500)


def test_fit_recovers_known_coefficients(baseline):
    alpha = np.array([-2.3, -2.6, -2.9])
    fit = ap.fit(ap.simulate(alpha, GAMMA, PHI, baseline, 33, np.random.default_rng(1)), baseline, n_lags=3)
    assert fit.fit.converged
    assert fit.phi == pytest.approx(PHI, abs=0.03)
    assert fit.alpha == pytest.approx(alpha, abs=0.2)
    assert np.all(np.abs(fit.gamma - GAMMA) < 4.0 * fit.gamma_se)


def test_rate_uses_the_step_integral_and_log_one_plus_lagged_counts(baseline):
    quadrature = step_quadrature(baseline)
    fit = ap.AutoregressivePoissonFit(alpha=np.array([0.5]), phi=np.array([0.3, 0.1]), gamma=GAMMA, alpha_se=np.zeros(1), phi_se=np.zeros(2), gamma_se=np.zeros(GAMMA.size), quadrature=quadrature, fit=None)
    expected = quadrature.integrals(GAMMA)[10] * np.exp(0.5 + 0.3 * np.log1p(4.0) + 0.1 * np.log1p(2.0))
    assert fit.rate(np.array([0]), np.array([10]), np.array([[4.0, 2.0]]))[0] == pytest.approx(expected, rel=1e-12)


def test_simulated_counts_have_the_requested_shape_and_are_counts(baseline):
    counts = ap.simulate(np.array([-2.5, -3.0]), GAMMA, PHI, baseline, 5, np.random.default_rng(2))
    assert counts.shape == (2, 5, 180)
    assert np.all(counts >= 0) and np.all(counts == np.floor(counts))


def test_a_lag_coefficient_of_zero_gives_the_spline_poisson_mean(baseline):
    alpha = np.array([-1.0])
    counts = ap.simulate(alpha, GAMMA, np.zeros(3), baseline, 3000, np.random.default_rng(3))
    np.testing.assert_allclose(counts.mean(axis=(0, 1)), np.exp(alpha[0]) * step_quadrature(baseline).integrals(GAMMA), rtol=0.05)


def test_a_count_array_that_does_not_fit_the_window_is_refused(baseline):
    with pytest.raises(ValueError):
        ap.fit(np.zeros((1, 2, 10)), baseline)
    with pytest.raises(ValueError):
        ap.lagged_design(np.zeros((1, 2, 4)), 4)
