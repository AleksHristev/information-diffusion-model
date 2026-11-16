"""The Poisson regression: recovery, score, covariance, offset, deviance and input checks."""

import numpy as np
import pytest
from scipy.special import gammaln

from information_diffusion.predictive import poisson_regression as pr


def test_it_recovers_known_coefficients_on_a_large_sample():
    rng = np.random.default_rng(1)
    n = 200_000
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.uniform(0, 1, n)])
    beta = np.array([0.5, 0.3, -0.7])
    y = rng.poisson(np.exp(X @ beta))
    result = pr.fit(y, X)
    assert result.converged
    np.testing.assert_allclose(result.coefficients, beta, atol=0.01)


def test_the_score_is_zero_and_the_covariance_is_the_inverse_information():
    rng = np.random.default_rng(2)
    X = np.column_stack([np.ones(500), rng.standard_normal(500)])
    y = rng.poisson(np.exp(0.2 + 0.4 * X[:, 1]))
    result = pr.fit(y, X)
    mu = np.exp(X @ result.coefficients)
    np.testing.assert_allclose(X.T @ (y - mu), 0.0, atol=1e-6)
    np.testing.assert_allclose(result.covariance, np.linalg.inv(X.T @ (X * mu[:, None])), rtol=1e-6)


def test_an_offset_shifts_the_linear_predictor():
    rng = np.random.default_rng(3)
    X = np.ones((50_000, 1))
    offset = rng.normal(0.0, 1.0, 50_000)
    y = rng.poisson(np.exp(0.3 + offset))
    assert pr.fit(y, X, offset).coefficients[0] == pytest.approx(0.3, abs=0.02)


def test_the_deviance_and_log_likelihood_match_their_definitions():
    y = np.array([0.0, 1.0, 4.0])
    mu = np.array([0.5, 1.5, 3.0])
    assert pr.deviance(y, mu) == pytest.approx(2 * (-(0 - 0.5) + (np.log(1 / 1.5) - (1 - 1.5)) + (4 * np.log(4 / 3) - (4 - 3))))
    assert pr.log_likelihood(y, mu) == pytest.approx(float(np.sum(y * np.log(mu) - mu - gammaln(y + 1))))


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        pr.fit(np.array([1.0, -1.0]), np.ones((2, 1)))
    with pytest.raises(ValueError):
        pr.fit(np.array([1.0, 2.0]), np.ones((3, 1)))
