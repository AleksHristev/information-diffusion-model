"""Sparse VAR and stability selection against OLS and small planted designs."""

import numpy as np
import pytest

from information_diffusion.influence import sparse_var as sv
from information_diffusion.influence import stability_selection as ss


def _simulate(rng, coefficient, steps=2000, events=1):
    nodes = coefficient.shape[0]
    paths = []
    for _ in range(events):
        y = np.zeros((steps + 50, nodes))
        for t in range(1, steps + 50):
            y[t] = coefficient @ y[t - 1] + rng.standard_normal(nodes)
        paths.append(y[50:])
    return paths


def test_coordinate_descent_with_zero_penalty_is_ols_on_standardised_data():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 5))
    y = X[:, 0] * 0.5 - X[:, 3] * 0.2 + rng.standard_normal(2000)
    gram = X.T @ X / 2000
    cross = (X.T @ y / 2000)[:, None]
    b = sv.coordinate_descent(gram, cross, np.zeros(1))
    np.testing.assert_allclose(b[:, 0], np.linalg.solve(gram, cross[:, 0]), atol=1e-3)


def test_coordinate_descent_is_the_soft_threshold_for_orthogonal_design():
    gram = np.eye(3)
    cross = np.array([[0.5], [0.05], [-0.4]])
    b = sv.coordinate_descent(gram, cross, np.array([0.1]))
    np.testing.assert_allclose(b[:, 0], [0.4, 0.0, -0.3])


def test_moments_do_not_cross_event_boundaries():
    first, second = np.array([[1.0], [2.0], [3.0]]), np.array([[10.0], [20.0], [30.0]])
    moments = sv.build_moments([first, second], 1)
    assert moments.xx[0, 0, 0] == pytest.approx(1.0)
    assert moments.xy[0, 0, 0] == pytest.approx(0.0)
    assert moments.n.tolist() == [2.0, 2.0]


def test_post_lasso_recovers_a_planted_sparse_var():
    rng = np.random.default_rng(2)
    a = 0.1 * np.eye(6)
    a[3, 0], a[5, 2] = 0.3, -0.25
    events = _simulate(rng, a, steps=1500, events=4)
    fit = sv.fit_ebic(sv.build_moments(events, 1), 1)
    matrix = fit.coefficient_matrices()[0]
    assert matrix[3, 0] == pytest.approx(0.3, abs=0.05) and matrix[5, 2] == pytest.approx(-0.25, abs=0.05)
    assert matrix[1, 4] == 0.0


def test_lag_selection_finds_order_one():
    rng = np.random.default_rng(3)
    a = 0.3 * np.eye(4)
    a[1, 0] = 0.3
    events = _simulate(rng, a, steps=4000, events=2)
    fit, totals = sv.select_lag_order(sv.build_moments(events, 3))
    assert fit.lag_order == 1 and totals.size == 3


def test_refit_weighted_with_unit_weights_matches_the_fit():
    rng = np.random.default_rng(4)
    a = 0.2 * np.eye(4)
    a[2, 1] = 0.4
    events = _simulate(rng, a, steps=800, events=6)
    moments = sv.build_moments(events, 1)
    fit = sv.fit_ebic(moments, 1)
    np.testing.assert_allclose(sv.refit_weighted(moments, fit, np.ones(6)), fit.coefficients, atol=1e-3)


def test_residual_covariance_matches_direct_residuals():
    rng = np.random.default_rng(5)
    a = 0.2 * np.eye(3)
    events = _simulate(rng, a, steps=1000, events=1)
    moments = sv.build_moments(events, 1)
    fit = sv.fit_ebic(moments, 1)
    demeaned = events[0] - events[0].mean(axis=0)
    residual = demeaned[1:] - demeaned[:-1] @ fit.coefficients
    np.testing.assert_allclose(sv.residual_covariance(moments, fit), residual.T @ residual / residual.shape[0], atol=1e-10)


def test_stability_selection_finds_the_planted_links():
    rng = np.random.default_rng(6)
    a = 0.1 * np.eye(8)
    a[4, 0], a[6, 1] = 0.3, 0.35
    events = _simulate(rng, a, steps=500, events=20)
    moments = sv.build_moments(events, 1)
    fit = sv.fit_ebic(moments, 1)
    frequency = ss.selection_frequencies(moments, fit, 60, rng)
    mask = ss.stable_links(frequency, 8)
    assert mask[4, 0] and mask[6, 1] and mask.sum() <= 4
