"""Checks of the Linear Threshold likelihood, gradient and identification."""

import numpy as np
import pytest
from scipy.special import expit

from information_diffusion.predictive import linear_threshold as lt
from information_diffusion.simulation import cascades


def test_likelihood_against_direct_summation_on_a_small_panel():
    activation = np.array([[0, 3], [0, -1]])
    adjacency = np.array([[0, 1], [0, 0]], dtype=bool)
    b, theta = 1.2, 2.0
    weights = np.zeros((2, 2))
    weights[0, 1] = b
    thresholds = np.array([0.0, theta])
    expected = 0.0
    for step, event in zip([1, 2, 3], [0, 0, 1]):
        eta = (b if step <= 2 else 0.0) - theta
        expected += event * eta - np.log1p(np.exp(eta))
    for step in [1, 2, 3, 4]:
        eta = (b if step <= 2 else 0.0) - theta
        expected += -np.log1p(np.exp(eta))
    assert lt.log_likelihood(activation, adjacency, weights, thresholds, 5, window=2) == pytest.approx(expected)


def test_score_matches_finite_differences():
    rng = np.random.default_rng(11)
    adjacency = np.zeros((6, 6), dtype=bool)
    for parent, child in [(0, 2), (1, 2), (2, 3), (1, 4), (3, 5), (4, 5)]:
        adjacency[parent, child] = True
    weights = np.zeros((6, 6))
    weights[adjacency] = rng.uniform(0.5, 2.5, adjacency.sum())
    thresholds = rng.uniform(1.0, 4.0, 6)
    activation = cascades.simulate_linear_threshold(weights, thresholds, np.array([0, 1]), 20, 30, 3, rng)
    gradient_w, gradient_t = lt.score(activation, adjacency, weights, thresholds, 30, window=3)
    for parent, child in np.argwhere(adjacency):
        h = 1e-5
        up, down = weights.copy(), weights.copy()
        up[parent, child] += h
        down[parent, child] -= h
        numeric = (lt.log_likelihood(activation, adjacency, up, thresholds, 30, window=3) - lt.log_likelihood(activation, adjacency, down, thresholds, 30, window=3)) / (2 * h)
        assert gradient_w[parent, child] == pytest.approx(numeric, rel=1e-5, abs=1e-5)
    for child in range(6):
        h = 1e-5
        up, down = thresholds.copy(), thresholds.copy()
        up[child] += h
        down[child] -= h
        numeric = (lt.log_likelihood(activation, adjacency, weights, up, 30, window=3) - lt.log_likelihood(activation, adjacency, weights, down, 30, window=3)) / (2 * h)
        assert gradient_t[child] == pytest.approx(numeric, rel=1e-5, abs=1e-5)


def test_classical_form_cannot_separate_seed_weight_from_threshold():
    rng = np.random.default_rng(0)
    weights = np.array([[0.0, 2.0], [0.0, 0.0]])
    activation = cascades.simulate_linear_threshold(weights, np.array([0.0, 3.0]), np.array([0]), 300, 40, None, rng)
    adjacency = np.array([[0, 1], [0, 0]], dtype=bool)
    result = lt.fit(activation, adjacency, 40, window=None, intervals=False)
    assert np.isnan(result.weight[0]) and np.isnan(result.threshold[1])


def test_finite_window_identifies_weight_and_threshold():
    rng = np.random.default_rng(0)
    weights = np.array([[0.0, 2.0], [0.0, 0.0]])
    activation = cascades.simulate_linear_threshold(weights, np.array([0.0, 3.0]), np.array([0]), 400, 40, 4, rng)
    adjacency = np.array([[0, 1], [0, 0]], dtype=bool)
    result = lt.fit(activation, adjacency, 40, window=4)
    assert result.weight[0] == pytest.approx(2.0, abs=0.5) and result.threshold[1] == pytest.approx(3.0, abs=0.5)
    assert result.weight_lower[0] < result.weight[0] < result.weight_upper[0]
    assert result.threshold_lower[1] < result.threshold[1] < result.threshold_upper[1]
    assert expit(result.weight[0] - result.threshold[1]) > expit(-result.threshold[1])


def test_two_seed_parents_of_one_child_are_not_identified_but_threshold_is():
    rng = np.random.default_rng(2)
    weights = np.zeros((3, 3))
    weights[0, 2], weights[1, 2] = 1.5, 2.5
    thresholds = np.array([0.0, 0.0, 3.0])
    activation = cascades.simulate_linear_threshold(weights, thresholds, np.array([0, 1]), 300, 40, 3, rng)
    adjacency = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    result = lt.fit(activation, adjacency, 40, window=3)
    assert not result.identified.any() and np.isnan(result.weight).all()
    assert result.threshold[2] == pytest.approx(3.0, abs=0.6)
