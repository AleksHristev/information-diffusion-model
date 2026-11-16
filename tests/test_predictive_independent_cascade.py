"""Checks of the Independent Cascade likelihood, gradient and fit."""

import numpy as np
import pytest

from information_diffusion.predictive import independent_cascade as ic


def chain_data():
    activation = np.array([[0, 1, 2], [0, 1, -1], [0, -1, -1]])
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    return activation, adjacency


def test_likelihood_matches_hand_computation_on_chain():
    activation, adjacency = chain_data()
    a, b = 0.3, 0.6
    probabilities = np.zeros((3, 3))
    probabilities[0, 1], probabilities[1, 2] = a, b
    expected = 2 * np.log(a) + np.log(1 - a) + np.log(b) + np.log(1 - b)
    assert ic.log_likelihood(activation, adjacency, probabilities, 10) == pytest.approx(expected)


def test_likelihood_of_common_child_with_two_parents():
    activation = np.array([[0, 0, 1], [0, 0, -1]])
    adjacency = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    p = np.zeros((3, 3))
    p[0, 2], p[1, 2] = 0.2, 0.5
    expected = np.log(1 - 0.8 * 0.5) + np.log(0.8) + np.log(0.5)
    assert ic.log_likelihood(activation, adjacency, p, 10) == pytest.approx(expected)


def test_late_parent_counts_as_failure_for_the_earlier_one():
    activation = np.array([[0, 1, 3, 2]])
    adjacency = np.zeros((4, 4), dtype=bool)
    adjacency[0, 1] = adjacency[1, 2] = adjacency[3, 2] = True
    p = np.zeros((4, 4))
    p[0, 1], p[1, 2], p[3, 2] = 0.4, 0.3, 0.7
    expected = np.log(0.4) + np.log(1 - 0.3) + np.log(0.7)
    assert ic.log_likelihood(activation, adjacency, p, 10) == pytest.approx(expected)


def test_attempts_at_the_horizon_are_censored():
    activation = np.array([[0, 2, -1]])
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
    p = np.zeros((3, 3))
    p[0, 1], p[1, 2] = 0.5, 0.9
    assert ic.log_likelihood(activation, adjacency, p, 3) == pytest.approx(np.log(1 - 0.5))


def test_score_matches_finite_differences():
    rng = np.random.default_rng(3)
    activation = np.array([[0, 1, 1, 2, -1], [0, -1, 1, 2, 3], [0, 2, 1, -1, 3], [0, 1, 2, 3, -1]])
    adjacency = np.zeros((5, 5), dtype=bool)
    for parent, child in [(0, 1), (0, 2), (2, 1), (1, 3), (2, 3), (3, 4), (1, 4)]:
        adjacency[parent, child] = True
    p = np.zeros((5, 5))
    p[adjacency] = rng.uniform(0.1, 0.9, adjacency.sum())
    gradient = ic.score(activation, adjacency, p, 6)
    for parent, child in np.argwhere(adjacency):
        h = 1e-6
        up, down = p.copy(), p.copy()
        up[parent, child] += h
        down[parent, child] -= h
        numeric = (ic.log_likelihood(activation, adjacency, up, 6) - ic.log_likelihood(activation, adjacency, down, 6)) / (2 * h)
        assert gradient[parent, child] == pytest.approx(numeric, rel=1e-5, abs=1e-5)


def test_em_gives_the_closed_form_for_a_single_parent():
    n = 20
    k = 7
    activation = np.full((n, 2), -1)
    activation[:, 0] = 0
    activation[:k, 1] = 1
    adjacency = np.array([[0, 1], [0, 0]], dtype=bool)
    fit = ic.fit(activation, adjacency, 10)
    assert fit.probability[0] == pytest.approx(k / n, abs=1e-7)
    assert fit.identified[0]
    assert fit.lower[0] < k / n < fit.upper[0]
    level = 0.5 * 3.841458820694124
    def drop(x):
        return k * np.log(x) + (n - k) * np.log(1 - x) - (k * np.log(k / n) + (n - k) * np.log(1 - k / n)) + level
    assert drop(fit.lower[0]) == pytest.approx(0.0, abs=2e-3)
    assert drop(fit.upper[0]) == pytest.approx(0.0, abs=2e-3)


def test_estimate_at_the_boundary_has_zero_lower_limit():
    n = 30
    activation = np.full((n, 2), -1)
    activation[:, 0] = 0
    adjacency = np.array([[0, 1], [0, 0]], dtype=bool)
    fit = ic.fit(activation, adjacency, 10)
    assert fit.probability[0] == 0.0 and fit.lower[0] == 0.0 and 0.0 < fit.upper[0] < 0.2


def test_em_score_is_zero_at_an_interior_optimum():
    rng = np.random.default_rng(5)
    n_cascades = 800
    activation = np.full((n_cascades, 3), -1)
    activation[:, :2] = 0
    success_a = rng.random(n_cascades) < 0.3
    success_b = rng.random(n_cascades) < 0.5
    a_seeded = rng.random(n_cascades) < 0.6
    b_seeded = rng.random(n_cascades) < 0.6
    activation[:, 0] = np.where(a_seeded, 0, -1)
    activation[:, 1] = np.where(b_seeded, 0, -1)
    both = (a_seeded & success_a) | (b_seeded & success_b)
    activation[both, 2] = 1
    adjacency = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    fit = ic.fit(activation, adjacency, 10)
    assert fit.identified.all()
    p = np.zeros((3, 3))
    p[fit.edges[:, 0], fit.edges[:, 1]] = fit.probability
    gradient = ic.score(activation, adjacency, p, 10)
    assert np.abs(gradient[adjacency]).max() < 1e-3
    assert fit.probability == pytest.approx([0.3, 0.5], abs=0.1)


def test_parents_always_newly_active_together_are_not_identified():
    activation = np.array([[0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1]])
    adjacency = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    fit = ic.fit(activation, adjacency, 10)
    assert not fit.identified.any()
    assert np.isnan(fit.probability).all()


def test_links_outside_adjacency_are_ignored():
    activation = np.array([[0, -1, 1]])
    adjacency = np.zeros((3, 3), dtype=bool)
    adjacency[1, 2] = True
    assert ic.log_likelihood(activation, adjacency, np.zeros((3, 3)), 5) == 0.0
    fit = ic.fit(activation, adjacency, 5)
    assert fit.edges.shape == (1, 2)
    assert fit.trials[0] == 0 and np.isnan(fit.probability[0])
