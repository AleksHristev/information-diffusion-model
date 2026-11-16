"""Checks of the cascade simulators and the random-graph helper."""

import numpy as np

from information_diffusion.simulation import cascades


def test_digraph_has_the_right_links_and_no_links_into_seeds():
    rng = np.random.default_rng(0)
    seeds = np.arange(8)
    for links in (60, 120):
        adjacency = cascades.random_digraph(30, links, seeds, rng)
        assert adjacency.sum() == links and not adjacency.diagonal().any() and not adjacency[:, seeds].any()


def test_every_non_seed_is_reachable_from_the_seeds():
    rng = np.random.default_rng(1)
    seeds = np.arange(8)
    adjacency = cascades.random_digraph(30, 40, seeds, rng)
    reached = set(seeds.tolist())
    frontier = list(reached)
    while frontier:
        node = frontier.pop()
        for child in np.flatnonzero(adjacency[node]):
            if child not in reached:
                reached.add(int(child))
                frontier.append(int(child))
    assert reached == set(range(30))


def test_independent_cascade_one_step_delay_and_no_reactivation():
    rng = np.random.default_rng(2)
    p = np.zeros((3, 3))
    p[0, 1] = p[1, 2] = 1.0
    activation = cascades.simulate_independent_cascade(p, np.array([0]), 5, 10, rng)
    assert (activation == np.array([[0, 1, 2]] * 5)).all()


def test_independent_cascade_respects_probability_zero_and_seeds_per_cascade():
    rng = np.random.default_rng(3)
    p = np.zeros((3, 3))
    p[0, 1] = 0.0
    seeds = np.array([[1, 0, 0], [0, 0, 1]], dtype=bool)
    activation = cascades.simulate_independent_cascade(p, seeds, 2, 10, rng)
    assert activation[0].tolist() == [0, -1, -1] and activation[1].tolist() == [-1, -1, 0]


def test_linear_threshold_window_limits_exposure():
    rng = np.random.default_rng(4)
    weights = np.zeros((2, 2))
    weights[0, 1] = 30.0
    activation = cascades.simulate_linear_threshold(weights, np.array([0.0, 28.0]), np.array([0]), 200, 12, 2, rng)
    active_steps = activation[activation[:, 1] >= 0, 1]
    assert active_steps.size > 0 and active_steps.max() <= 2
    assert (activation[:, 0] == 0).all()


def test_linear_threshold_active_nodes_are_not_at_risk():
    rng = np.random.default_rng(5)
    activation = cascades.simulate_linear_threshold(np.zeros((2, 2)), np.array([0.0, 0.0]), np.array([0]), 50, 30, 3, rng)
    assert (activation[:, 1] >= 1).all()
