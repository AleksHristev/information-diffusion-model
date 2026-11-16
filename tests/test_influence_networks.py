"""Link FDR, community detection, sensor placement and the lead-lag simulator."""

import numpy as np
import pandas as pd
import pytest

from information_diffusion.influence import community_detection as cd
from information_diffusion.influence import link_fdr
from information_diffusion.influence import sensor_placement as sp
from information_diffusion.simulation import lead_lag
from information_diffusion.statistical_framework import benjamini_yekutieli as by


def test_link_fdr_is_the_shared_rule():
    p = np.array([1e-5, 0.0002, 0.01, 0.2, 0.9])
    np.testing.assert_array_equal(link_fdr.reject_links(p, 0.05), by.reject(p, 0.05))


def test_off_diagonal_gives_the_ordered_pairs():
    matrix = np.arange(16.0).reshape(4, 4)
    assert link_fdr.off_diagonal(matrix).tolist() == [1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0]


def test_louvain_splits_two_cliques():
    w = np.zeros((10, 10))
    w[:5, :5] = 1.0
    w[5:, 5:] = 1.0
    np.fill_diagonal(w, 0.0)
    w[0, 5] = w[5, 0] = 0.01
    labels = cd.louvain(w, np.random.default_rng(1))
    assert cd.adjusted_rand_index(labels, [0] * 5 + [1] * 5) == 1.0


def test_modularity_of_two_disconnected_cliques_is_one_half():
    w = np.zeros((8, 8))
    w[:4, :4] = 1.0
    w[4:, 4:] = 1.0
    np.fill_diagonal(w, 0.0)
    assert cd.modularity(w, [0, 0, 0, 0, 1, 1, 1, 1]) == pytest.approx(0.5)


def test_adjusted_rand_index_hand_cases():
    assert cd.adjusted_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert cd.adjusted_rand_index([0, 0, 1, 1], [0, 1, 0, 1]) < 0.0


def test_louvain_rejects_asymmetric_input():
    with pytest.raises(ValueError):
        cd.louvain(np.array([[0.0, 1.0], [0.0, 0.0]]), np.random.default_rng(0))


def test_greedy_matches_the_optimum_on_a_small_coverage_instance():
    c = np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 1.0], [0.0, 0.1, 0.9]])
    assert sorted(sp.greedy(c, 2)) == [0, 2]
    assert sp.objective(c, [0, 2]) == pytest.approx(sp.exhaustive_optimum(c, 2))


def test_greedy_is_within_the_submodular_bound_on_random_instances():
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = rng.random((40, 9)) * (rng.random((40, 9)) < 0.3)
        assert sp.objective(c, sp.greedy(c, 3)) >= (1 - 1 / np.e) * sp.exhaustive_optimum(c, 3) - 1e-12


def small_network(window=lead_lag.WINDOW_B):
    rows = []
    for follower, leaders in ((3, (0, 1, 2)), (4, (0, 1))):
        k = len(leaders)
        loading = np.sqrt(lead_lag.FOLLOWER_LINK_VARIANCE / (k + k * (k - 1) * lead_lag.COMMON_SHARE))
        for position, leader in enumerate(leaders):
            rows.append({"leader": leader, "follower": follower, "lag_seconds": lead_lag.LAG_SECONDS[position], "loading": loading, "thin_follower": follower == 4})
    return lead_lag.Network(("SPY", "QQQ", "IWM", "XLK", "XLU"), np.array([2000.0, 1500.0, 1200.0, 800.0, 300.0]), np.array([False, False, False, False, True]), (0, 1, 2),
                            pd.DataFrame(rows), lead_lag.COMMON_SHARE, window)


def test_every_node_has_unit_variance_on_the_clock():
    network = small_network()
    for follower, group in network.links.groupby("follower"):
        k = len(group)
        total = group.loading.iloc[0] ** 2 * (k + k * (k - 1) * lead_lag.COMMON_SHARE) + lead_lag.SECTOR_VARIANCE + lead_lag.IDIOSYNCRATIC_VARIANCE
        assert total == pytest.approx(1.0)


def test_cumulative_shape_integrates_the_spike_shape():
    grid = np.linspace(1800.0, 3000.0, 200001)
    numerical = float(np.trapezoid(lead_lag.shape(grid), grid))
    assert numerical == pytest.approx(float(lead_lag.cumulative_shape(3000.0) - lead_lag.cumulative_shape(1800.0)), rel=1e-5)


def test_event_simulation_is_seed_reproducible():
    network = small_network()
    follower = 3
    first = lead_lag.simulate_event(network, np.random.default_rng(11), followers=[follower])
    second = lead_lag.simulate_event(network, np.random.default_rng(11), followers=[follower])
    np.testing.assert_array_equal(first[follower][1], second[follower][1])
    assert set(first) == {0, 1, 2, follower}


def test_var_design_is_stable_and_sparse():
    a, sigma = lead_lag.var_design(0.06, network=small_network())
    assert np.max(np.abs(np.linalg.eigvals(a))) < 1 and np.count_nonzero(a - np.diag(np.diag(a))) == 5
    np.testing.assert_allclose(np.diag(sigma), 1.0)
