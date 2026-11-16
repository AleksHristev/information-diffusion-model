"""Hand cases for the price-discovery estimators: ADF, VECM, information shares, Gonzalo-Granger, ILS and the minute lead-lag."""

import numpy as np
import pytest

from information_diffusion.influence import adf, post_decision_lag as pdl, price_discovery as pdm, vecm
from information_diffusion.simulation import price_discovery as dgp


def test_adf_statistic_on_a_hand_series():
    result = adf.adf_at(np.array([1.0, 0.0, 2.0, 1.0, 3.0]), np.arange(1, 5), lags=0)
    assert result.rho == pytest.approx(-1.5)
    assert result.intercept == pytest.approx(2.0)
    assert result.statistic == pytest.approx(-np.sqrt(2.0))


def test_rows_never_span_a_dropped_interval():
    keep = np.array([True, True, True, False, True, True, True, True])
    np.testing.assert_array_equal(adf.run_position(keep), [1, 2, 3, 0, 1, 2, 3, 4])
    np.testing.assert_array_equal(adf.valid_rows(keep, 0), [1, 2, 5, 6, 7])
    np.testing.assert_array_equal(adf.valid_rows(keep, 1), [2, 6, 7])


def test_masked_statistic_equals_the_regression_on_kept_rows_only():
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.standard_normal(400))
    keep = np.ones(400, dtype=bool)
    keep[100:150] = False
    keep[300:310] = False
    rows = adf.valid_rows(keep, 1)
    x = np.column_stack([np.ones(rows.size), series[rows - 1], series[rows - 1] - series[rows - 2]])
    y = series[rows] - series[rows - 1]
    beta, rss = np.linalg.lstsq(x, y, rcond=None)[:2]
    se = np.sqrt(rss[0] / (rows.size - 3) * np.linalg.inv(x.T @ x)[1, 1])
    assert adf.adf(series, keep, lags=1).statistic == pytest.approx(beta[1] / se)


def test_mackinnon_critical_values():
    assert adf.mackinnon_critical_value(0.05, 100) == pytest.approx(-2.86154 - 2.8903 / 100 - 4.234 / 100**2 - 40.040 / 100**3)
    assert adf.mackinnon_critical_value(0.05, 100) == pytest.approx(-2.8909, abs=1e-4)
    assert adf.mackinnon_critical_value(0.01, 10**9) == pytest.approx(-3.43035, abs=1e-6)
    assert adf.mackinnon_critical_value(0.10, 10**9) == pytest.approx(-2.56677, abs=1e-6)


def test_simulated_p_value():
    null = np.array([-4.0, -3.0, -2.0, -1.0])
    assert adf.simulated_p_value(-2.5, null) == pytest.approx(3 / 5)
    assert adf.simulated_critical_value(np.linspace(-3, 1, 401), 0.05) == pytest.approx(-2.8)


def test_adf_rejects_a_stationary_spread_and_not_a_walk():
    rng = np.random.default_rng(2)
    stationary = np.zeros(3000)
    for t in range(1, 3000):
        stationary[t] = 0.7 * stationary[t - 1] + rng.standard_normal()
    assert adf.adf(stationary).statistic < adf.mackinnon_critical_value(0.01, 3000)
    assert adf.adf(np.cumsum(rng.standard_normal(3000))).statistic > adf.mackinnon_critical_value(0.01, 3000)


def _random_inputs(rng):
    alpha = np.array([-rng.uniform(0.01, 0.5), rng.uniform(0.01, 0.5)])
    a = rng.normal(size=(2, 2))
    return alpha, a @ a.T + 0.1 * np.eye(2)


def test_gonzalo_granger_shares_sum_to_one_and_have_the_closed_form():
    alpha = np.array([-0.05, 0.10])
    shares = pdm.component_shares(alpha)
    assert shares.sum() == pytest.approx(1.0)
    assert shares[0] == pytest.approx(0.10 / 0.15)
    rng = np.random.default_rng(3)
    for _ in range(50):
        assert pdm.component_shares(rng.normal(size=2)).sum() == pytest.approx(1.0)


def test_information_share_bounds_bracket_every_ordering():
    rng = np.random.default_rng(4)
    for _ in range(200):
        alpha, sigma = _random_inputs(rng)
        first, second = pdm.information_share_orderings(alpha, sigma)
        lower, upper, midpoint = pdm.information_share_bounds(alpha, sigma)
        assert 0.0 <= lower <= first <= upper <= 1.0 and lower <= second <= upper
        assert lower <= midpoint <= upper


def test_information_share_matches_the_baillie_et_al_closed_form():
    alpha = np.array([-0.05, 0.10])
    s_k, s_p, r = 1.0, 1.5, 0.3
    sigma = np.array([[s_k**2, r * s_k * s_p], [r * s_k * s_p, s_p**2]])
    g_k, g_p = pdm.component_shares(alpha)
    top = (g_k * s_k + g_p * r * s_p) ** 2
    expected_first = top / (top + (g_p * s_p * np.sqrt(1 - r**2)) ** 2)
    first, _ = pdm.information_share_orderings(alpha, sigma)
    assert first == pytest.approx(expected_first)
    top_p = (g_p * s_p + g_k * r * s_k) ** 2
    expected_second = 1 - top_p / (top_p + (g_k * s_k * np.sqrt(1 - r**2)) ** 2)
    assert pdm.information_share_orderings(alpha, sigma)[1] == pytest.approx(expected_second)


def test_uncorrelated_innovations_give_equal_bounds():
    alpha, sigma = np.array([-0.2, 0.1]), np.diag([1.0, 4.0])
    g_k, g_p = pdm.component_shares(alpha)
    expected = g_k**2 / (g_k**2 + 4 * g_p**2)
    lower, upper, _ = pdm.information_share_bounds(alpha, sigma)
    assert lower == pytest.approx(expected) and upper == pytest.approx(expected)


def test_ils_lies_in_the_unit_interval_and_is_half_when_symmetric():
    rng = np.random.default_rng(5)
    for _ in range(200):
        alpha, sigma = _random_inputs(rng)
        ils = pdm.measures(alpha, sigma)[4]
        assert 0.0 <= ils <= 1.0
    symmetric = pdm.measures(np.array([-0.1, 0.1]), np.array([[1.0, 0.3], [0.3, 1.0]]))
    assert symmetric[3] == pytest.approx(0.5)
    assert symmetric[2] == pytest.approx(0.5)
    assert symmetric[4] == pytest.approx(0.5)


def test_ils_formula():
    assert pdm.information_leadership_share(0.6, 0.4) == pytest.approx(2.25 / (2.25 + 1 / 2.25))


def test_vecm_recovers_alpha_and_sigma_on_a_long_series():
    rng = np.random.default_rng(6)
    alpha, sigma = dgp.vecm_parameters(0.7, 6, 1.0, 1.5, 0.3)
    gamma1 = np.array([[0.10, 0.05], [0.05, 0.10]])
    kalshi, polymarket = dgp.simulate_vecm(alpha, gamma1, sigma, 60_000, 1, rng, df=5.0)
    pair = vecm.Pair(kalshi[0], polymarket[0], np.ones(60_000, dtype=bool), np.arange(60_000))
    fit = vecm.fit(vecm.build_design([pair], 1))
    np.testing.assert_allclose(fit.alpha, alpha, atol=0.01)
    np.testing.assert_allclose(fit.gamma[0], gamma1, atol=0.02)
    np.testing.assert_allclose(fit.sigma, sigma, rtol=0.05, atol=0.03)
    assert pdm.component_shares(alpha)[0] == pytest.approx(0.7)


def test_vecm_design_skips_gaps_and_stacks_pairs_with_their_own_intercepts():
    keep = np.array([True] * 5 + [False] + [True] * 4)
    x = np.arange(10, dtype=float)
    pair_a = vecm.Pair(x, 0.5 * x, keep, np.arange(10))
    pair_b = vecm.Pair(2 * x, x, np.ones(10, dtype=bool), np.arange(10))
    design = vecm.build_design([pair_a, pair_b], 1)
    assert design.x.shape[0] == 5 + 8
    np.testing.assert_array_equal(design.key[:5], [2, 3, 4, 8, 9])
    assert design.x[:5, 0].sum() == 5 and design.x[5:, 1].sum() == 8
    np.testing.assert_allclose(design.x[:5, 2], 0.5 * np.array([1, 2, 3, 7, 8]))


def test_weighted_refit_with_integer_weights_equals_the_duplicated_sample():
    rng = np.random.default_rng(7)
    alpha, sigma = dgp.vecm_parameters(0.3, 6, 1.0, 1.0, 0.2)
    kalshi, polymarket = dgp.simulate_vecm(alpha, np.zeros((2, 2)), sigma, 500, 1, rng)
    design = vecm.build_design([vecm.Pair(kalshi[0], polymarket[0], np.ones(500, dtype=bool), np.arange(500))], 0)
    weights = rng.integers(0, 3, size=design.x.shape[0]).astype(float)
    weighted = vecm.fit(design, weights)
    repeat = np.repeat(np.arange(design.x.shape[0]), weights.astype(int))
    duplicated = vecm.fit(vecm.Design(design.x[repeat], design.y[repeat], design.key[repeat], 0, 1))
    np.testing.assert_allclose(weighted.coefficients, duplicated.coefficients)
    np.testing.assert_allclose(weighted.sigma, duplicated.sigma)


def test_cross_products_align_kalshi_leading_by_one_minute():
    rng = np.random.default_rng(8)
    kalshi = rng.standard_normal(30)
    polymarket = np.concatenate(([0.0], kalshi[:-1]))
    cross = pdl.cross_products(kalshi, polymarket, (1, 30))
    assert pdl.peak_lag(cross) == 1.0
    assert cross[pdl.MAX_LAG + 1] == pytest.approx(float(kalshi[:-1] @ kalshi[:-1]))


def test_centroid_of_a_triangle_kernel_is_the_delay():
    cross = np.zeros(2 * pdl.MAX_LAG + 1)
    cross[pdl.MAX_LAG] = 0.7
    cross[pdl.MAX_LAG + 1] = 0.3
    assert pdl.centroid(cross) == pytest.approx(0.3)
    cross = np.zeros(2 * pdl.MAX_LAG + 1)
    cross[pdl.MAX_LAG - 2], cross[pdl.MAX_LAG - 1] = 0.25, 0.75
    assert pdl.centroid(cross) == pytest.approx(-1.25)


def test_minute_changes_set_undefined_changes_to_zero():
    np.testing.assert_allclose(pdl.minute_changes(np.array([np.nan, np.nan, 0.4, 0.5])), [0.0, 0.0, 0.1])


def test_window_bounds_keep_pairs_inside_the_window():
    kalshi = np.zeros(90)
    polymarket = np.zeros(90)
    kalshi[29] = 1.0
    polymarket[30] = 1.0
    assert pdl.cross_products(kalshi, polymarket, (1, 30)).sum() == 0.0
    assert pdl.cross_products(kalshi, polymarket, (1, 90))[pdl.MAX_LAG + 1] == 1.0
