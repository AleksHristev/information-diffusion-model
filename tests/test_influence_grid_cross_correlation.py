"""One-second grid cross-correlation lead-lag."""

import numpy as np
import pytest

from information_diffusion.influence import grid_cross_correlation as gcc


def test_previous_tick_returns_on_a_known_grid():
    returns = gcc.previous_tick_returns([0.5, 1.5, 2.5], [10.0, 20.0, 30.0], 3.0)
    assert returns.tolist() == [0.0, 10.0, 10.0]


def test_estimate_recovers_an_integer_lag_on_a_clean_grid():
    rng = np.random.default_rng(1)
    rx = rng.standard_normal(5000)
    ry = np.roll(rx, 3)
    ry[:3] = 0.0
    lags = gcc.lag_grid()
    assert gcc.estimate_lag(gcc.event_cross_products(rx, ry), lags) == pytest.approx(3.0)


def test_negative_lag_means_y_leads():
    rng = np.random.default_rng(2)
    ry = rng.standard_normal(5000)
    rx = np.roll(ry, 2)
    rx[:2] = 0.0
    assert gcc.estimate_lag(gcc.event_cross_products(rx, ry), gcc.lag_grid()) == pytest.approx(-2.0)


def test_cross_products_pool_over_events():
    rng = np.random.default_rng(3)
    lags = gcc.lag_grid(max_lag=2)
    matrix = np.array([gcc.event_cross_products(rng.standard_normal(300), rng.standard_normal(300), max_lag=2) for _ in range(4)])
    assert gcc.estimate_lag(matrix, lags) == gcc.estimate_lag(matrix.sum(axis=0), lags)


def test_bootstrap_lags_have_the_requested_shape():
    rng = np.random.default_rng(4)
    matrix = np.array([gcc.event_cross_products(rng.standard_normal(300), rng.standard_normal(300)) for _ in range(5)])
    assert gcc.bootstrap_lags(matrix, gcc.lag_grid(), 25, rng).shape == (25,)
