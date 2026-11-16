"""Hoffmann-Rosenbaum-Yoshida contrast against the direct double sum, and recovery on a clean lagged signal."""

import numpy as np
import pytest

from information_diffusion.influence import hayashi_yoshida as hy
from information_diffusion.influence import hoffmann_rosenbaum_yoshida as hry


def _direct(tx, x, ty, y, theta):
    return sum((x[i] - x[i - 1]) * (y[j] - y[j - 1]) for i in range(1, tx.size) for j in range(1, ty.size) if tx[i - 1] < ty[j] - theta and ty[j - 1] - theta < tx[i])


def test_contrast_matches_direct_sum_at_several_shifts():
    rng = np.random.default_rng(1)
    tx, ty = np.sort(rng.random(40)) * 20, np.sort(rng.random(60)) * 20
    x, y = np.cumsum(rng.standard_normal(40)), np.cumsum(rng.standard_normal(60))
    grid = hry.lag_grid(step=0.05, max_lag=3.0)
    contrast = hry.event_contrast(tx, x, ty, y, grid)
    for theta in (-2.75, -0.5, 0.0, 0.35, 2.9):
        assert contrast[int(np.argmin(np.abs(grid - theta)))] == pytest.approx(_direct(tx, x, ty, y, float(grid[np.argmin(np.abs(grid - theta))])))


def test_contrast_at_zero_lag_is_the_hy_covariance():
    rng = np.random.default_rng(2)
    tx, ty = np.sort(rng.random(50)) * 10, np.sort(rng.random(80)) * 10
    x, y = np.cumsum(rng.standard_normal(50)), np.cumsum(rng.standard_normal(80))
    grid = hry.lag_grid(step=0.5, max_lag=2.0)
    assert hry.event_contrast(tx, x, ty, y, grid)[grid.size // 2] == pytest.approx(hy.covariance(tx, x, ty, y))


def test_chunking_does_not_change_the_contrast():
    rng = np.random.default_rng(3)
    tx, ty = np.sort(rng.random(400)) * 50, np.sort(rng.random(500)) * 50
    x, y = np.cumsum(rng.standard_normal(400)), np.cumsum(rng.standard_normal(500))
    grid = hry.lag_grid(step=0.05, max_lag=2.0)
    np.testing.assert_allclose(hry.event_contrast(tx, x, ty, y, grid, batch_pairs=100), hry.event_contrast(tx, x, ty, y, grid))


def _lagged_pair(rng, lag, seconds, rate=100.0):
    grid = np.arange(-5.0, seconds + 1e-9, 0.001)
    path = np.cumsum(rng.standard_normal(grid.size) * np.sqrt(0.001))
    tx = np.sort(rng.random(int(seconds * rate)) * seconds)
    ty = np.sort(rng.random(int(seconds * rate)) * seconds)
    index = np.minimum(np.searchsorted(grid, tx), grid.size - 1)
    x = path[index]
    y = path[np.minimum(np.searchsorted(grid, ty - lag), grid.size - 1)]
    return tx, x, ty, y


@pytest.mark.parametrize("lag", [0.05, 0.3, 1.5, -0.2])
def test_estimate_recovers_planted_lag(lag):
    rng = np.random.default_rng(7)
    grid = hry.lag_grid(step=0.001, max_lag=3.0)
    contrasts = np.array([hry.event_contrast(*_lagged_pair(rng, lag, 300.0), grid) for _ in range(5)])
    assert hry.estimate_lag(contrasts, grid) == pytest.approx(lag, abs=0.02)


def test_bootstrap_lags_hit_the_truth_on_a_clean_signal():
    rng = np.random.default_rng(8)
    lag = 0.25
    grid = hry.lag_grid(step=0.01, max_lag=2.0)
    contrasts = np.array([hry.event_contrast(*_lagged_pair(rng, lag, 200.0), grid) for _ in range(12)])
    boots = hry.bootstrap_lags(contrasts, grid, 99, rng)
    assert np.percentile(boots, 2.5) <= lag + 0.05 and np.percentile(boots, 97.5) >= lag - 0.05


def test_plateau_of_equal_values_returns_its_midpoint():
    values = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1])
    assert hry._plateau_midpoint(values) == 3


def test_estimate_on_a_flat_contrast_returns_a_grid_point():
    grid = hry.lag_grid(step=0.1, max_lag=1.0)
    assert hry.estimate_lag(np.zeros(grid.size), grid) in grid
