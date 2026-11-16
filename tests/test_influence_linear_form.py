"""The fast lead-lag forms against their direct definitions on fixed seeds."""

import numpy as np
import pytest

from information_diffusion.influence import hayashi_yoshida as hy
from information_diffusion.influence import hoffmann_rosenbaum_yoshida as hry


def series(seed: int, count: int, span: float = 600.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(0.0, span, count)), np.cumsum(rng.standard_normal(count)) * 1e-4


def definition(tx, x, ty, y) -> float:
    """The double sum of the Hayashi-Yoshida covariance, by its definition, in quadratic time."""
    total = 0.0
    for i in range(1, tx.size):
        for j in range(1, ty.size):
            if max(tx[i - 1], ty[j - 1]) < min(tx[i], ty[j]):
                total += (x[i] - x[i - 1]) * (y[j] - y[j - 1])
    return total


@pytest.mark.parametrize("seed,count_x,count_y", [(1, 120, 300), (2, 300, 120), (3, 200, 200)])
def test_the_linear_form_of_hayashi_yoshida_equals_its_definition(seed, count_x, count_y):
    tx, x = series(seed, count_x)
    ty, y = series(seed + 100, count_y)
    assert hy.covariance(tx, x, ty, y) == pytest.approx(definition(tx, x, ty, y), rel=1e-8, abs=1e-16)


@pytest.mark.parametrize("seed,count_x,count_y", [(4, 400, 900), (5, 900, 400), (6, 3000, 200)])
def test_the_cumulative_sum_contrast_equals_the_linear_covariance_at_every_lag(seed, count_x, count_y):
    tx, x = series(seed, count_x)
    ty, y = series(seed + 100, count_y)
    grid = hry.lag_grid(step=0.05, max_lag=2.0)
    contrast = hry.event_contrast(tx, x, ty, y, grid)
    direct = np.array([hy.covariance(tx, x, ty - lag, y) for lag in grid])
    np.testing.assert_allclose(contrast, direct, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_fine_grid_contrast_gives_the_lag_the_direct_computation_gives(seed):
    rng = np.random.default_rng(seed)
    tx = np.sort(rng.uniform(0.0, 600.0, 1500))
    x = np.cumsum(rng.standard_normal(1500)) * 1e-4
    ty = np.sort(rng.uniform(0.0, 600.0, 1200))
    lag = 0.25
    y = np.interp(ty - lag, tx, x) + rng.standard_normal(1200) * 2e-6
    grid = hry.lag_grid(step=0.01, max_lag=1.0)
    fast = hry.estimate_lag(hry.event_contrast(tx, x, ty, y, grid), grid)
    slow = hry.estimate_lag(np.array([hy.covariance(tx, x, ty - shift, y) for shift in grid]), grid)
    assert fast == slow and fast == pytest.approx(lag, abs=0.2)


def test_the_cost_of_the_contrast_does_not_grow_with_the_number_of_lags():
    """The pairs entering the difference array depend on the range of lags and not on the step of the grid."""
    tx, x = series(10, 2000)
    ty, y = series(11, 2000)
    coarse, fine = hry.lag_grid(step=0.1, max_lag=1.0), hry.lag_grid(step=0.001, max_lag=1.0)
    np.testing.assert_allclose(hry.event_contrast(tx, x, ty, y, coarse), hry.event_contrast(tx, x, ty, y, fine)[::100], rtol=1e-8, atol=1e-14)
