"""Compiled Hawkes state recursions and the Hoffmann-Rosenbaum-Yoshida contrast against direct computations and finite differences."""

import numpy as np
import pytest

from information_diffusion.epidemic import event_merging as em
from information_diffusion.epidemic import hawkes_multivariate as mv
from information_diffusion.influence import hoffmann_rosenbaum_yoshida as hry

T, RELEASE = 5400.0, 1800.0


def _relative(new, old, floor: float = 0.0) -> float:
    new, old = np.asarray(new, dtype=float), np.asarray(old, dtype=float)
    mask = np.abs(old) > floor
    np.testing.assert_array_equal(np.isfinite(new[mask]), np.isfinite(old[mask]))
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(new[mask] - old[mask]) / np.abs(old[mask])))


@pytest.mark.parametrize("beta", (0.3, 1.0, 7.0, 40.0))
def test_compiled_grid_states_match_the_exact_double_sum(beta):
    rng = np.random.default_rng(int(beta * 10))
    times_list = [np.sort(rng.uniform(0.0, T, count)) for count in (60, 400)]
    pooled = mv._Pooled(times_list, T, 0.001)
    R, moments, _, _ = pooled.states_and_moments(beta)
    for source, event_times in enumerate(times_list):
        exact_states = np.empty(pooled.grid.size)
        exact_moments = np.empty(pooled.grid.size)
        for position, point in enumerate(pooled.grid):
            lags = point - event_times[event_times < point]
            exact_states[position] = np.exp(-beta * lags).sum()
            exact_moments[position] = (lags * np.exp(-beta * lags)).sum()
        assert _relative(R[:, source], exact_states, floor=1e-250) <= 1e-12
        assert _relative(moments[:, source], exact_moments, floor=1e-250) <= 1e-12


def test_hry_short_series_return_zeros():
    grid = hry.lag_grid(step=0.5, max_lag=1.0)
    np.testing.assert_array_equal(hry.event_contrast([1.0], [0.1], [1.0, 2.0], [0.1, 0.2], grid), np.zeros(grid.size))


def test_multivariate_moment_states_are_the_beta_derivative():
    rng = np.random.default_rng(4)
    times_list = [em.merge_events(np.sort(rng.uniform(0.0, 800.0, count)))[0] for count in (120, 300)]
    pooled = mv._Pooled(times_list, 800.0, 0.001)
    beta, h = 1.2, 1e-6
    R, M, comp, dcomp = pooled.states_and_moments(beta)
    np.testing.assert_allclose(M, (pooled.states(beta - h)[0] - pooled.states(beta + h)[0]) / (2.0 * h), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dcomp, (pooled.states(beta + h)[1] - pooled.states(beta - h)[1]) / (2.0 * h), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(R, pooled.states(beta)[0])
    np.testing.assert_allclose(comp, pooled.states(beta)[1])


