"""Hayashi-Yoshida covariance and two-scale variance against hand-computed cases."""

import numpy as np
import pytest

from information_diffusion.influence import hayashi_yoshida as hy
from information_diffusion.influence import two_scale_variance as tsrv


def test_overlaps_by_hand():
    assert hy.covariance([0, 1, 2], [0, 1, 3], [0.5, 1.5], [0, 5]) == pytest.approx(15.0)


def test_only_overlapping_intervals_count():
    value = hy.covariance([0, 1, 3], [0, 1, 2], [0, 0.5, 2, 4], [0, 2, 6, 14])
    assert value == pytest.approx(1 * (2 + 4) + 1 * (4 + 8))


def test_touching_endpoints_do_not_overlap():
    assert hy.covariance([0, 1], [0, 3], [1, 2], [0, 4]) == 0.0


def test_gap_between_series_contributes_nothing():
    assert hy.covariance([0, 1, 2], [0, 1, 2], [5, 6, 7], [0, 1, 2]) == 0.0


def test_synchronous_grids_give_realised_covariance():
    rng = np.random.default_rng(0)
    t = np.sort(rng.random(500)) * 100
    x, y = np.cumsum(rng.standard_normal(500)), np.cumsum(rng.standard_normal(500))
    assert hy.covariance(t, x, t, y) == pytest.approx(float(np.sum(np.diff(x) * np.diff(y))))


def test_covariance_is_symmetric():
    rng = np.random.default_rng(1)
    tx, ty = np.sort(rng.random(60)) * 10, np.sort(rng.random(90)) * 10
    x, y = np.cumsum(rng.standard_normal(60)), np.cumsum(rng.standard_normal(90))
    assert hy.covariance(tx, x, ty, y) == pytest.approx(hy.covariance(ty, y, tx, x))


def test_sorted_overlap_matches_the_double_sum():
    rng = np.random.default_rng(2)
    tx, ty = np.sort(rng.random(40)) * 10, np.sort(rng.random(70)) * 10
    x, y = np.cumsum(rng.standard_normal(40)), np.cumsum(rng.standard_normal(70))
    brute = sum((x[i] - x[i - 1]) * (y[j] - y[j - 1]) for i in range(1, 40) for j in range(1, 70) if tx[i - 1] < ty[j] and ty[j - 1] < tx[i])
    assert hy.covariance(tx, x, ty, y) == pytest.approx(brute)


def test_recovers_a_known_covariance_under_asynchronous_sampling():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 1000.0, 200001)
    dt = t[1] - t[0]
    shocks = rng.standard_normal((2, t.size)) * np.sqrt(dt)
    rho = 0.6
    x = np.cumsum(shocks[0])
    y = np.cumsum(rho * shocks[0] + np.sqrt(1 - rho**2) * shocks[1])
    ix = np.sort(rng.choice(t.size, 5000, replace=False))
    iy = np.sort(rng.choice(t.size, 3000, replace=False))
    assert hy.covariance(t[ix], x[ix], t[iy], y[iy]) / 1000.0 == pytest.approx(rho, abs=0.03)


def test_two_scale_variance_removes_iid_noise_on_average():
    rng = np.random.default_rng(4)
    n, sigma, paths = 40000, 0.01, 40
    true_iv = n * sigma**2
    observed = [np.cumsum(sigma * rng.standard_normal(n)) + 0.05 * rng.standard_normal(n) for _ in range(paths)]
    assert np.sum(np.diff(observed[0]) ** 2) > 20 * true_iv
    assert np.mean([tsrv.two_scale_variance(path) for path in observed]) == pytest.approx(true_iv, rel=0.12)


def test_two_scale_variance_needs_a_few_points():
    assert np.isnan(tsrv.two_scale_variance([0.0, 1.0, 2.0]))
