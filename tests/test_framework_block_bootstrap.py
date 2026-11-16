"""Reference tests for the block-bootstrap index generators."""

import numpy as np
import pytest
from scipy.signal import lfilter

from information_diffusion.statistical_framework import block_bootstrap


def _bootstrap_variance_of_mean(x, generator, replications):
    return np.var([x[generator()].mean() for _ in range(replications)])


def test_moving_block_variance_matches_its_closed_form():
    rng = np.random.default_rng(501)
    x = rng.standard_normal(120)
    b, k = 10, 12
    blocks = np.lib.stride_tricks.sliding_window_view(x, b).mean(axis=1)
    closed_form = blocks.var() / k
    simulated = _bootstrap_variance_of_mean(x, lambda: block_bootstrap.moving_block_indices(120, b, rng), 60_000)
    assert simulated == pytest.approx(closed_form, rel=0.03)


def test_stationary_bootstrap_variance_matches_politis_romano():
    rng = np.random.default_rng(502)
    n, mean_block = 100, 5.0
    x = rng.standard_normal(n)
    c = x - x.mean()
    gamma = np.array([np.sum(c[: n - i] * c[i:]) / n for i in range(n)])
    q = 1 / mean_block
    weights = (1 - np.arange(1, n) / n) * (1 - q) ** np.arange(1, n) + (np.arange(1, n) / n) * (1 - q) ** (n - np.arange(1, n))
    closed_form = gamma[0] + 2 * np.sum(weights * gamma[1:])
    simulated = n * _bootstrap_variance_of_mean(x, lambda: block_bootstrap.stationary_indices(n, mean_block, rng), 60_000)
    assert simulated == pytest.approx(closed_form, rel=0.03)


def test_moving_block_bootstrap_recovers_the_ar1_long_run_variance():
    rng = np.random.default_rng(503)
    n, b, series = 20_000, 60, 8
    estimates = []
    for _ in range(series):
        x = lfilter([1.0], [1.0, -0.5], rng.standard_normal(n))
        estimates.append(n * _bootstrap_variance_of_mean(x, lambda: block_bootstrap.moving_block_indices(n, b, rng), 400))
    assert np.mean(estimates) == pytest.approx(4.0, rel=0.1)


def test_event_indices_draw_whole_events_with_replacement():
    rng = np.random.default_rng(504)
    draws = block_bootstrap.event_indices(33, rng)
    assert draws.shape == (33,) and draws.min() >= 0 and draws.max() < 33
