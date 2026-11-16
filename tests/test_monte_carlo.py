"""The Monte Carlo runner's independence of the worker count, the Benjamini-Yekutieli adjustment and bootstrap indices."""

import numpy as np
import pytest

from information_diffusion.simulation import monte_carlo
from information_diffusion.statistical_framework import benjamini_yekutieli as by
from information_diffusion.statistical_framework import block_bootstrap


def _draw(stream):
    return float(np.random.default_rng(stream).random())


def test_replicates_do_not_depend_on_worker_count():
    assert monte_carlo.run_replicates(_draw, 8, 123, workers=1) == monte_carlo.run_replicates(_draw, 8, 123, workers=2)


def test_benjamini_yekutieli_on_a_hand_worked_case():
    p = [0.001, 0.01, 0.04, 0.5]
    assert by.adjusted_pvalues(p) == pytest.approx([1 / 120, 1 / 24, 1 / 9, 1.0])
    assert by.reject(p, 0.05).tolist() == [True, True, False, False]


def test_moving_blocks_are_contiguous_and_the_right_length():
    indices = block_bootstrap.moving_block_indices(103, 10, np.random.default_rng(1))
    assert len(indices) == 103 and indices.max() < 103 and np.all(np.diff(indices[:10]) == 1)


def test_stationary_bootstrap_has_the_requested_mean_block_length():
    rng = np.random.default_rng(2)
    indices = block_bootstrap.stationary_indices(200_000, 8.0, rng)
    continues = np.mean(np.diff(indices) % 200_000 == 1)
    assert 1 / (1 - continues) == pytest.approx(8.0, rel=0.02) and indices.min() >= 0 and indices.max() < 200_000
