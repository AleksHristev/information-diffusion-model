"""Reference tests for the Holm step-down procedure."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import holm

EXAMPLE = [0.01, 0.04, 0.03, 0.005]
EXAMPLE_ADJUSTED = [0.03, 0.06, 0.06, 0.02]
BH_EXAMPLE = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]


def test_worked_example():
    assert holm.adjusted_pvalues(EXAMPLE) == pytest.approx(EXAMPLE_ADJUSTED)
    assert holm.reject(EXAMPLE).tolist() == [True, False, False, True]


def test_rejection_matches_the_step_down_definition():
    rng = np.random.default_rng(301)
    for _ in range(200):
        p = rng.uniform(size=rng.integers(2, 12)) ** rng.uniform(1, 3)
        order = np.argsort(p)
        m = p.size
        expected = np.zeros(m, dtype=bool)
        for rank, index in enumerate(order):
            if p[index] <= 0.05 / (m - rank):
                expected[index] = True
            else:
                break
        assert holm.reject(p, 0.05).tolist() == expected.tolist()


def test_textbook_input_from_benjamini_hochberg():
    assert holm.reject(BH_EXAMPLE).tolist() == [True] * 3 + [False] * 12
    assert holm.adjusted_pvalues(BH_EXAMPLE)[:3] == pytest.approx([0.0015, 0.0056, 0.0247])


def test_all_ones_and_empty():
    assert holm.adjusted_pvalues([1.0, 1.0]) == pytest.approx([1.0, 1.0])
    assert holm.adjusted_pvalues([]).size == 0


def test_invalid_pvalues_are_rejected():
    with pytest.raises(ValueError):
        holm.adjusted_pvalues([0.1, 1.2])
    with pytest.raises(ValueError):
        holm.adjusted_pvalues([0.1, np.nan])


def test_family_wise_error_is_controlled_under_the_null():
    rng = np.random.default_rng(302)
    m, sims = 5, 20000
    for rho in (0.0, 0.7):
        z = np.sqrt(rho) * rng.standard_normal((sims, 1)) + np.sqrt(1 - rho) * rng.standard_normal((sims, m))
        p = 2 * stats.norm.sf(np.abs(z))
        fwer = np.mean([holm.reject(row, 0.05).any() for row in p])
        assert fwer <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / sims)
