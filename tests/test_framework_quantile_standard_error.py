"""Checks of the Maritz-Jarrett standard error of a sample quantile."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import quantile_standard_error as qse


def test_hand_worked_median_of_five():
    assert qse.standard_error([1.0, 2.0, 3.0, 4.0, 5.0], 0.5) == pytest.approx(np.sqrt(1.328))


def test_it_matches_the_large_sample_formula_for_a_normal_95th_percentile():
    n = 2000
    draws = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    expected = np.sqrt(0.95 * 0.05 / n) / stats.norm.pdf(stats.norm.ppf(0.95))
    assert qse.standard_error(draws, 0.95) == pytest.approx(expected, rel=0.1)


def test_it_ignores_order_and_location_and_scales_with_the_draws():
    draws = np.random.default_rng(1).standard_t(5, 500)
    reference = qse.standard_error(draws, 0.95)
    assert qse.standard_error(draws[::-1], 0.95) == pytest.approx(reference)
    assert qse.standard_error(draws + 7.0, 0.95) == pytest.approx(reference)
    assert qse.standard_error(draws * 3.0, 0.95) == pytest.approx(3.0 * reference)


def test_it_shrinks_as_the_replications_grow():
    draws = np.random.default_rng(2).standard_normal(8000)
    assert qse.standard_error(draws, 0.95) < qse.standard_error(draws[:500], 0.95)


@pytest.mark.parametrize("draws,probability", [([1.0, 2.0], 0.5), (np.arange(10.0), 0.99), (np.arange(10.0), 0.01)])
def test_a_quantile_at_the_edge_of_the_sample_is_refused(draws, probability):
    with pytest.raises(ValueError):
        qse.standard_error(draws, probability)
