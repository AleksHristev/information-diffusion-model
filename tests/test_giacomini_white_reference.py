"""The stored critical values of the Giacomini-White statistic and the AR(1) and tail-index diagnostic."""

import numpy as np
import pytest

from information_diffusion.statistical_framework import giacomini_white as gw
from information_diffusion.statistical_framework import giacomini_white_critical_values as table
from information_diffusion.statistical_framework import giacomini_white_diagnostic as diagnostic


@pytest.mark.parametrize("n", [21, 6, 8])
def test_table_gives_each_tail_probability_back_at_its_own_critical_value(n):
    probabilities = np.asarray(table.PROBABILITIES)
    critical = np.asarray(table.CRITICAL_VALUES[n])
    assert np.all(np.diff(critical) <= 0)
    for target in (0.05, 0.025, 0.05 / 3):
        index = int(np.argmin(np.abs(probabilities - target)))
        assert gw.simulated_p_value(critical[index], n) == pytest.approx(probabilities[index], rel=1e-2)


def test_unlisted_sizes_are_simulated_by_the_same_procedure():
    critical = gw.simulate_critical_values(5, 11, sims=100_000)
    assert critical.size == len(table.PROBABILITIES) and np.all(np.diff(critical) <= 1e-9)


def test_ar1_coefficient_matches_a_hand_calculation():
    d = [1.0, 2.0, 3.0, 4.0]
    assert diagnostic.ar1_coefficient(d) == pytest.approx(0.25)


def test_tail_index_recovers_student_degrees_of_freedom():
    d = np.random.default_rng(7).standard_t(4, 40_000)
    assert diagnostic.tail_index(d) == pytest.approx(4.0, abs=0.4)


def test_gaussian_data_give_a_large_tail_index():
    assert diagnostic.tail_index(np.random.default_rng(8).standard_normal(40_000)) > 40


