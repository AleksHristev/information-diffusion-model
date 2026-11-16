"""The Giacomini-White p-value read from simulated draws of the absolute statistic."""

import numpy as np
import pytest

from information_diffusion.statistical_framework import giacomini_white as gw

DRAWS = [0.5, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


@pytest.mark.parametrize("statistic,expected", [(0.0, 10 / 10), (2.5, 6 / 10), (-2.5, 6 / 10), (2.0, 8 / 10), (7.0, 2 / 10), (8.0, 1 / 10)])
def test_the_p_value_is_one_plus_the_draws_at_least_as_extreme_over_replications_plus_one(statistic, expected):
    assert float(gw.design_p_value(statistic, DRAWS)) == pytest.approx(expected)


def test_the_p_value_does_not_depend_on_the_order_of_the_draws_and_is_vectorised():
    np.testing.assert_allclose(gw.design_p_value(np.array([2.5, 8.0]), DRAWS[::-1]), [0.6, 0.1])


