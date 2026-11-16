"""Reference tests for CR1 cluster-robust standard errors."""

import numpy as np
import pytest

from information_diffusion.statistical_framework import cluster_robust


def test_hand_computed_small_case():
    X = np.column_stack([np.ones(6), np.arange(6.0)])
    y = np.array([1.0, 2.0, 2.0, 4.0, 3.0, 6.0])
    fit = cluster_robust.cr1(y, X, [0, 0, 1, 1, 2, 2])
    assert fit.coefficients == pytest.approx([6 / 7, 6 / 7])
    assert fit.covariance == pytest.approx(np.array([[915 / 9604, -219 / 9604], [-219 / 9604, 291 / 48020]]))
    assert fit.clusters == 3 and fit.degrees_of_freedom == 2


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        cluster_robust.cr1(np.ones(4), np.ones((4, 1)), [0, 0, 0, 0])
    with pytest.raises(ValueError):
        cluster_robust.cr1(np.ones(4), np.ones((3, 1)), [0, 0, 1, 1])
