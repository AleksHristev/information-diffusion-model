"""Reference tests for the Freedman-Lane permutation test."""

import itertools

import numpy as np
import pytest

from information_diffusion.statistical_framework import freedman_lane as fl


def test_exact_case_matches_the_published_spearman_distribution():
    y = np.array([1.0, 3.0, 2.0, 4.0])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert fl.freedman_lane(y, x, exact=True).p_value == pytest.approx(8 / 24)
    assert fl.freedman_lane(y, x, exact=True, alternative="greater").p_value == pytest.approx(4 / 24)
    assert fl.freedman_lane(y, x, exact=True, alternative="less").p_value == pytest.approx(23 / 24)


def _reference_enumeration(y, x, z):
    n = y.size
    Z = np.column_stack([np.ones(n), z])
    X = np.column_stack([Z, x])
    fitted = Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    residual = y - fitted

    def t(values):
        beta = np.linalg.lstsq(X, values, rcond=None)[0]
        rss = np.sum((values - X @ beta) ** 2)
        return beta[-1] / np.sqrt(rss / (n - X.shape[1]) * np.linalg.inv(X.T @ X)[-1, -1])

    observed = t(y)
    stats_all = np.array([t(fitted + residual[list(perm)]) for perm in itertools.permutations(range(n))])
    return observed, np.mean(np.abs(stats_all) >= np.abs(observed) - 1e-9), np.mean(stats_all >= observed - 1e-9)


def test_exact_enumeration_with_a_nuisance_covariate():
    rng = np.random.default_rng(1001)
    y, x, z = rng.standard_normal(6), rng.standard_normal(6), rng.standard_normal(6)
    observed, two_sided, one_sided = _reference_enumeration(y, x, z)
    result = fl.freedman_lane(y, x, z, exact=True)
    assert result.statistic == pytest.approx(observed)
    assert result.p_value == pytest.approx(two_sided)
    assert fl.freedman_lane(y, x, z, exact=True, alternative="greater").p_value == pytest.approx(one_sided)


def test_strong_slope_is_detected_with_the_right_sign():
    rng = np.random.default_rng(1006)
    n = 30
    z = rng.standard_normal((n, 3))
    x = rng.standard_normal(n)
    y = 2.0 * x + z.sum(1) * 0.3 + rng.standard_normal(n)
    result = fl.freedman_lane(y, x, z, rng=rng)
    assert result.p_value < 0.01 and result.coefficient > 0
    assert fl.freedman_lane(y, x, z, rng=rng, alternative="greater").p_value < 0.01
    assert fl.freedman_lane(y, x, z, rng=rng, alternative="less").p_value > 0.9


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(1007)
    y = rng.standard_normal(10)
    with pytest.raises(ValueError):
        fl.freedman_lane(y, y[:5], None, rng=rng)
    with pytest.raises(ValueError):
        fl.freedman_lane(y, y, None, draws=99)
    with pytest.raises(ValueError):
        fl.freedman_lane(y, np.ones(10), None, rng=rng)
    with pytest.raises(ValueError):
        fl.freedman_lane(y, y, None, rng=rng, alternative="sideways")
    with pytest.raises(ValueError):
        fl.freedman_lane(np.ones(10), rng.standard_normal(10), None, rng=rng)
