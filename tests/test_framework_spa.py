"""Reference tests for Hansen's test of superior predictive ability."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import spa as spa_module


def _run(benchmark, alternatives, rng, replications=999):
    return spa_module.spa(benchmark, alternatives, block_length=1, rng=rng, replications=replications)


def test_p_values_are_ordered():
    rng = np.random.default_rng(901)
    result = _run(rng.standard_normal(200), rng.standard_normal((200, 5)) + 0.1, rng)
    assert result.p_lower <= result.p_consistent <= result.p_upper


def test_no_alternative_beats_the_benchmark_gives_p_one():
    rng = np.random.default_rng(902)
    n = 400
    result = _run(np.zeros(n), np.ones((n, 3)) + rng.standard_normal((n, 3)), rng)
    assert result.statistic == 0.0
    assert result.p_consistent == 1.0


def test_clearly_superior_alternative_rejects():
    rng = np.random.default_rng(903)
    n = 300
    result = _run(rng.standard_normal(n) + 1.0, rng.standard_normal((n, 2)), rng)
    assert result.p_consistent < 0.01


def test_one_alternative_matches_the_normal_tail():
    rng = np.random.default_rng(904)
    n = 2000
    z = rng.standard_normal(n)
    d = (z - z.mean()) / z.std() + 1.5 / np.sqrt(n)
    result = _run(d, np.zeros(n), rng, replications=20_000)
    assert result.statistic > 0
    assert result.p_consistent == pytest.approx(stats.norm.sf(result.statistic), abs=0.015)


def test_poor_alternatives_do_not_hurt_the_consistent_test_but_hurt_the_reality_check():
    rng = np.random.default_rng(905)
    p_c, p_rc = [], []
    for _ in range(60):
        n = 250
        benchmark = rng.standard_normal(n)
        alternatives = rng.standard_normal((n, 5)) + np.array([0.0, 2.0, 2.0, 2.0, 2.0])
        result = _run(benchmark, alternatives, rng, replications=299)
        p_c.append(result.p_consistent)
        p_rc.append(result.p_upper)
    assert np.mean(p_c) <= np.mean(p_rc)


def test_zero_variance_alternative_is_ignored():
    rng = np.random.default_rng(906)
    benchmark = rng.standard_normal(50)
    result = _run(benchmark, np.column_stack([benchmark, benchmark + 1.0]), rng, replications=199)
    assert np.isfinite(result.statistic) and result.statistic == 0.0


def test_invalid_inputs_are_rejected():
    rng = np.random.default_rng(907)
    with pytest.raises(ValueError):
        spa_module.spa(np.ones(5), np.ones((4, 2)), block_length=1, rng=rng)
    with pytest.raises(ValueError):
        spa_module.spa(np.ones((5, 2)), np.ones((5, 2)), block_length=1, rng=rng)
