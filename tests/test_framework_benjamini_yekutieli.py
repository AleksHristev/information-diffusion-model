"""Reference tests for the Benjamini-Yekutieli adjustment."""

import numpy as np
import pytest
from scipy import stats

from information_diffusion.statistical_framework import benjamini_yekutieli as by

BH_EXAMPLE = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]


def test_textbook_input_hand_computed():
    assert by.reject(BH_EXAMPLE, 0.05).tolist() == [True] * 3 + [False] * 12
    expected = [0.00497734, 0.00995469, 0.03152318]
    assert by.adjusted_pvalues(BH_EXAMPLE)[:3] == pytest.approx(expected, rel=1e-5)


def test_adjusted_pvalues_are_monotone_in_the_sorted_order():
    rng = np.random.default_rng(401)
    p = rng.uniform(size=50)
    adjusted = by.adjusted_pvalues(p)
    assert np.all(np.diff(adjusted[np.argsort(p)]) >= -1e-15)


def test_false_discovery_rate_under_dependence():
    rng = np.random.default_rng(402)
    m, m1, sims, q = 60, 12, 4000, 0.05
    z = np.sqrt(0.5) * rng.standard_normal((sims, 1)) + np.sqrt(0.5) * rng.standard_normal((sims, m))
    z[:, :m1] += 3.5
    p = 2 * stats.norm.sf(np.abs(z))
    fdp = []
    for row in p:
        rejected = by.reject(row, q)
        total = rejected.sum()
        fdp.append(rejected[m1:].sum() / max(total, 1))
    assert np.mean(fdp) <= q * (m - m1) / m + 3 * np.std(fdp) / np.sqrt(sims)
