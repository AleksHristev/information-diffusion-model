"""Statistics of the reaction order of a set of instruments: Kendall's W, ranks, quartiles and out-of-sample correlation."""

from __future__ import annotations

import numpy as np
import pytest

from information_diffusion.market_making import rank_statistics as rk
from information_diffusion.market_making import order_statistics as mo

NINETEEN = ["QQQ", "SPY", "IWM", "SOXX", "SMH", "XLK", "XLF", "XBI", "XLY", "XLI", "XLE", "IGV", "XLV", "VGT", "VNQ", "XOP", "KRE", "XLU", "XLP"]


def rolling_tests(m: int, window: int) -> list[mo.TestRelease]:
    return [mo.TestRelease(f"e{t}", t, tuple(range(t - window, t)), ", ".join(f"e{i}" for i in range(t - window, t))) for t in range(window, m)]


def test_identical_orders_give_one_everywhere_and_ranks_are_recomputed_within_the_set():
    m, n = 20, 19
    base = np.arange(1, n + 1, dtype=float) * 10.0
    lag = np.tile(base, (m, 1))
    gap = np.tile(base[::-1], (m, 1))
    rows, funds, oos = mo.order_statistics(lag, gap, lag.copy(), NINETEEN, list(range(3, n)), {}, rolling_tests(m, 12), seed=1, permutations=99)
    by = rows.set_index(["question", "measure"]).value
    assert by[("(1) agreement across releases", 'reaction (stored absorption time)')] == pytest.approx(1.0)
    assert by[("(2) out of sample", 'reaction (stored absorption time)')] == pytest.approx(1.0)
    assert by[("(4) second definition", 'Kendall tau between the two orders, per release')] == pytest.approx(1.0)
    assert by[("(3) quote gap", 'Spearman rho of the quote gap and the lag, per release')] == pytest.approx(-1.0)
    assert sorted(funds.mean_rank) == pytest.approx(list(range(1, 17)))
    assert set(rows.instruments.dropna()) == {16} and len(oos) == 2 * (m - 12)


def test_missing_lags_tie_last_and_the_quartiles_count_them_as_later():
    m = 12
    lag = np.tile(np.array([0.0, 1.0, 2.0, 5.0, np.nan]), (m, 1))
    rows, funds, _ = mo.order_statistics(lag, lag.copy(), lag.copy(), ["SPY", "A", "B", "C", "D"], [1, 2, 3, 4], {}, rolling_tests(m, 4), seed=1, permutations=9)
    f = funds.set_index("instrument")
    assert f.at["D", "never"] == m and np.isnan(f.at["D", "median_lag_ms"]) and f.at["D", "mean_rank"] == 4.0
    assert f.at["A", "mean_rank"] == 1.0 and f.at["C", "median_lag_ms"] == 5.0


def test_out_of_sample_uses_only_the_previous_releases():
    m, n = 6, 5
    lag = np.tile(np.arange(n, dtype=float), (m, 1))
    lag[5] = lag[5][::-1]
    test = [mo.TestRelease("e5", 5, (1, 2, 3, 4), "e1, e2, e3, e4")]
    frame = mo.oos(lag, test)
    assert frame.rho.iloc[0] == pytest.approx(-1.0) and frame.previous.iloc[0] == "e1, e2, e3, e4"
    lag[0] = lag[5]
    assert mo.oos(lag, test).rho.iloc[0] == pytest.approx(-1.0)


def simulate(rng, m, n_other, strength, index_on_top):
    """Lags: a common order of the other funds of the given strength plus noise; index funds at 0 to 2 ms when present."""
    common = np.arange(n_other, dtype=float)
    other = strength * common[None, :] + rng.normal(0, 1, (m, n_other))
    other = np.exp(other) * 100.0
    if not index_on_top:
        return other
    index = np.column_stack([np.zeros(m), rng.uniform(0, 2, m), rng.uniform(0, 2, m)])
    return np.hstack([index, other])


def test_w_of_the_sixteen_is_recovered_and_index_funds_on_top_raise_it():
    rng = np.random.default_rng(29)
    m, reps = 33, 40
    w16_null, w19_null, w16_signal, w16_alone = [], [], [], []
    for _ in range(reps):
        null = simulate(rng, m, 16, 0.0, True)
        w16_null.append(rk.kendall_w(mo.rank_matrix(null[:, 3:])))
        w19_null.append(rk.kendall_w(mo.rank_matrix(null)))
        signal = simulate(rng, m, 16, 0.3, True)
        w16_signal.append(rk.kendall_w(mo.rank_matrix(signal[:, 3:])))
        w16_alone.append(rk.kendall_w(mo.rank_matrix(simulate(rng, m, 16, 0.3, False))))
    assert np.mean(w16_null) == pytest.approx(1.0 / m, abs=0.01)
    assert np.mean(w19_null) > 0.25
    assert np.mean(w16_signal) == pytest.approx(np.mean(w16_alone), abs=0.03)
    rows, _, _ = mo.order_statistics(null, null, null, NINETEEN, list(range(3, 19)), {}, rolling_tests(m, 12), seed=3, permutations=199)
    assert rows.set_index(["question", "measure"]).at[("(1) agreement across releases", 'reaction (stored absorption time)'), "p_value"] > 0.01


