"""The release detector, rank statistics, reaction measures and the wide-quote rule on hand-made inputs and simulated tapes."""

from __future__ import annotations

import numpy as np
import pytest

from information_diffusion.market_making import release_detector as det
from information_diffusion.market_making import reaction_measures as meas
from information_diffusion.market_making import rank_statistics as rk
from information_diffusion.market_making import tape_measures, phases
from information_diffusion.market_making.wide_quote_rule import WideRule
from information_diffusion.market_making.pause_rule import PauseRule
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov

NS = 1_000_000_000
MS = 1_000_000
PARAMS = ASParams(gamma=0.5, sigma=0.02, k=150.0)


def quotes(t, mid, half=0.005, bid_size=100, ask_size=200):
    t = np.asarray(t, np.int64)
    mid = np.asarray(mid, float)
    return phases.valid_quotes(t, mid - half, mid + half, np.full(t.size, bid_size), np.full(t.size, ask_size))


def test_trailing_exits_are_the_directed_trailing_signals():
    rng = np.random.default_rng(3)
    t = np.sort(rng.integers(0, 2_000 * NS, 3_000)).astype(np.int64)
    mid = 100.0 + np.round(np.cumsum(rng.choice([-0.005, 0.0, 0.005], size=t.size)), 3)
    times, d = det.trailing_exits(t, mid, 300.0)
    assert times.tolist() == tape_measures.trailing_signals(t, mid, 300.0).tolist() and times.size > 10
    for x, s in zip(times[:50], d[:50]):
        i = int(np.flatnonzero(t == x)[0])
        before = mid[(t >= x - 300 * NS) & (t < x)]
        assert (s == 1 and mid[i] > before.max()) or (s == -1 and mid[i] < before.min())


def test_coincidences_need_the_same_direction_within_the_window():
    e = lambda times, dirs: (np.array(times, np.int64), np.array(dirs, np.int64))
    exits = {"SPY": e([100 * MS, 900 * MS], [1, -1]), "QQQ": e([105 * MS, 901 * MS], [1, 1]), "IWM": e([400 * MS], [1])}
    assert det.coincidences(exits, 2, 10 * MS).tolist() == [105 * MS]
    assert det.coincidences(exits, 2, 5 * MS).tolist() == [105 * MS]
    assert det.coincidences(exits, 2, 4 * MS).tolist() == []
    assert det.coincidences(exits, 2, 500 * MS).tolist() == [105 * MS, 400 * MS]
    assert det.coincidences(exits, 2, 501 * MS).tolist() == [105 * MS, 400 * MS, 901 * MS]
    assert det.coincidences(exits, 3, 500 * MS).tolist() == [400 * MS]
    same = {"SPY": e([7], [1]), "QQQ": e([7], [1]), "IWM": e([], [])}
    assert det.coincidences(same, 2, 0).tolist() == [7]
    assert det.first_at_or_after(np.array([5, 10, 20], np.int64), 10) == 10 and det.first_at_or_after(np.array([5], np.int64), 6) is None


def test_detector_recovers_a_joint_jump_on_simulated_tapes():
    """Three funds quiet for six minutes, then each jumps up at a known nanosecond (QQQ first, SPY 3 ms later, IWM 40 ms later):
    the two-fund signals fire at SPY's jump (w >= 3 ms) and the three-fund one at IWM's (w >= 40 ms)."""
    rng = np.random.default_rng(26)
    start, t0 = 0, 400 * NS
    jumps = {"QQQ": t0 + 812_345_679, "SPY": t0 + 815_345_679, "IWM": t0 + 852_345_679}
    exits = {}
    for f, j in jumps.items():
        pre = np.sort(rng.integers(start, j, 3_000)).astype(np.int64)
        pre_mid = 100.0 + 0.005 * rng.integers(0, 3, pre.size)
        pre_mid[:2] = (100.0, 100.01)
        post = np.sort(rng.integers(j + 1, j + 3 * NS, 200)).astype(np.int64)
        t = np.concatenate([pre, [j], post])
        mid = np.concatenate([pre_mid, [100.10], np.full(post.size, 100.10)])
        times, d = det.trailing_exits(t, mid, 300.0, start_ns=start + 300 * NS)
        times, d = times[times >= t0], d[times >= t0]
        assert times.tolist() == [j] and d.tolist() == [1]
        exits[f] = (times, d)
    assert det.coincidences(exits, 2, 10 * MS).tolist() == [jumps["SPY"]]
    assert det.coincidences(exits, 2, 1 * MS).tolist() == []
    assert det.coincidences(exits, 3, 500 * MS).tolist() == [jumps["IWM"]]
    delay = det.first_at_or_after(det.coincidences(exits, 2, 100 * MS), t0) - jumps["QQQ"]
    assert delay == 3_000_000


def test_ranks_kendall_w_and_ties():
    assert rk.ranks_with_missing(np.array([3.0, np.nan, 1.0, np.nan])).tolist() == [2.0, 3.5, 1.0, 3.5]
    same = np.tile([1.0, 2.0, 3.0, 4.0], (5, 1))
    assert rk.kendall_w(same) == pytest.approx(1.0)
    opposite = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert rk.kendall_w(opposite) == pytest.approx(0.0)
    r = np.array([[1, 2, 3, 4], [2, 1, 3, 4], [1, 3, 2, 4]], float)
    s = ((r.sum(axis=0) - 3 * 5 / 2) ** 2).sum()
    assert rk.kendall_w(r) == pytest.approx(12 * s / (9 * (64 - 4)))
    tied = np.array([[1.0, 2.5, 2.5, 4.0], [1.0, 2.0, 3.0, 4.0]])
    s = ((tied.sum(axis=0) - 2 * 5 / 2) ** 2).sum()
    assert rk.kendall_w(tied) == pytest.approx(12 * s / (4 * 60 - 2 * 6))


def test_kendall_w_recovery_null_and_signal():
    """Under no common order the mean of W is about 1/m (m releases) and the permutation p rejects at about its level; with
    a strong common order W is near 1 and p is the smallest possible."""
    rng = np.random.default_rng(7)
    m, n, ws, rejections = 10, 8, [], 0
    for _ in range(200):
        r = np.vstack([rng.permutation(n) + 1.0 for _ in range(m)])
        ws.append(rk.kendall_w(r))
        rejections += rk.kendall_w_permutation(r, rng, 199)[1] <= 0.05
    assert np.mean(ws) == pytest.approx(1 / m, abs=0.02)
    assert 0.01 <= rejections / 200 <= 0.10
    effect = np.arange(n, dtype=float)
    r = np.vstack([rk.ranks_with_missing(effect + rng.normal(0, 0.5, n)) for _ in range(m)])
    w, p = rk.kendall_w_permutation(r, rng, 199)
    assert w > 0.8 and p == pytest.approx(1 / 200)


def test_spearman_permutation_and_residuals():
    rng = np.random.default_rng(11)
    x = np.arange(20.0)
    rho, p, n = rk.spearman_permutation(x, x ** 3, rng, 999)
    assert rho == pytest.approx(1.0) and p == pytest.approx(1 / 1000) and n == 20
    rho, p, n = rk.spearman_permutation(np.r_[x, np.nan], np.r_[-x, 5.0], rng, 999)
    assert rho == pytest.approx(-1.0) and n == 20
    rejections = sum(rk.spearman_permutation(rng.normal(size=25), rng.normal(size=25), rng, 199)[1] <= 0.05 for _ in range(300))
    assert 0.02 <= rejections / 300 <= 0.09
    assert np.allclose(rk.rank_residuals(np.arange(1.0, 6.0), np.arange(1.0, 6.0)), 0.0)
    res = rk.rank_residuals(np.array([1.0, 3.0, 2.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    assert res.sum() == pytest.approx(0.0) and rk.spearman_ranks(res, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(0.0, abs=1e-12)
    assert rk.tau(np.array([1, 2, 3]), np.array([10, 20, 30])) == pytest.approx(1.0)
    assert rk.sign_test(np.array([1.0, 2.0, -1.0, 0.0, np.nan])) == (2, 1, pytest.approx(1.0))
    assert rk.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 5.0])) == 3.0
    assert rk.weighted_median(np.array([1.0, 2.0, 3.0]), np.array([5.0, 1.0, 1.0])) == 1.0
    assert np.isnan(rk.weighted_median(np.array([1.0]), np.array([0.0])))


def test_print_cells_and_intervals():
    q = quotes([0, 2 * NS, 10 * NS], [100.00, 100.00, 100.10])
    t = np.array([NS + 5 * MS, NS + 50 * MS, NS + 600 * MS, 3 * NS + 1], np.int64)
    cells = meas.print_cells(t, np.array([100.005, 100.04, 99.995, 100.0]), np.array([100.0, 200.0, 300.0, 50.0]), q, NS, 2 * NS, MS, 5.0)
    assert cells["u_ns"].tolist() == [5 * MS, 50 * MS, 600 * MS]
    assert cells["bucket"].tolist() == [0, 2, 0]
    assert cells["edge"][1] == pytest.approx((100.04 - 100.0) / 100.0 * 1e4)
    assert meas.interval_index(np.array([0, 10 * MS - 1, 10 * MS, 2 * NS, -1]), ((0.0, 10.0), (10.0, 100.0), (100.0, 500.0), (500.0, 1000.0), (1000.0, 2000.0))).tolist() == [0, 0, 1, -1, -1]


def test_time_to_half_move_size_and_peak_spread():
    q = quotes([0, 10 * NS, 10 * NS + 300 * MS, 10 * NS + 900 * MS], [100.00, 100.02, 100.05, 100.08], half=np.array([0.005, 0.02, 0.04, 0.01]))
    base, a = 9 * NS, 10 * NS
    assert meas.time_to_half(q, a, base, a + 5 * NS) == 300 * MS
    assert meas.move_bps(q, base, a + 5 * NS) == pytest.approx(8.0)
    assert np.isnan(meas.time_to_half(quotes([0], [100.0]), a, base, a + 5 * NS))
    assert meas.peak_spread(q, a, a + 5 * NS) == pytest.approx(0.08 / 100.05 * 1e4)
    assert meas.peak_spread(q, a + 100 * MS, a + 200 * MS) == pytest.approx(0.04 / 100.02 * 1e4)
    assert meas.median_update_gap(np.array([0, 5, 15, 30, 100], np.int64), 0, 31) == 10.0


def test_first_tick_move():
    q = quotes([0, 5, 10, 20, 30], [100.000, 100.005, 99.995, 100.015, 100.010])
    assert meas.first_tick_move(q, 10, 1.0) == 10
    assert meas.first_tick_move(q, 6, -1.0) == 4
    assert np.isnan(meas.first_tick_move(q, 10, 0.0)) and np.isnan(meas.first_tick_move(q, 0, 1.0))
    assert np.isnan(meas.first_tick_move(quotes([0, 5], [100.0, 100.005]), 1, 1.0))


def test_wide_rule_reduces_to_rule_two_and_to_the_pause_rule():
    rng = np.random.default_rng(1)
    a = 100 * NS
    t = np.sort(np.concatenate([rng.integers(0, 300 * NS, 4_000), [a - 1, a, a + 1, a + 100 * MS - 1, a + 100 * MS, a + 5 * NS - 1, a + 5 * NS]])).astype(np.int64)
    mid = np.full(t.size, 100.0)
    base = AvellanedaStoikov(PARAMS).quote_terms(t, mid, 300 * NS)
    one = WideRule(PARAMS, (a,), 1.0, 5 * NS).quote_terms(t, mid, 300 * NS)
    for field in ("center", "skew", "half_spread"):
        assert np.array_equal(getattr(one, field), getattr(base, field)), field
    pause = WideRule(PARAMS, (a,), 1.0, 5 * NS, 100 * MS).quote_terms(t, mid, 300 * NS)
    ref = PauseRule(PARAMS, (a,), 100 * MS, 5 * NS).quote_terms(t, mid, 300 * NS)
    assert np.array_equal(pause.half_spread, ref.half_spread)
    wide = WideRule(PARAMS, (a,), 3.0, 5 * NS, 100 * MS).quote_terms(t, mid, 300 * NS)
    span = (t >= a) & (t < a + 100 * MS)
    pulled = (t >= a + 100 * MS) & (t < a + 5 * NS)
    assert np.allclose(wide.half_spread[span], 3.0 * base.half_spread[span])
    assert np.all(wide.half_spread[pulled] > 1e5)
    assert np.array_equal(wide.half_spread[~span & ~pulled], base.half_spread[~span & ~pulled])
    no_pause = WideRule(PARAMS, (a,), 2.0, 5 * NS).quote_terms(t, mid, 300 * NS)
    inside = (t >= a) & (t < a + 5 * NS)
    assert np.allclose(no_pause.half_spread[inside], 2.0 * base.half_spread[inside]) and np.array_equal(no_pause.half_spread[~inside], base.half_spread[~inside])
