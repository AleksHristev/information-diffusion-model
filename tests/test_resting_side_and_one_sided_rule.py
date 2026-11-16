"""Resting side, further moves, direction series, inventory statistics and one-sided quoting, on hand-made inputs and a simulated tape."""

from __future__ import annotations

import numpy as np
import pytest

from information_diffusion.market_making import resting_side as side
from information_diffusion.market_making import phases
from information_diffusion.market_making.one_sided_rule import OneSidedRule, inside
from information_diffusion.market_making.pause_rule import PauseRule
from information_diffusion.market_making.quote_terms import quote_ticks
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov

NS = 1_000_000_000
MS = 1_000_000
PARAMS = ASParams(gamma=0.5, sigma=0.02, k=150.0)


def quotes(t, mid, half=0.005):
    t = np.asarray(t, np.int64)
    mid = np.asarray(mid, float)
    return phases.valid_quotes(t, mid - half, mid + half, np.full(t.size, 100), np.full(t.size, 100))


def test_further_move_and_sides():
    q = quotes([0, 10 * NS, 15 * NS], [100.00, 100.02, 99.99])
    x = side.further_move(q, np.array([10 * NS, 10 * NS, 10 * NS]), np.array([1, -1, 0]), 5.0)
    assert x[0] == pytest.approx((99.99 - 100.02) / 100.02 * 1e4) and x[1] == pytest.approx(-x[0]) and np.isnan(x[2])
    r = side.resting_side(np.array([1, -1, 1]))
    assert r.tolist() == [-1, 1, -1]
    assert side.side_of_move(r, np.array([1, 1, 0])).tolist() == [-1, 1, 0]


def test_direction_series_with_a_fixed_and_a_moving_base():
    q = quotes([0, 5 * NS, 9 * NS, 12 * NS, 20 * NS], [100.00, 100.00, 100.01, 99.99, 100.02])
    t0 = 10 * NS
    times, d = side.own_direction_series(q, np.array([0]), np.array([t0 - NS]), 0, 30 * NS)
    lookup = dict(zip(times.tolist(), d.tolist()))
    assert lookup[12 * NS] == -1 and lookup[20 * NS] == 1 and lookup[9 * NS] == 0
    times, d = side.own_direction_series(q, np.array([13 * NS, 19 * NS]), np.array([12 * NS, 18 * NS]), 0, 30 * NS)
    lookup = dict(zip(times.tolist(), d.tolist()))
    assert lookup[13 * NS] == 0 and lookup[20 * NS] == 1 and lookup[5 * NS] == 0
    qs = [quotes([0, 11 * NS], [100.0, 100.10]), quotes([0, 11 * NS], [50.0, 49.98]), quotes([0], [20.0])]
    times, d = side.index_direction_series(qs, np.array([0]), np.array([t0 - NS]), 0, 30 * NS)
    assert dict(zip(times.tolist(), d.tolist()))[11 * NS] == np.sign(np.log(100.10 / 100.0) + np.log(49.98 / 50.0))


def test_inventory_stats():
    t = np.array([2, 4, 8], np.int64)
    level = np.array([3.0, -1.0, 0.0])
    mean_abs, var = side.inventory_stats(t, level, 0, 10)
    path = np.array([0, 0, 3, 3, -1, -1, -1, -1, 0, 0], float)
    assert mean_abs == pytest.approx(np.abs(path).mean()) and var == pytest.approx(path.var())
    mean_abs, _ = side.inventory_stats(t, level, 3, 6)
    assert mean_abs == pytest.approx((3 * 1 + 1 * 2) / 3)


def test_inside_matches_brute_force():
    rng = np.random.default_rng(4)
    s = rng.integers(0, 1_000, 30)
    e = s + rng.integers(1, 200, 30)
    t = np.arange(0, 1_300)
    want = np.zeros(t.size, bool)
    for a, b in zip(s, e):
        want |= (t >= a) & (t < b)
    assert np.array_equal(inside(t, s, e), want)
    assert not inside(np.array([5]), np.zeros(0, np.int64), np.zeros(0, np.int64)).any()


def times():
    rng = np.random.default_rng(1)
    return np.sort(rng.integers(0, 300 * NS, 3_000)).astype(np.int64)


def test_one_sided_keeps_rule_two_side_exactly_and_withdraws_the_other():
    t = times()
    rng = np.random.default_rng(2)
    mid = 100.0 + 0.005 * rng.integers(-400, 400, t.size)
    a = 100 * NS
    dt = np.array([0, a + 2 * NS, a + 4 * NS], np.int64)
    dv = np.array([1.0, -1.0, 0.0])
    rule = OneSidedRule(PARAMS, np.array([a]), np.array([a + 5 * NS]), dt, dv)
    mine = rule.quote_terms(t, mid, 300 * NS)
    base = AvellanedaStoikov(PARAMS).quote_terms(t, mid, 300 * NS)
    up = (t >= a) & (t < a + 2 * NS)
    down = (t >= a + 2 * NS) & (t < a + 4 * NS)
    rest = ~(up | down)
    tick = 0.01
    for q in range(-20, 21):
        b0, a0 = quote_ticks(base, q, tick)
        b1, a1 = quote_ticks(mine, q, tick)
        assert np.array_equal(b1[up], b0[up]) and np.all(a1[up] - b0[up] >= int(2 * 1000.0 / tick) - 1)
        assert np.array_equal(a1[down], a0[down]) and np.all(a0[down] - b1[down] >= int(2 * 1000.0 / tick) - 1)
        assert np.array_equal(b1[rest], b0[rest]) and np.array_equal(a1[rest], a0[rest])


def test_one_sided_pause_equals_the_pause_rule_and_takes_precedence():
    t = times()
    mid = np.full(t.size, 100.0)
    a = 100 * NS
    rule = OneSidedRule(PARAMS, np.array([a]), np.array([a + 60 * NS]), np.array([0], np.int64), np.array([1.0]), np.array([a]), np.array([a + 5 * NS]))
    mine = rule.quote_terms(t, mid, 300 * NS)
    ref = PauseRule(PARAMS, (a,), 0, 5 * NS).quote_terms(t, mid, 300 * NS)
    paused = (t >= a) & (t < a + 5 * NS)
    assert np.array_equal(mine.half_spread[paused], ref.half_spread[paused]) and np.array_equal(mine.center[paused], ref.center[paused])
    one = (t >= a + 5 * NS) & (t < a + 60 * NS)
    assert np.all(mine.half_spread[one] == 1000.0)
    out = ~(paused | one)
    assert np.array_equal(mine.half_spread[out], ref.half_spread[out]) and np.array_equal(mine.center[out], ref.center[out])


def test_persistence_on_a_simulated_drift():
    """A fund that keeps drifting up after the reaction: at every offset the direction is up and the further move over the
    next 5 s is up, so the share of the same sign is one and the mean further move is the drift."""
    t = np.arange(0, 120 * NS, 100 * MS, dtype=np.int64)
    t0, react = 60 * NS, 61 * NS
    mid = np.where(t < react, 100.0, 100.0 + 0.01 * ((t - react) // (500 * MS) + 1))
    q = quotes(t, mid)
    at = react + np.array([int(o * MS) for o in (0.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0)])
    times_, d = side.own_direction_series(q, np.array([0]), np.array([t0 - NS]), 0, 120 * NS)
    from information_diffusion.market_making.quote_terms import as_of
    dirs = as_of(times_, d, at, 0.0)
    assert np.all(dirs == 1)
    x = side.further_move(q, at, dirs, 5.0)
    assert np.all(x > 0) and x[0] == pytest.approx(0.10 / 100.01 * 1e4, rel=1e-6)
