"""Measures of the first seconds at the full resolution of the tape, and the pause rule, on hand-made and simulated tapes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from information_diffusion.epidemic import absorption
from information_diffusion.market_making import tape_measures as tape
from information_diffusion.market_making import phases
from information_diffusion.market_making.pause_rule import PauseRule, paused
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov

NS = 1_000_000_000
PARAMS = ASParams(gamma=0.5, sigma=0.02, k=150.0)


def make_quotes(t, bid, ask, bid_size=None, ask_size=None) -> phases.Quotes:
    n = len(t)
    return phases.valid_quotes(np.asarray(t, np.int64), np.asarray(bid, float), np.asarray(ask, float), np.full(n, 100) if bid_size is None else bid_size, np.full(n, 200) if ask_size is None else ask_size)


def test_anchor_is_integer_nanoseconds():
    t0 = 1_757_075_400 * NS
    assert tape.anchor_ns(t0, 0.779645696) == t0 + 779_645_696
    assert tape.anchor_ns(t0, 0.013149561) == t0 + 13_149_561
    for ns in (1, 255, 256, 999_999_999, 2_070_000_001):
        assert tape.anchor_ns(t0, ns / 1e9) == t0 + ns


def test_quote_in_force_is_inclusive_and_nan_before_the_first():
    q = make_quotes([100, 200, 300], [10.00, 10.01, 10.02], [10.02, 10.03, 10.04])
    v = tape.quote_values(q, [99, 100, 199, 200, 1000])
    assert np.isnan(v["mid"][0]) and v["mid"][1] == pytest.approx(10.01) and v["mid"][2] == pytest.approx(10.01) and v["mid"][3] == pytest.approx(10.02)
    assert tape.spread_bps(v)[1] == pytest.approx(0.02 / 10.01 * 1e4)
    assert tape.displayed_size(v)[1] == 300


def test_share_of_move_and_the_two_tick_rule():
    assert tape.share_of_move(100.00, 100.03, 100.04).item() == pytest.approx(0.75)
    assert tape.share_of_move(100.00, 99.99, 99.96).item() == pytest.approx(0.25)
    assert np.isnan(tape.share_of_move(100.00, 100.01, 100.015))
    assert tape.share_of_move(100.00, 100.02, 100.02).item() == pytest.approx(1.0)
    assert not tape.moved_enough(100.0, 100.019) and tape.moved_enough(100.0, 100.02)


def test_directions_use_rounded_midpoints():
    assert tape.own_direction(np.array([100.005, 100.0, np.nan]), np.array([100.0, 100.0 + 1e-12, 100.0])).tolist() == [1.0, 0.0, 0.0]
    m_at = np.array([[101.0, 99.0], [50.0, 50.0], [np.nan, np.nan]])
    m_base = np.array([[100.0, 100.0], [50.5, 49.0], [10.0, 10.0]])
    d = tape.index_direction(m_at, m_base)
    assert d.tolist() == [np.sign(np.log(1.01) + np.log(50 / 50.5)), np.sign(np.log(0.99) + np.log(50 / 49))]
    assert tape.index_direction(np.full((3, 1), np.nan), np.full((3, 1), 1.0)).tolist() == [0.0]


def test_taker_edge_crosses_at_the_quote_in_force_and_marks_later():
    q = make_quotes([0, 5 * NS, 60 * NS], [100.00, 100.10, 100.20], [100.02, 100.12, 100.22], np.array([300, 400, 500]), np.array([700, 800, 900]))
    edge, size = tape.taker_edge(q, np.array([0, 0, 0]), np.array([1, -1, 0]), 5.0)
    assert edge[0] == pytest.approx((100.11 - 100.02) / 100.01 * 1e4) and size[0] == 700
    assert edge[1] == pytest.approx(-(100.11 - 100.00) / 100.01 * 1e4) and size[1] == 300
    assert np.isnan(edge[2]) and size[2] == 0
    edge60, _ = tape.taker_edge(q, np.array([0]), np.array([1]), 60.0)
    assert edge60[0] == pytest.approx((100.21 - 100.02) / 100.01 * 1e4)
    e, _ = tape.taker_edge(q, np.array([5 * NS - 1]), np.array([1]), 5.0)
    assert e[0] == pytest.approx((100.11 - 100.02) / 100.01 * 1e4)


def test_counts_and_sums_are_half_open():
    t = np.array([0, 10, 10, 20, 30], dtype=np.int64)
    assert tape.counts_in(t, [0, 10, 11], [10, 20, 31]).tolist() == [1, 2, 2]
    assert tape.sums_in(t, np.array([1, 2, 3, 4, 5]), [0, 10], [10, 31]).tolist() == [1.0, 14.0]


def test_trailing_signals_match_a_brute_force_scan():
    rng = np.random.default_rng(3)
    t = np.sort(rng.integers(0, 2_000 * NS, 3_000)).astype(np.int64)
    mid = 100.0 + np.round(np.cumsum(rng.choice([-0.005, 0.0, 0.005], size=t.size)), 3)
    w = 300.0
    got = tape.trailing_signals(t, mid, w)
    want = []
    for i in range(t.size):
        if t[i] < t[0] + int(w * NS):
            continue
        before = (t >= t[i] - int(w * NS)) & (t < t[i])
        if before.any() and (round(mid[i] - mid[before].max(), 9) > 0 or round(mid[i] - mid[before].min(), 9) < 0):
            want.append(t[i])
    assert got.tolist() == want and len(want) > 10


def test_trailing_signal_reproduces_the_absorption_time_when_the_range_is_fixed():
    t0 = 1_000 * NS
    t = np.concatenate([np.arange(t0 - 299 * NS, t0, NS), [t0 + 700_123_456, t0 + 900_000_000]]).astype(np.int64)
    mid = np.concatenate([100.0 + 0.01 * (np.arange(299) % 3), [100.05, 100.06]])
    first = tape.trailing_signals(t, mid, 300.0, start_ns=t0)[0]
    tau = absorption.absorption_time(t, mid + 0.005, mid - 0.005, t0)["absorption_seconds"]
    assert first == tape.anchor_ns(t0, tau) == t0 + 700_123_456


def test_episodes_join_overlapping_pauses():
    s = np.array([0, 2 * NS, 10 * NS, 15 * NS], dtype=np.int64)
    starts, ends = tape.episodes(s, 0, 5 * NS)
    assert starts.tolist() == [0, 10 * NS] and ends.tolist() == [7 * NS, 20 * NS]
    starts, ends = tape.episodes(s, 1 * NS, 5 * NS)
    assert starts.tolist() == [1 * NS, 11 * NS, 16 * NS] and ends.tolist() == [7 * NS, 15 * NS, 20 * NS]
    starts, _ = tape.episodes(np.array([0, NS // 2], np.int64), 1 * NS, 5 * NS)
    assert starts.tolist() == [1 * NS]
    assert tape.episodes(np.zeros(0, np.int64), 0, 1)[0].size == 0


def test_nbbo_events_attribute_each_changed_side():
    t = np.array([0, 100, 100, 250, 1_000], dtype=np.int64)
    bid = np.array([10.00, 10.00, 10.01, 10.01, 10.01])
    bid_size = np.array([100, 100, 100, 100, 200])
    bid_ex = np.array(["P", "P", "Q", "Q", "Q"], dtype=object)
    ask = np.array([10.02, 10.03, 10.03, 10.02, 10.02])
    ask_size = np.array([100, 100, 100, 100, 100])
    ask_ex = np.array(["Z", "Z", "Z", "P", "P"], dtype=object)
    ev = tape.nbbo_events(t, bid, bid_size, bid_ex, ask, ask_size, ask_ex)
    assert list(zip(ev.row, ev.exchange)) == [(0, "P"), (0, "Z"), (1, "Z"), (2, "Q"), (3, "P"), (4, "Q")]
    within = np.sort(tape.gaps_within(ev.t.to_numpy(), ev.exchange.to_numpy()))
    assert within.tolist() == [100, 250, 900]
    across = tape.gaps_across_rows(ev.t.to_numpy(), ev.row.to_numpy(), ev.exchange.to_numpy())
    assert across.tolist() == [0, 150, 750]


def test_print_gaps_and_shares():
    t = np.array([0, 0, 500, 2_000, 20_000, 2_000_000], dtype=np.int64)
    x = np.array(["P", "Q", "P", "P", "Q", "Q"], dtype=object)
    assert np.sort(tape.gaps_within(t, x)).tolist() == [500, 1_500, 20_000, 1_980_000]
    assert tape.gaps_across(t, x).tolist() == [0, 500, 18_000]
    s = tape.gap_shares(np.array([0, 999, 1_000, 9_999, 99_999, 999_999, 1_000_000]))
    assert s == {"gaps": 7, "zero": 1, "below_1000ns": 2, "below_10000ns": 4, "below_100000ns": 5, "below_1000000ns": 6}


def test_shift_draws_bounds_and_integers():
    rng = np.random.default_rng(7)
    e = tape.shift_draws(rng, np.array([True, False, True]), 4_000)
    assert e.dtype == np.int64 and e.shape == (4_000, 3)
    assert np.abs(e[:, [0, 2]]).max() <= 500_000 and np.abs(e[:, 1]).max() <= 2_500_000
    assert np.abs(e[:, 1]).max() > 2_400_000 and np.abs(e[:, 0]).max() > 480_000
    assert abs(e[:, 1].mean()) < 60_000


def test_stable_from_needs_every_longer_offset():
    o = [0.0, 1.0, 5.0, 10.0]
    assert tape.stable_from(o, [3.0, 0.1, 2.0, 0.1], [1.0, 1.0, 1.0, 1.0]) == 10.0
    assert tape.stable_from(o, [3.0, 0.4, 0.5, 0.1], [1.0, 1.0, 1.0, 1.0]) == 1.0
    assert tape.stable_from(o, [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]) == 0.0
    assert np.isnan(tape.stable_from(o, [0.0, 0.0, 0.0, 0.9], [1.0, 1.0, 1.0, 1.0]))
    assert np.isnan(tape.stable_from(o, [0.0, 0.0, 0.0, np.nan], [1.0, 1.0, 1.0, 1.0]))


def test_release_interval_covers_the_mean():
    rng = np.random.default_rng(11)
    hits = 0
    for _ in range(200):
        x = rng.normal(1.0, 1.0, size=(40, 3))
        lo, hi = tape.release_interval(x, lambda v: float(np.nanmean(v)), rng, resamples=400)
        hits += lo <= 1.0 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def times() -> np.ndarray:
    rng = np.random.default_rng(1)
    a = 100 * NS
    edges = np.array([a + o for o in (-NS, 0, 10_000, 1_000_000, 5 * NS, 7 * NS)], dtype=np.int64)
    return np.sort(np.concatenate([rng.integers(0, 300 * NS, 5_000), edges, edges - 1, edges + 1])).astype(np.int64)


def test_pause_boundaries_are_exact_nanoseconds_and_rule_two_outside():
    t = times()
    mid = np.full(t.size, 100.0)
    a = 100 * NS
    base = AvellanedaStoikov(PARAMS).quote_terms(t, mid, 300 * NS)
    for o in tape.offsets_ns():
        terms = PauseRule(PARAMS, (a,), int(o), 5 * NS).quote_terms(t, mid, 300 * NS)
        mask = paused(t, (a,), int(o), 5 * NS)
        assert np.array_equal(mask, (t >= a + o) & (t < a + 5 * NS))
        assert np.all(terms.half_spread[mask] > 1e5) and np.array_equal(terms.half_spread[~mask], base.half_spread[~mask])
    assert paused(np.array([a + 9_999, a + 10_000], np.int64), (a,), 10_000, 5 * NS).tolist() == [False, True]


def test_paused_with_many_anchors_is_the_union():
    rng = np.random.default_rng(5)
    anchors = np.sort(rng.integers(0, 100 * NS, 40)).astype(np.int64)
    t = np.sort(rng.integers(0, 110 * NS, 20_000)).astype(np.int64)
    for lo, hi in ((0, 5 * NS), (250_000, 5 * NS), (-NS, 5 * NS)):
        want = np.zeros(t.size, dtype=bool)
        for x in anchors:
            want |= (t >= x + lo) & (t < x + hi)
        assert np.array_equal(paused(t, tuple(int(x) for x in anchors), lo, hi), want)
        starts, ends = tape.episodes(anchors, lo, hi)
        inside = np.zeros(t.size, dtype=bool)
        for s, e in zip(starts, ends):
            inside |= (t >= s) & (t < e)
        assert np.array_equal(inside, want)


def test_recovery_of_a_known_reaction_on_a_simulated_tape():
    """A tape with a jump of known size at a known nanosecond: the reaction, the share of the move, the taker's edge and the
    shifted reaction come back exactly."""
    rng = np.random.default_rng(25)
    t0 = 5_000 * NS
    react = t0 + 812_345_679
    pre = np.sort(rng.integers(t0 - 300 * NS, t0, 4_000)).astype(np.int64)
    post = np.sort(rng.integers(react + 1, react + 70 * NS, 4_000)).astype(np.int64)
    t = np.concatenate([pre, [react], post])
    mid = np.concatenate([100.0 + 0.005 * rng.integers(-2, 3, pre.size), [100.10], 100.10 + 0.005 * np.minimum(np.arange(post.size) // 400, 6)])
    q = make_quotes(t, mid - 0.005, mid + 0.005)
    tau = absorption.absorption_time(t, mid + 0.005, mid - 0.005, t0)["absorption_seconds"]
    t_l = tape.anchor_ns(t0, tau)
    assert t_l == react
    base = tape.quote_values(q, [t0 - NS])["mid"][0]
    end = tape.quote_values(q, [t_l + 5 * NS])["mid"][0]
    at = tape.quote_values(q, t_l + tape.offsets_ns())["mid"]
    share = tape.share_of_move(np.full(at.size, base), at, np.full(at.size, end))
    assert share[0] == pytest.approx((100.10 - base) / (end - base))
    before = tape.quote_values(q, [t_l - 1])["mid"]
    assert tape.share_of_move(base, before, end)[0] == pytest.approx((before[0] - base) / (end - base)) and abs(before[0] - base) < 0.02
    d = tape.own_direction(at, np.full(at.size, base))
    assert np.all(d == 1.0)
    edge, size = tape.taker_edge(q, t_l + tape.offsets_ns(), d, 5.0)
    want = (tape.quote_values(q, t_l + tape.offsets_ns() + 5 * NS)["mid"] - (at + 0.005)) / at * 1e4
    assert np.allclose(edge, want) and np.all(size == 200)
    shifted = tape.quote_values(q, t_l - 400_000 + tape.offsets_ns())["mid"]
    assert np.all(shifted[:5] < 100.05) and np.all(shifted[6:] >= 100.10 - 1e-9)


def test_gap_shares_on_a_quantised_clock():
    """On a clock in multiples of 256 ns no non-zero gap is below 256 ns, so the share below 1 microsecond is coarse."""
    rng = np.random.default_rng(9)
    true = np.sort(rng.integers(0, 2 * NS, 50_000)).astype(np.int64)
    coarse = (true // 256) * 256
    g = np.diff(coarse)
    assert np.all((g == 0) | (g >= 256))
    assert tape.gap_shares(np.diff(true))["zero"] < tape.gap_shares(g)["zero"]
    frame = pd.DataFrame({"t": coarse})
    assert frame.t.mod(256).eq(0).all()
    assert tape.quantised(coarse) and not tape.quantised(true) and not tape.quantised(np.zeros(0, np.int64))
    mixed = np.concatenate([coarse[:990], true[990:1000] | 1])
    assert tape.quantised(mixed) and not tape.quantised(np.concatenate([coarse[:980], true[980:1000] | 1]))
