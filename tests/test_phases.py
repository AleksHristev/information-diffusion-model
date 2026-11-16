"""The per-print estimators of the release phases, and the phase-rule schedules and clock ticks, on hand-made quotes and prints."""

from __future__ import annotations

import numpy as np

from information_diffusion.market_making import phases

MS = 1_000_000


def quotes():
    return phases.valid_quotes([0, 10_000 * MS, 20_000 * MS], [100.00, 100.10, 0.0], [100.02, 100.12, 100.0], [500, 300, 1], [400, 200, 1])


def test_invalid_quotes_are_dropped_and_lookups_never_look_ahead():
    q = quotes()
    assert len(q.t) == 2
    assert np.isclose(phases.mid_at(q, 9_999 * MS)[()], 100.01) and np.isclose(phases.mid_at(q, 10_000 * MS)[()], 100.11)
    assert np.isnan(phases.mid_at(q, -1)[()])


def test_side_distance_position():
    q = quotes()
    t = np.array([1_000, 2_000, 3_000, 4_000, 5_000, 6_000]) * MS
    p = np.array([100.02, 100.00, 100.015, 100.01, 100.04, 99.97])
    c = phases.classify(t, p, q, MS)
    assert c.keep.tolist() == [True, True, True, False, True, True]
    assert c.side.tolist() == [1, -1, 1, 1, -1]
    np.testing.assert_allclose(c.distance_bps, np.array([0.01, 0.01, 0.005, 0.03, 0.04]) / 100.01 * 1e4)
    assert c.position.tolist() == [1, 1, 2, 0, 0]


def test_passive_edge_is_distance_plus_markout():
    q = quotes()
    t = np.array([9_000 * MS])
    c = phases.classify(t, np.array([100.02]), q, MS)
    edge = phases.passive_edge(np.array([100.02]), c.side, c.mid, q, t, 5)
    assert np.isclose(edge[0], (100.02 - 100.11) / 100.01 * 1e4)
    assert np.isclose(edge[0], c.distance_bps[0] + (100.01 - 100.11) / 100.01 * 1e4)


def test_buckets_and_phases():
    assert phases.bucket(np.array([0.5, 1.0, 2.9, 3.0, 6.0, 12.0, 25.0, 30.0])).tolist() == [0, 1, 1, 2, 3, 4, 5, 5]
    out = phases.phase_of(np.array([-61, -30, 0.5, 2.0, 6.0, 30.0, 70.0, 300.0]), 1.0)
    assert out.tolist() == ["outside", "before", "between", "phase1", "phase2", "phase2", "later", "outside"]


def test_schedules_and_clock_ticks():
    from information_diffusion.market_making import phase_rules as pr
    from information_diffusion.market_making.market import MarketData

    u = np.array([-1.5, -1.0, 0.0, 6.9, 7.0, 30.0, 60.0, 75.0, 90.0])
    g8 = pr.schedule("rule8", u, 3.0)
    assert np.isinf(g8[[1, 2, 3]]).all() and g8[0] == 1 and g8[4] == 3 and g8[5] == 3 and np.isclose(g8[6], 3) and np.isclose(g8[7], 2) and g8[8] == 1
    assert np.isinf(pr.schedule("rule10_near", u)[[0, 8]]).all() and (pr.schedule("rule10_near", u)[[4, 7]] == 1).all()
    assert (pr.schedule("rule9", u)[[1, 6]] == [3.0, 1.0]).all()
    market = MarketData.from_arrays([0, 5 * NS_PER_S], [10.0, 10.1], [10.02, 10.12], [100, 200], [300, 400], [1 * NS_PER_S], [10.01], [5])
    ticked = pr.with_clock_ticks(market, np.array([-1, 2 * NS_PER_S, 5 * NS_PER_S]))
    assert ticked.quote_t.tolist() == [0, 2 * NS_PER_S, 5 * NS_PER_S]
    assert ticked.bid[1] == market.bid[0] and ticked.ask_size[1] == market.ask_size[0]


NS_PER_S = 1_000_000_000
