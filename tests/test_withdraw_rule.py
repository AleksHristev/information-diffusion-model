"""The rule that withdraws rule 2's quotes for a window around each release."""

from __future__ import annotations

import numpy as np

from information_diffusion.market_making.phase_rules import PhaseRule
from information_diffusion.market_making.withdraw_rule import NS, PullRule, pulled
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov

PARAMS = ASParams(gamma=0.5, sigma=0.02, k=150.0)
ANCHORS = (10 * NS, 400 * NS)


def times() -> np.ndarray:
    rng = np.random.default_rng(1)
    edges = np.array([a + int(u * NS) for a in ANCHORS for u in (-1, 7, 1, 60)])
    t = np.sort(np.concatenate([rng.integers(0, 700 * NS, 5_000), edges, edges - 1, edges + 1]))
    return t.astype(np.int64)


def test_the_seven_second_pull_is_rule_seven_term_for_term():
    t = times()
    mid = np.full(t.size, 100.0)
    mine = PullRule(PARAMS, ANCHORS, 7.0).quote_terms(t, mid, 700 * NS)
    theirs = PhaseRule(PARAMS, "rule7", ANCHORS).quote_terms(t, mid, 700 * NS)
    for field in ("center", "skew", "half_spread"):
        assert np.array_equal(getattr(mine, field), getattr(theirs, field)), field


def test_pull_boundaries_and_rule_two_outside():
    t = times()
    mid = np.full(t.size, 100.0)
    base = AvellanedaStoikov(PARAMS).quote_terms(t, mid, 700 * NS)
    for end in (1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 30.0, 60.0):
        terms = PullRule(PARAMS, ANCHORS, end).quote_terms(t, mid, 700 * NS)
        mask = pulled(t, ANCHORS, -1.0, end)
        assert np.all(terms.half_spread[mask] == 1000000.0)
        assert np.array_equal(terms.half_spread[~mask], base.half_spread[~mask]) and np.array_equal(terms.skew, base.skew)
        a = ANCHORS[0]
        at = np.array([a - NS, a + int(end * NS) - 1, a + int(end * NS), a - NS - 1], dtype=np.int64)
        assert pulled(at, (a,), -1.0, end).tolist() == [True, True, False, False]


