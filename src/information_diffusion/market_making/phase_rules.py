"""Quoting rules 7 to 11: rule 2 quotes with the half-spread scheduled, widened or pulled around release times."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

FAR_FACTOR_PRIMARY = 3.0
PULL_FROM_S = -1.0
PULL_TO_S = 7.0
FAR_FROM_S = 7.0
FAR_TO_S = 60.0
DECAY_TO_S = 90.0
RULE9_FROM_S = -1.0
RULE9_TO_S = 60.0
RULE9_FACTOR = 3.0
RULE10_NEAR_FROM_S = 7.0
RULE10_NEAR_TO_S = 90.0
SIGNAL_PULL_S = 5.0
SIGNAL_FAR_S = 55.0
PULL_HALF_SPREAD_DOLLARS = 1000000.0
SIGNAL_DELAY_MS = 5
SIGNAL_WINDOW_S = 1.0

NS = 1_000_000_000
PULL = np.inf


def schedule(kind: str, u: np.ndarray, factor: float = FAR_FACTOR_PRIMARY) -> np.ndarray:
    """Compute the half-spread multiplier g(u) for one anchor, with u = t - a in seconds and inf marking a pulled quote.

    rule7 pulls for -1 <= u < 7. rule8 pulls for -1 <= u < 7, widens by factor for 7 <= u < 60 and decays linearly to 1
    by u = 90. rule9 widens by 3 for -1 <= u < 60 and never pulls. rule10_near rests only for 7 <= u < 90 and is pulled
    elsewhere. rule11, anchored at a signal time, pulls for 0 <= u < 5, widens by factor for 5 <= u < 60 and decays to 1
    by u = 90. Otherwise g = 1.
    Arguments: kind is "rule7", "rule8", "rule9", "rule10_near" or "rule11"; u are the times since the anchor in
    seconds; factor is the widening factor (3 by default).
    Returns: g at each time.
    """
    u = np.asarray(u, dtype=float)
    g = np.ones(u.shape)
    if kind in ("rule7", "rule8"):
        pull = (u >= PULL_FROM_S) & (u < PULL_TO_S)
        if kind == "rule8":
            g[(u >= FAR_FROM_S) & (u < FAR_TO_S)] = factor
            decay = (u >= FAR_TO_S) & (u < DECAY_TO_S)
            g[decay] = factor - (factor - 1.0) * (u[decay] - FAR_TO_S) / (DECAY_TO_S - FAR_TO_S)
        g[pull] = PULL
    elif kind == "rule9":
        g[(u >= RULE9_FROM_S) & (u < RULE9_TO_S)] = RULE9_FACTOR
    elif kind == "rule10_near":
        active = (u >= RULE10_NEAR_FROM_S) & (u < RULE10_NEAR_TO_S)
        g[~active] = PULL
    elif kind == "rule11":
        far_end = SIGNAL_PULL_S + SIGNAL_FAR_S
        g[(u >= SIGNAL_PULL_S) & (u < far_end)] = factor
        decay = (u >= far_end) & (u < far_end + (DECAY_TO_S - FAR_TO_S))
        g[decay] = factor - (factor - 1.0) * (u[decay] - far_end) / (DECAY_TO_S - FAR_TO_S)
        g[(u >= 0) & (u < SIGNAL_PULL_S)] = PULL
    else:
        raise ValueError(kind)
    return g


def combined(kind: str, t_ns: np.ndarray, anchors_ns: tuple[int, ...], factor: float) -> np.ndarray:
    """Combine the schedule over several anchors: pulled if any anchor pulls, otherwise the largest multiplier.

    For rule10_near the order rests if any anchor's window is active; with no anchors it is always pulled, and the
    other kinds give g = 1.
    Arguments: kind is as in schedule; t_ns are the times in nanoseconds; anchors_ns are the anchor times; factor is
    the widening factor.
    Returns: g at each time.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    if not anchors_ns:
        return np.ones(len(t_ns)) if kind != "rule10_near" else np.full(len(t_ns), PULL)
    each = np.stack([schedule(kind, (t_ns - a) / NS, factor) for a in anchors_ns])
    if kind == "rule10_near":
        return np.where(np.isfinite(each).any(axis=0), 1.0, PULL)
    pulled = np.isinf(each).any(axis=0)
    out = np.where(np.isinf(each), 1.0, each).max(axis=0)
    out[pulled] = PULL
    return out


@dataclass(frozen=True)
class PhaseRule:
    """Rule 2 quotes with the half-spread multiplied by a schedule around anchor times.

    A pulled quote is placed PULL_HALF_SPREAD_DOLLARS from the centre, a price no trade reaches.
    Fields: params, the ASParams; kind, one of rule7, rule8, rule9, rule10_near and rule11; anchors_ns, the anchor
    times in nanoseconds; factor, the widening factor; label, a free label.
    Reference: Avellaneda and Stoikov (2008), Quantitative Finance, for the base quotes.
    """

    params: ASParams
    kind: str
    anchors_ns: tuple[int, ...]
    factor: float = FAR_FACTOR_PRIMARY
    label: str = ""
    name: ClassVar[str] = "phase_rule"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the scheduled half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        g = combined(self.kind, t_ns, self.anchors_ns, self.factor)
        half = np.where(np.isinf(g), PULL_HALF_SPREAD_DOLLARS, half_spread * np.where(np.isinf(g), 1.0, g))
        return QuoteTerms(mid, skew, half)


def signal_values(grid_ns: np.ndarray, leaders: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Compute the leaders' signal x(t) = max_j |log m_j(t - 5 ms) - log m_j(t - 5 ms - 1 s)| on a time grid.

    Arguments: grid_ns are the grid times in nanoseconds; leaders is a list of (times, mid) pairs, one per leader.
    Returns: x at each grid time, ignoring leaders without a quote yet (NaN when none has one).
    """
    delay, window = SIGNAL_DELAY_MS * 1_000_000, int(SIGNAL_WINDOW_S * NS)
    values = []
    for t, mid in leaders:
        now = np.searchsorted(t, grid_ns - delay, side="right") - 1
        then = np.searchsorted(t, grid_ns - delay - window, side="right") - 1
        ok = (now >= 0) & (then >= 0)
        r = np.full(grid_ns.size, np.nan)
        r[ok] = np.abs(np.log(mid[now[ok]] / mid[then[ok]]))
        values.append(r)
    return np.nanmax(np.vstack(values), axis=0) if values else np.full(grid_ns.size, np.nan)


def signal_time(grid_ns: np.ndarray, values: np.ndarray, threshold: float) -> int | None:
    """Return the first grid time in nanoseconds at which values exceed threshold, or None if they never do."""
    above = np.flatnonzero(np.nan_to_num(values, nan=-np.inf) > threshold)
    return int(grid_ns[above[0]]) if above.size else None


def clock_ticks(anchors_ns, signal_ns) -> np.ndarray:
    """List the schedule boundaries of rules 6 to 11 around the anchors and signal times.

    The boundaries include every second through the 60 to 90 s decays and every 10 s from 120 to 600 s.
    Arguments: anchors_ns are the scheduled anchor times and signal_ns the signal times, in nanoseconds.
    Returns: the sorted unique boundary times in nanoseconds.
    """
    out = []
    for a in anchors_ns:
        out += [a + int(u * NS) for u in (-60, PULL_FROM_S, 0, PULL_TO_S, FAR_TO_S, DECAY_TO_S, 120, 600)]
        out += [a + int(u * NS) for u in np.arange(FAR_TO_S, DECAY_TO_S + 1, 1.0)]
        out += [a + int(u * NS) for u in np.arange(120, 601, 10.0)]
    for s in signal_ns:
        far_end = SIGNAL_PULL_S + SIGNAL_FAR_S
        out += [s + int(u * NS) for u in (0, SIGNAL_PULL_S, far_end)]
        out += [s + int(u * NS) for u in np.arange(far_end, far_end + (DECAY_TO_S - FAR_TO_S) + 1, 1.0)]
    return np.unique(np.asarray(out, dtype=np.int64))


def with_clock_ticks(market, ticks_ns: np.ndarray):
    """Insert a copy of the prevailing NBBO at each clock tick, so schedule boundaries become decision points.

    No copy is added before the first update or where an update already exists; copies change no price, size or queue.
    Arguments: market is the MarketData; ticks_ns are the tick times in nanoseconds.
    Returns: the augmented MarketData.
    """
    q = market.quote_t
    ticks = np.asarray(ticks_ns, dtype=np.int64)
    k = np.searchsorted(q, ticks, side="right") - 1
    new = (k >= 0) & ~np.isin(ticks, q)
    ticks, k = ticks[new], k[new]
    if ticks.size == 0:
        return market
    position = np.searchsorted(q, ticks, side="right")
    insert = lambda array: np.insert(array, position, array[k])
    return dataclasses.replace(market, quote_t=np.insert(q, position, ticks), bid=insert(market.bid), ask=insert(market.ask), bid_size=insert(market.bid_size), ask_size=insert(market.ask_size))
