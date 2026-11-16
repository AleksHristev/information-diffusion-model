"""Classification of prints against the NBBO and the passive edge earned by resting orders around releases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


TICK = 0.01
DISTANCE_EDGES_BPS = (1.0, 3.0, 6.0, 12.0, 25.0)
BEFORE_S = 60.0
PHASE1_END_S = 5.0
PHASE2_END_S = 60.0
LATER_END_S = 240.0


@dataclass(frozen=True)
class Quotes:
    """Valid NBBO updates (0 < bid < ask) in time order.

    Fields: t in nanoseconds, bid and ask in dollars, bid_size and ask_size in shares.
    """

    t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        """Return the NBBO midpoint in dollars."""
        return (self.bid + self.ask) / 2.0


def valid_quotes(t, bid, ask, bid_size, ask_size) -> Quotes:
    """Keep the valid NBBO updates, those with finite 0 < bid < ask.

    Arguments: t are the update times in nanoseconds; bid, ask, bid_size and ask_size are the quotes.
    Returns: a Quotes; raises if the kept times are not non-decreasing.
    """
    t, bid, ask = np.asarray(t, dtype=np.int64), np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    keep = (bid > 0) & (ask > bid) & np.isfinite(bid) & np.isfinite(ask)
    if np.any(np.diff(t[keep]) < 0):
        raise ValueError("quote times must be non-decreasing")
    return Quotes(t[keep], bid[keep], ask[keep], np.asarray(bid_size)[keep], np.asarray(ask_size)[keep])


def last_index(quote_t: np.ndarray, at_ns) -> np.ndarray:
    """Return the index of the last quote at or before each time in at_ns, or -1 if none, given quote times quote_t."""
    return np.searchsorted(quote_t, np.asarray(at_ns, dtype=np.int64), side="right") - 1


def mid_at(q: Quotes, at_ns) -> np.ndarray:
    """Return the midpoint of the last valid quote in q at or before each time in at_ns, NaN where none exists."""
    k = last_index(q.t, at_ns)
    if len(q.t) == 0:
        return np.full(np.shape(k), np.nan)
    return np.where(k >= 0, q.mid[np.maximum(k, 0)], np.nan)


@dataclass(frozen=True)
class Classified:
    """Classified prints, after dropping prints with no reference quote and prints at the mid.

    Fields: keep is the mask over all prints; side (+1 buy, -1 sell), reference mid in dollars, distance from the mid in
    basis points and position (0 beyond the best quote, 1 at it, 2 inside the spread) are given for the kept prints.
    """

    keep: np.ndarray
    side: np.ndarray
    mid: np.ndarray
    distance_bps: np.ndarray
    position: np.ndarray


def classify(print_t, price, q: Quotes, lag_ns: int, tick: float = TICK) -> Classified:
    """Classify prints against the last valid NBBO at or before t - lag.

    The side is +1 if p >= ask, -1 if p <= bid and otherwise the sign of p - m (prints at the mid are dropped); the
    distance is s (p - m) / m in basis points; the position compares s (p - touch) with half a tick. Prices are compared
    after rounding to 1e-9 dollars.
    Arguments: print_t are the print times in nanoseconds; price are the print prices in dollars; q are the valid
    quotes; lag_ns is the quote lag in nanoseconds; tick is the tick size in dollars.
    Returns: a Classified.
    Reference: Lee and Ready (1991).
    """
    print_t, price = np.asarray(print_t, dtype=np.int64), np.asarray(price, dtype=float)
    k = last_index(q.t, print_t - lag_ns)
    has = k >= 0
    kk = np.maximum(k, 0)
    bid, ask = q.bid[kk], q.ask[kk]
    m = (bid + ask) / 2.0
    above_mid = np.round(price - m, 9)
    side = np.where(np.round(price - ask, 9) >= 0, 1, np.where(np.round(price - bid, 9) <= 0, -1, np.sign(above_mid))).astype(int)
    keep = has & (side != 0)
    touch = np.where(side > 0, ask, bid)
    past = np.round(side * (price - touch), 9)
    half = round(tick / 2, 9)
    position = np.where(past > half, 0, np.where(past <= -half, 2, 1))
    distance = side * (price - m) / m * 1e4
    return Classified(keep, side[keep], m[keep], distance[keep], position[keep])


def passive_edge(price, side, m, q: Quotes, at_ns, horizon_s: float) -> np.ndarray:
    """Compute the passive edge e_H = s (p - m(t + H)) / m * 1e4 of each print, in basis points.

    This is the distance from the mid plus the passive side's mark-out at horizon H.
    Arguments: price, side and m are the print price, side and reference mid; q are the valid quotes; at_ns are the
    print times in nanoseconds; horizon_s is H in seconds.
    Returns: the passive edge, NaN where no quote exists at t + H.
    Reference: Hasbrouck (2007), Empirical Market Microstructure.
    """
    future = mid_at(q, np.asarray(at_ns, dtype=np.int64) + int(round(horizon_s * 1e9)))
    return np.asarray(side) * (np.asarray(price) - future) / np.asarray(m) * 1e4


def bucket(distance_bps: np.ndarray) -> np.ndarray:
    """Return the distance bucket 0 to 5 of each distance in basis points: below 1, 1-3, 3-6, 6-12, 12-25, above 25."""
    return np.searchsorted(np.asarray(DISTANCE_EDGES_BPS), np.asarray(distance_bps), side="right")


def phase_of(seconds_since_t0: np.ndarray, t_l: float) -> np.ndarray:
    """Assign each time to a phase around a release at t0 that the market reacts to at a later time tL.

    Phases: "before" (the 60 s before t0), "between" (t0 to tL), "phase1" (the first 5 s after tL), "phase2" (5 to
    60 s after tL), "later" (60 to 240 s after tL) and "outside" (any other time).
    Arguments: seconds_since_t0 are the times since t0 in seconds; t_l is the reaction time tL in seconds since t0.
    Returns: an object array of phase names.
    """
    u = np.asarray(seconds_since_t0, dtype=float)
    v = u - t_l
    out = np.full(u.shape, "outside", dtype=object)
    out[(u >= -BEFORE_S) & (u < 0)] = "before"
    out[(u >= 0) & (v < 0)] = "between"
    out[(v >= 0) & (v < PHASE1_END_S)] = "phase1"
    out[(v >= PHASE1_END_S) & (v < PHASE2_END_S)] = "phase2"
    out[(v >= PHASE2_END_S) & (v < LATER_END_S)] = "later"
    return out
