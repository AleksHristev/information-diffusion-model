"""Per-release reaction measures: print cells, time to half of the move, move size, peak spread, update gap and first tick move."""

from __future__ import annotations

import numpy as np

from information_diffusion.market_making import tape_measures, phases
from information_diffusion.market_making.phases import Quotes

TICK = 0.01

NS = 1_000_000_000
TOL = 1e-9


def print_cells(print_t, price, shares, q: Quotes, anchor_ns: int, window_ns: int, ref_lag_ns: int, horizon_s: float) -> dict[str, np.ndarray]:
    """Classify the prints in [anchor_ns, anchor_ns + window_ns) and compute their passive edge.

    Arguments: print_t, price and shares are the prints; q are the valid quotes; anchor_ns is the anchor time;
    window_ns is the window length; ref_lag_ns is the reference quote lag; horizon_s is the edge horizon in seconds.
    Returns: a dict, per kept print, of u_ns (nanoseconds after the anchor), distance bucket, shares, passive edge in
    basis points (NaN if undefined) and distance in basis points.
    Reference: Lee and Ready (1991), Journal of Finance (the side of a print); Hasbrouck (2007), Empirical Market
    Microstructure.
    """
    t = np.asarray(print_t, dtype=np.int64)
    inside = (t >= anchor_ns) & (t < anchor_ns + window_ns)
    t, p, s = t[inside], np.asarray(price, dtype=float)[inside], np.asarray(shares, dtype=float)[inside]
    if t.size == 0 or len(q.t) == 0:
        empty = np.zeros(0)
        return {"u_ns": np.zeros(0, np.int64), "bucket": np.zeros(0, np.int64), "shares": empty, "edge": empty, "distance": empty}
    c = phases.classify(t, p, q, ref_lag_ns)
    edge = phases.passive_edge(p[c.keep], c.side, c.mid, q, t[c.keep], horizon_s)
    return {"u_ns": t[c.keep] - anchor_ns, "bucket": phases.bucket(c.distance_bps).astype(np.int64), "shares": s[c.keep], "edge": edge, "distance": c.distance_bps}


def interval_index(u_ns: np.ndarray, intervals_ms) -> np.ndarray:
    """Find the interval that holds each time.

    Arguments: u_ns are times in nanoseconds; intervals_ms are consecutive (a_k, b_k) pairs in milliseconds, each
    starting where the previous one ends.
    Returns: the index k of the interval [a_k, b_k) that holds each time, or -1 if none does.
    """
    edges = np.array([int(round(x * 1e6)) for x in [a for a, _ in intervals_ms] + [intervals_ms[-1][1]]], dtype=np.int64)
    k = np.searchsorted(edges, np.asarray(u_ns, dtype=np.int64), side="right") - 1
    return np.where((k >= 0) & (k < edges.size - 1), k, -1)


def time_to_half(q: Quotes, anchor_ns: int, base_ns: int, end_ns: int, tick: float = TICK, min_ticks: int = 2) -> float:
    """Compute the time to half of the move D = m(end) - m(base).

    Arguments: q are the valid quotes; anchor_ns is the anchor; base_ns and end_ns fix the move; tick is the tick size;
    min_ticks is the smallest move in ticks (2 by default).
    Returns: the nanoseconds after the anchor at which the midpoint first covers half of D, from the quote in force at
    the anchor on; NaN for a move under min_ticks or when half is never covered.
    """
    base, end = tape_measures.quote_values(q, [base_ns, end_ns])["mid"]
    move = end - base
    if not (np.isfinite(move) and abs(move) >= min_ticks * tick - TOL):
        return float("nan")
    k0 = int(phases.last_index(q.t, anchor_ns))
    idx = np.arange(max(k0, 0), len(q.t))
    idx = idx[(q.t[idx] <= end_ns)]
    covered = np.sign(move) * (q.mid[idx] - base) >= abs(move) / 2.0 - TOL
    if not covered.any():
        return float("nan")
    return float(max(int(q.t[idx[np.argmax(covered)]]), anchor_ns) - anchor_ns)


def move_bps(q: Quotes, base_ns: int, end_ns: int) -> float:
    """Return the move size |m(end_ns) - m(base_ns)| / m(base_ns) in basis points, NaN if a midpoint is missing."""
    base, end = tape_measures.quote_values(q, [base_ns, end_ns])["mid"]
    return float(abs(end - base) / base * 1e4) if np.isfinite(base) and np.isfinite(end) else float("nan")


def peak_spread(q: Quotes, lo_ns: int, hi_ns: int) -> float:
    """Return the largest quoted spread in basis points over the quote in force at lo_ns and the updates in (lo_ns, hi_ns)."""
    if len(q.t) == 0:
        return float("nan")
    k0 = int(phases.last_index(q.t, lo_ns))
    later = np.flatnonzero((q.t > lo_ns) & (q.t < hi_ns))
    idx = np.concatenate([[k0], later]) if k0 >= 0 else later
    if idx.size == 0:
        return float("nan")
    return float(((q.ask[idx] - q.bid[idx]) / q.mid[idx] * 1e4).max())


def median_update_gap(update_t: np.ndarray, lo_ns: int, hi_ns: int) -> float:
    """Return the median gap in nanoseconds between consecutive updates in [lo_ns, hi_ns), NaN with fewer than two."""
    t = np.sort(np.asarray(update_t, dtype=np.int64))
    t = t[(t >= lo_ns) & (t < hi_ns)]
    return float(np.median(np.diff(t))) if t.size >= 2 else float("nan")


def first_tick_move(q: Quotes, t0_ns: int, direction: float, ticks: int = 1, horizon_ns: int = 3600 * NS, tick: float = TICK) -> float:
    """Compute the time of the first tick move in a given direction after t0.

    Arguments: q are the valid quotes; t0_ns is the release time; direction is the sign of the move; ticks is the move
    size in ticks; horizon_ns is the search horizon (one hour by default); tick is the tick size.
    Returns: the nanoseconds after t0 of the first update whose midpoint is at least ticks ticks from the last midpoint
    before t0 in the direction, NaN if none within the horizon or the direction is 0.
    """
    if direction == 0 or len(q.t) == 0:
        return float("nan")
    k = int(np.searchsorted(q.t, int(t0_ns), side="left")) - 1
    if k < 0:
        return float("nan")
    base = q.mid[k]
    idx = np.arange(k + 1, len(q.t))
    idx = idx[q.t[idx] <= t0_ns + horizon_ns]
    moved = np.round(np.sign(direction) * (q.mid[idx] - base), 9) >= ticks * tick - TOL
    return float(int(q.t[idx[np.argmax(moved)]]) - t0_ns) if moved.any() else float("nan")
