"""Nanosecond-resolution measures of the tape after a release: quote values, move shares, directions, edges, signals and gaps."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from information_diffusion.market_making import phases
from information_diffusion.market_making.phases import Quotes

OFFSETS_MS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)
TICK = 0.01
MOVE_MIN_TICKS = 2
REALTIME_RANGE_S = 300.0
GAP_THRESHOLDS_NS = (1000, 10000, 100000, 1000000)
QUANTISED_STEP_NS = 256
QUANTISED_SHARE = 0.99
SHIFT_DRAWS = 200
SHIFT_SAME_TAPE_MS = 0.5
SHIFT_OTHER_TAPE_MS = 2.5
STABLE_SHARE = 0.5
BOOTSTRAP_RESAMPLES = 2000
LEVEL = 0.95

NS = 1_000_000_000
PRICE_DECIMALS = 9
TOLERANCE = 1e-9


def anchor_ns(t0_ns: int, seconds: float) -> int:
    """Return the anchor time t0_ns + round(1e9 seconds) in integer nanoseconds, for a time in seconds after t0."""
    return int(t0_ns) + int(round(float(seconds) * 1e9))


def offsets_ns(offsets_ms=OFFSETS_MS) -> np.ndarray:
    """Return the offsets given in milliseconds (OFFSETS_MS by default) as integer nanoseconds."""
    return np.array([int(round(o * 1e6)) for o in offsets_ms], dtype=np.int64)


def quote_values(q: Quotes, at_ns) -> dict[str, np.ndarray]:
    """Look up the quote in force at each time.

    Arguments: q are the valid quotes; at_ns are the times in nanoseconds.
    Returns: a dict of bid, ask, bid_size, ask_size and mid arrays, NaN before the first valid update.
    """
    at = np.asarray(at_ns, dtype=np.int64)
    k = phases.last_index(q.t, at)
    ok = (k >= 0) & (len(q.t) > 0)
    kk = np.maximum(k, 0)
    pick = lambda a: np.where(ok, np.asarray(a, dtype=float)[kk], np.nan) if len(q.t) else np.full(at.shape, np.nan)
    bid, ask = pick(q.bid), pick(q.ask)
    return {"bid": bid, "ask": ask, "bid_size": pick(q.bid_size), "ask_size": pick(q.ask_size), "mid": (bid + ask) / 2.0}


def spread_bps(values: dict[str, np.ndarray]) -> np.ndarray:
    """Return the quoted spread (ask - bid) / mid in basis points from a quote_values dict."""
    return (values["ask"] - values["bid"]) / values["mid"] * 1e4


def displayed_size(values: dict[str, np.ndarray]) -> np.ndarray:
    """Return the displayed size, bid size plus ask size in shares, from a quote_values dict."""
    return values["bid_size"] + values["ask_size"]


def share_of_move(m_base, m_at, m_end, tick: float = TICK, min_ticks: int = MOVE_MIN_TICKS) -> np.ndarray:
    """Compute the share of the move (m_at - m_base) / (m_end - m_base).

    Arguments: m_base, m_at and m_end are midpoints in dollars; tick is the tick size; min_ticks is the smallest move
    in ticks (2 by default).
    Returns: the share, NaN where the move is below min_ticks ticks or a midpoint is missing.
    """
    m_base, m_at, m_end = (np.asarray(x, dtype=float) for x in (m_base, m_at, m_end))
    move = m_end - m_base
    ok = np.isfinite(move) & np.isfinite(m_at) & (np.abs(move) >= min_ticks * tick - TOLERANCE)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, (m_at - m_base) / np.where(ok, move, 1.0), np.nan)


def moved_enough(m_base, m_end, tick: float = TICK, min_ticks: int = MOVE_MIN_TICKS) -> np.ndarray:
    """Return True where |m_end - m_base| is finite and at least min_ticks ticks of size tick."""
    move = np.asarray(m_end, dtype=float) - np.asarray(m_base, dtype=float)
    return np.isfinite(move) & (np.abs(move) >= min_ticks * tick - TOLERANCE)


def own_direction(m_at, m_base) -> np.ndarray:
    """Return the own direction sign(m_at - m_base) after rounding to 1e-9 dollars, 0 where either midpoint is missing."""
    diff = np.round(np.asarray(m_at, dtype=float) - np.asarray(m_base, dtype=float), PRICE_DECIMALS)
    return np.where(np.isfinite(diff), np.sign(diff), 0.0)


def index_direction(m_at: np.ndarray, m_base: np.ndarray) -> np.ndarray:
    """Return the index direction, the sign of the mean over instruments (axis 0) of log m_at - log m_base.

    Instruments missing either midpoint are skipped; the direction is 0 where no instrument has both.
    """
    m_at, m_base = np.asarray(m_at, dtype=float), np.asarray(m_base, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(m_at) - np.log(m_base)
    ok = np.isfinite(r)
    n = ok.sum(axis=0)
    mean = np.where(n > 0, np.where(ok, r, 0.0).sum(axis=0) / np.maximum(n, 1), 0.0)
    return np.sign(np.round(mean, 15))


def taker_edge(q: Quotes, at_ns, direction, horizon_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the taker edge d (m(T + H) - P) / m(T) * 1e4 per share of crossing the spread at T, with no latency.

    P is the ask for d = +1 and the bid for d = -1.
    Arguments: q are the valid quotes; at_ns are the times T in nanoseconds; direction is d; horizon_s is H in seconds.
    Returns: the edge in basis points and the displayed size at the quote crossed; NaN and size 0 where d = 0 or a
    quote is missing.
    Reference: Lee and Ready (1991), Journal of Finance (the side of a print); Hasbrouck (2007), Empirical Market
    Microstructure, ch. 5 (mark-outs).
    """
    at = np.asarray(at_ns, dtype=np.int64)
    d = np.asarray(direction, dtype=float)
    now = quote_values(q, at)
    later = quote_values(q, at + int(round(horizon_s * 1e9)))["mid"]
    price = np.where(d > 0, now["ask"], now["bid"])
    size = np.where(d > 0, now["ask_size"], now["bid_size"])
    trade = (d != 0) & np.isfinite(price) & np.isfinite(later) & np.isfinite(now["mid"])
    with np.errstate(invalid="ignore"):
        edge = np.where(trade, d * (later - price) / now["mid"] * 1e4, np.nan)
    return edge, np.where(trade, size, 0.0)


def counts_in(times_ns: np.ndarray, lo_ns, hi_ns) -> np.ndarray:
    """Return the number of events with lo_ns <= t < hi_ns, for sorted event times times_ns."""
    t = np.asarray(times_ns, dtype=np.int64)
    return np.searchsorted(t, np.asarray(hi_ns, dtype=np.int64), side="left") - np.searchsorted(t, np.asarray(lo_ns, dtype=np.int64), side="left")


def sums_in(times_ns: np.ndarray, values: np.ndarray, lo_ns, hi_ns) -> np.ndarray:
    """Return the sum of values over events with lo_ns <= t < hi_ns, for sorted event times times_ns."""
    t = np.asarray(times_ns, dtype=np.int64)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=float))])
    return c[np.searchsorted(t, np.asarray(hi_ns, dtype=np.int64), side="left")] - c[np.searchsorted(t, np.asarray(lo_ns, dtype=np.int64), side="left")]


def trailing_signals(t_ns: np.ndarray, mid: np.ndarray, window_s: float = REALTIME_RANGE_S, start_ns: int | None = None) -> np.ndarray:
    """Find the NBBO updates whose midpoint leaves the range of the midpoints of the updates in [t - W, t).

    Arguments: t_ns are the update times in nanoseconds; mid are the midpoints; window_s is W in seconds (300 by
    default); start_ns is the first time that can signal (by default the first update plus W).
    Returns: the signal times in nanoseconds.
    """
    t_ns, mid = np.asarray(t_ns, dtype=np.int64), np.asarray(mid, dtype=float)
    if t_ns.size == 0:
        return np.zeros(0, dtype=np.int64)
    s = pd.Series(mid, index=pd.to_datetime(t_ns, utc=True))
    window = pd.Timedelta(int(round(window_s * 1e9)), unit="ns")
    high = s.rolling(window, closed="left").max().to_numpy()
    low = s.rolling(window, closed="left").min().to_numpy()
    first = int(t_ns[0]) + int(round(window_s * 1e9)) if start_ns is None else int(start_ns)
    with np.errstate(invalid="ignore"):
        out = (t_ns >= first) & np.isfinite(high) & ((np.round(mid - high, PRICE_DECIMALS) > 0) | (np.round(mid - low, PRICE_DECIMALS) < 0))
    return t_ns[out]


def episodes(signals_ns: np.ndarray, from_ns: int, to_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge the pauses [s + from_ns, s + to_ns) of the signals s into episodes, the connected pieces of their union.

    Returns: the starts and ends of the episodes in nanoseconds.
    """
    s = np.sort(np.asarray(signals_ns, dtype=np.int64))
    if s.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lo, hi = s + int(from_ns), s + int(to_ns)
    starts, ends = [int(lo[0])], [int(hi[0])]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= ends[-1]:
            ends[-1] = max(ends[-1], int(b))
        else:
            starts.append(int(a))
            ends.append(int(b))
    return np.array(starts, dtype=np.int64), np.array(ends, dtype=np.int64)


def nbbo_events(t_ns, bid, bid_size, bid_exchange, ask, ask_size, ask_exchange) -> pd.DataFrame:
    """Attribute NBBO updates to exchanges: one event per side whose price, size or exchange changed.

    The first update counts both sides; duplicate (update, exchange) pairs are dropped.
    Arguments: t_ns are the update times; bid, bid_size, bid_exchange, ask, ask_size and ask_exchange are the quotes.
    Returns: a DataFrame with columns t, exchange and row (the update's position).
    """
    t = np.asarray(t_ns, dtype=np.int64)
    if t.size == 0:
        return pd.DataFrame({"t": np.zeros(0, np.int64), "exchange": np.zeros(0, object), "row": np.zeros(0, np.int64)})
    frames = []
    for price, size, ex in ((bid, bid_size, bid_exchange), (ask, ask_size, ask_exchange)):
        price, size, ex = np.asarray(price, dtype=float), np.asarray(size, dtype=float), np.asarray(ex, dtype=object)
        changed = np.ones(t.size, dtype=bool)
        changed[1:] = (np.round(price[1:] - price[:-1], PRICE_DECIMALS) != 0) | (size[1:] != size[:-1]) | (ex[1:] != ex[:-1])
        rows = np.flatnonzero(changed)
        frames.append(pd.DataFrame({"t": t[rows], "exchange": ex[rows], "row": rows}))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(["row", "exchange"])
    return out.sort_values(["row", "exchange"], kind="mergesort").reset_index(drop=True)


def gaps_within(t_ns: np.ndarray, exchange: np.ndarray) -> np.ndarray:
    """Return the gaps in nanoseconds between consecutive events of the same exchange, for events in time order."""
    f = pd.DataFrame({"t": np.asarray(t_ns, dtype=np.int64), "x": np.asarray(exchange, dtype=object)})
    if len(f) < 2:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.diff(g.t.to_numpy()) for _, g in f.groupby("x", sort=True)] + [np.zeros(0, dtype=np.int64)])


def gaps_across_rows(t_ns: np.ndarray, row: np.ndarray, exchange: np.ndarray) -> np.ndarray:
    """Return the gaps in nanoseconds between consecutive rows of the consolidated stream that share no exchange.

    Arguments: t_ns, row and exchange are the event times, row positions and exchanges; a row may carry several
    exchanges.
    """
    f = pd.DataFrame({"t": np.asarray(t_ns, dtype=np.int64), "row": np.asarray(row), "x": np.asarray(exchange, dtype=object)})
    if f.row.nunique() < 2:
        return np.zeros(0, dtype=np.int64)
    sets = f.groupby("row", sort=True).agg(t=("t", "first"), x=("x", lambda v: frozenset(v)))
    t, xs = sets.t.to_numpy(), sets.x.to_list()
    keep = np.array([not (xs[i] & xs[i - 1]) for i in range(1, len(xs))], dtype=bool)
    return np.diff(t)[keep]


def gaps_across(t_ns: np.ndarray, exchange: np.ndarray) -> np.ndarray:
    """Return the gaps in nanoseconds between consecutive events whose exchanges differ, for one exchange per event."""
    t, x = np.asarray(t_ns, dtype=np.int64), np.asarray(exchange, dtype=object)
    if t.size < 2:
        return np.zeros(0, dtype=np.int64)
    return np.diff(t)[x[1:] != x[:-1]]


def gap_shares(gaps_ns: np.ndarray, thresholds_ns=GAP_THRESHOLDS_NS) -> dict[str, float]:
    """Count the gaps in nanoseconds: in total, equal to zero, and strictly below each threshold.

    Returns: a dict with the counts "gaps", "zero" and "below_<h>ns" for each threshold h.
    """
    g = np.asarray(gaps_ns, dtype=np.int64)
    out = {"gaps": int(g.size), "zero": int((g == 0).sum())}
    for h in thresholds_ns:
        out[f"below_{h}ns"] = int((g < h).sum())
    return out


def quantised(times_ns: np.ndarray, step_ns: int = QUANTISED_STEP_NS, share: float = QUANTISED_SHARE) -> bool:
    """Return True when at least the given share of the timestamps are multiples of step_ns (False for an empty stream)."""
    t = np.asarray(times_ns, dtype=np.int64)
    return bool(t.size) and float(np.mean(t % step_ns == 0)) >= share


def shift_draws(rng: np.random.Generator, same_tape: np.ndarray, draws: int = SHIFT_DRAWS) -> np.ndarray:
    """Draw uniform clock shifts in integer nanoseconds.

    Arguments: rng is the random generator; same_tape is a boolean mask over instruments, True for those on the same
    tape as the reference instrument (for example the first to react), which are shifted within +/-0.5 ms, while the
    others are shifted within +/-2.5 ms; draws is the number of draws (SHIFT_DRAWS by default).
    Returns: an array (draws, instruments) of shifts.
    """
    same = np.asarray(same_tape, dtype=bool)
    half = np.where(same, SHIFT_SAME_TAPE_MS, SHIFT_OTHER_TAPE_MS) * 1e6
    return np.rint(rng.uniform(-1.0, 1.0, size=(draws, same.size)) * half[None, :]).astype(np.int64)


def stable_from(offsets, ranges, widths, share: float = STABLE_SHARE) -> float:
    """Find the smallest offset from which range <= share * width holds at it and at every longer offset.

    Arguments: offsets, ranges and widths are aligned arrays (a missing range or width fails); share is 0.5 by default.
    Returns: the stable offset, NaN when the last offset fails.
    """
    o, r, w = np.asarray(offsets, dtype=float), np.asarray(ranges, dtype=float), np.asarray(widths, dtype=float)
    ok = np.isfinite(r) & np.isfinite(w) & (r <= share * w + 1e-12)
    if ok.size == 0 or not ok[-1]:
        return float("nan")
    k = ok.size - 1
    while k > 0 and ok[k - 1]:
        k -= 1
    return float(o[k])


def release_interval(values: np.ndarray, stat: Callable[[np.ndarray], float], rng: np.random.Generator, resamples: int = BOOTSTRAP_RESAMPLES, level: float = LEVEL) -> tuple[float, float]:
    """Compute the bootstrap percentile interval of a statistic over resamples of the rows with replacement.

    Arguments: values are the rows (one per release); stat maps rows to a number; rng is the random generator;
    resamples is the number of resamples; level is the coverage (0.95 by default). NaN draws are dropped.
    Returns: the lower and upper bounds, NaN when no draw is finite.
    Reference: Efron and Tibshirani (1993).
    """
    v = np.asarray(values, dtype=float)
    n = v.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    draws = np.array([stat(v[rng.integers(0, n, n)]) for _ in range(resamples)], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
