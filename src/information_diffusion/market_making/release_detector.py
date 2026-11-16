"""Real-time release detector: trailing-range exits of each instrument and k-of-n coincidences across instruments."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_DECIMALS = 9


def trailing_exits(t_ns: np.ndarray, mid: np.ndarray, window_s: float, start_ns: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Find the exits of one instrument: NBBO updates whose midpoint leaves the range of the midpoints in [t - W, t).

    Arguments: t_ns are the update times in nanoseconds; mid are the midpoints; window_s is W in seconds; start_ns is
    the first time that can exit (by default the first update plus W).
    Returns: the exit times and directions, +1 above the range and -1 below.
    """
    t_ns, mid = np.asarray(t_ns, dtype=np.int64), np.asarray(mid, dtype=float)
    if t_ns.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    s = pd.Series(mid, index=pd.to_datetime(t_ns, utc=True))
    window = pd.Timedelta(int(round(window_s * 1e9)), unit="ns")
    high = s.rolling(window, closed="left").max().to_numpy()
    low = s.rolling(window, closed="left").min().to_numpy()
    first = int(t_ns[0]) + int(round(window_s * 1e9)) if start_ns is None else int(start_ns)
    ok = (t_ns >= first) & np.isfinite(high)
    with np.errstate(invalid="ignore"):
        up = ok & (np.round(mid - high, PRICE_DECIMALS) > 0)
        down = ok & (np.round(mid - low, PRICE_DECIMALS) < 0)
    keep = up | down
    return t_ns[keep], np.where(up[keep], 1, -1).astype(np.int64)


def coincidences(exits: dict[str, tuple[np.ndarray, np.ndarray]], k: int, window_ns: int) -> np.ndarray:
    """Find the k-of-n coincidence signals across instruments.

    An exit of an instrument at time t is a signal when at least k - 1 other instruments have an exit in the same
    direction in [t - window_ns, t].
    Arguments: exits maps an instrument to its (times, directions), as returned by trailing_exits; k is the number of
    instruments required; window_ns is the coincidence window in nanoseconds.
    Returns: the distinct signal times.
    Reference: Kay (1998), Fundamentals of Statistical Signal Processing, vol. II.
    """
    funds = list(exits)
    by = {(f, d): np.sort(exits[f][0][exits[f][1] == d]) for f in funds for d in (1, -1)}
    out = []
    for f in funds:
        t, d = exits[f]
        for direction in (1, -1):
            mine = np.sort(t[d == direction])
            if mine.size == 0:
                continue
            count = np.zeros(mine.size, dtype=np.int64)
            for g in funds:
                if g == f:
                    continue
                other = by[(g, direction)]
                if other.size == 0:
                    continue
                hi = np.searchsorted(other, mine, side="right")
                lo = np.searchsorted(other, mine - int(window_ns), side="left")
                count += (hi > lo).astype(np.int64)
            out.append(mine[count >= k - 1])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def first_at_or_after(signals_ns: np.ndarray, at_ns: int) -> int | None:
    """Return the first signal at or after at_ns as an integer number of nanoseconds, or None if there is none."""
    s = np.sort(np.asarray(signals_ns, dtype=np.int64))
    k = int(np.searchsorted(s, int(at_ns), side="left"))
    return int(s[k]) if k < s.size else None
