"""Direction of a price move, the side of resting orders relative to it, and inventory statistics over an interval."""

from __future__ import annotations

import numpy as np

from information_diffusion.market_making import tape_measures
from information_diffusion.market_making.phases import Quotes

PRICE_DECIMALS = 9


def further_move(q: Quotes, at_ns: np.ndarray, direction: np.ndarray, horizon_s: float) -> np.ndarray:
    """Compute the further move d (m(T + H) - m(T)) / m(T) * 1e4 in basis points.

    Arguments: q are the valid quotes; at_ns are the times T in nanoseconds; direction is d at each time; horizon_s is
    H in seconds.
    Returns: the further move, NaN where d = 0 or a midpoint is missing.
    Reference: Hasbrouck (2007).
    """
    at = np.asarray(at_ns, dtype=np.int64)
    now = tape_measures.quote_values(q, at)["mid"]
    later = tape_measures.quote_values(q, at + int(round(horizon_s * 1e9)))["mid"]
    d = np.asarray(direction, dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where((d != 0) & np.isfinite(now) & np.isfinite(later), d * (later - now) / now * 1e4, np.nan)


def resting_side(print_side: np.ndarray) -> np.ndarray:
    """Return the resting side -s of each print with aggressor side s.

    Reference: Lee and Ready (1991), Journal of Finance (the aggressor side).
    """
    return -np.asarray(print_side, dtype=int)


def side_of_move(resting: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Return +1 where the resting side equals the direction, -1 where it is against it, and 0 where the direction is 0."""
    r, d = np.asarray(resting, dtype=int), np.asarray(direction, dtype=int)
    return np.where(d == 0, 0, np.where(r == d, 1, -1))


def base_of(t_ns: np.ndarray, base_t: np.ndarray, base_ref: np.ndarray) -> np.ndarray:
    """Look up the reference instant of the base in force at each time.

    A base is the instant whose midpoint a move is measured from; it changes at the sorted times base_t, and the base
    set at base_t[i] is the instant base_ref[i].
    Arguments: t_ns are the times in nanoseconds; base_t are the base change times; base_ref are their reference
    instants.
    Returns: the reference instant of the latest base change at or before each time, -1 before the first.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    k = np.searchsorted(np.asarray(base_t, dtype=np.int64), t, side="right") - 1
    return np.where(k >= 0, np.asarray(base_ref, dtype=np.int64)[np.maximum(k, 0)], -1)


def own_direction_series(q: Quotes, base_t: np.ndarray, base_ref: np.ndarray, lo_ns: int, hi_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the step series of an instrument's own direction sign(m(T) - m(base)) over [lo_ns, hi_ns].

    The direction is evaluated at the instrument's quote updates, at each base change and at lo_ns, with the base
    midpoint the one in force at the reference instant of the base in force (see base_of); it is 0 before the first
    base change.
    Arguments: q are the instrument's valid quotes; base_t are the times at which the base changes and base_ref the
    reference instant of each new base, in nanoseconds; lo_ns and hi_ns bound the interval in nanoseconds.
    Returns: the times and the directions.
    """
    times = np.unique(np.concatenate([q.t[(q.t >= lo_ns) & (q.t <= hi_ns)], np.asarray(base_t, dtype=np.int64)[(np.asarray(base_t) >= lo_ns) & (np.asarray(base_t) <= hi_ns)], [lo_ns]]))
    ref = base_of(times, base_t, base_ref)
    now = tape_measures.quote_values(q, times)["mid"]
    base = np.where(ref >= 0, tape_measures.quote_values(q, np.maximum(ref, 0))["mid"], np.nan)
    d = tape_measures.own_direction(now, base)
    return times, d


def index_direction_series(quotes: list[Quotes], base_t: np.ndarray, base_ref: np.ndarray, lo_ns: int, hi_ns: int) -> tuple[np.ndarray, np.ndarray]:
    """Build the step series of the index direction, the sign of the mean log midpoint change since the base.

    The direction is evaluated at every quote update of any instrument, at each base change and at lo_ns; see
    tape_measures.index_direction for the mean over instruments.
    Arguments: quotes are the instruments' valid quotes; base_t and base_ref are as in own_direction_series; lo_ns and
    hi_ns bound the interval in nanoseconds.
    Returns: the times and the directions.
    """
    parts = [q.t[(q.t >= lo_ns) & (q.t <= hi_ns)] for q in quotes]
    times = np.unique(np.concatenate(parts + [np.asarray(base_t, dtype=np.int64)[(np.asarray(base_t) >= lo_ns) & (np.asarray(base_t) <= hi_ns)], [lo_ns]]))
    ref = base_of(times, base_t, base_ref)
    now = np.stack([tape_measures.quote_values(q, times)["mid"] for q in quotes])
    base = np.stack([np.where(ref >= 0, tape_measures.quote_values(q, np.maximum(ref, 0))["mid"], np.nan) for q in quotes])
    return times, tape_measures.index_direction(now, base)


def inventory_stats(fill_t: np.ndarray, inventory_after: np.ndarray, lo_ns: int, hi_ns: int) -> tuple[float, float]:
    """Compute inventory statistics over [lo_ns, hi_ns) for a step path q(t) that is zero before the first fill.

    Arguments: fill_t are the fill times in nanoseconds; inventory_after is the inventory after each fill; lo_ns and
    hi_ns bound the interval.
    Returns: the time-weighted mean of |q| and the time-weighted variance of q (NaN for an empty interval).
    """
    t, level = np.asarray(fill_t, dtype=np.int64), np.asarray(inventory_after, dtype=float)
    if hi_ns <= lo_ns:
        return float("nan"), float("nan")
    before = np.flatnonzero(t < lo_ns)
    start = level[before[-1]] if before.size else 0.0
    inside = (t >= lo_ns) & (t < hi_ns)
    times = np.concatenate(([lo_ns], t[inside], [hi_ns])).astype(float)
    levels = np.concatenate(([start], level[inside]))
    widths = np.diff(times)
    span = float(hi_ns - lo_ns)
    mean = float((levels * widths).sum() / span)
    return float((np.abs(levels) * widths).sum() / span), float(((levels - mean) ** 2 * widths).sum() / span)
