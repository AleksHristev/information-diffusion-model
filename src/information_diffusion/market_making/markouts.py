"""Mark-outs of fills: the signed NBBO mid move after each fill, in ticks and basis points."""

from __future__ import annotations

import numpy as np

from information_diffusion.market_making.market import MarketData

_NS_PER_S = 1_000_000_000


def mid_doubled_at(market: MarketData, t_ns: np.ndarray) -> np.ndarray:
    """Look up twice the mid in micro-dollars at each time, from the last NBBO update at or before it.

    Arguments: market is the MarketData; t_ns are the times in nanoseconds, none before the first NBBO update.
    Returns: the doubled mid at each time.
    """
    index = np.searchsorted(market.quote_t, t_ns, side="right") - 1
    if np.any(index < 0):
        raise ValueError("a time precedes the first NBBO update")
    return market.mid_doubled[index]


def markouts(market: MarketData, t_ns: np.ndarray, sides: np.ndarray, horizon_s: float, tick_units: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute mark-outs s (m(t + h) - m(t)) in ticks and 1e4 s (m(t + h) - m(t)) / m(t) in basis points.

    Positive values mean the mid moved in the direction of the fill; NaN where t + h is after the last NBBO update.
    Arguments: market is the MarketData; t_ns are the fill times in nanoseconds; sides are +1 for a buy and -1 for a
    sell; horizon_s is the horizon h in seconds; tick_units is the tick in micro-dollars.
    Returns: the mark-outs in ticks and in basis points.
    """
    target = np.asarray(t_ns, dtype=np.int64) + int(round(horizon_s * _NS_PER_S))
    last = market.quote_t[-1] if market.quote_t.size else np.iinfo(np.int64).min
    available = target <= last
    ticks, bps = np.full(len(target), np.nan), np.full(len(target), np.nan)
    if available.any():
        reference = mid_doubled_at(market, np.asarray(t_ns, dtype=np.int64)[available])
        change = np.asarray(sides, dtype=np.int64)[available] * (mid_doubled_at(market, target[available]) - reference)
        ticks[available] = change / (2.0 * tick_units)
        bps[available] = change * 1e4 / reference
    return ticks, bps
