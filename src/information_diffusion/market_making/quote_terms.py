"""Common quote terms of every quoting rule, rounding to the tick grid and as-of lookup of input series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SNAP = 1e-9


@dataclass(frozen=True)
class QuoteTerms:
    """Quote terms over the decision times, in dollars: reservation(q) = center - q skew, quoted -/+ half_spread.

    Fields: center, skew (per share of inventory) and half_spread.
    """

    center: np.ndarray
    skew: np.ndarray
    half_spread: np.ndarray


def quote_ticks(terms: QuoteTerms, inventory: int, tick: float) -> tuple[np.ndarray, np.ndarray]:
    """Round the quotes to the tick grid at one inventory level.

    The bid rounds down and the ask rounds up, values within SNAP of a grid point count as on it, and the half-spread
    is at least half a tick.
    Arguments: terms are the QuoteTerms; inventory is the inventory in shares; tick is the tick size in dollars.
    Returns: the bid and ask in whole ticks for every decision time.
    """
    reservation = terms.center - inventory * terms.skew
    half = np.maximum(terms.half_spread, tick / 2)
    bid = np.floor((reservation - half) / tick + SNAP).astype(np.int64)
    ask = np.ceil((reservation + half) / tick - SNAP).astype(np.int64)
    return bid, ask


def as_of(series_t_ns: np.ndarray, values: np.ndarray, query_t_ns: np.ndarray, default: np.ndarray | float) -> np.ndarray:
    """Look up a time series as of each query time, without look-ahead.

    Arguments: series_t_ns are non-decreasing timestamps in nanoseconds; values are the finite series values;
    query_t_ns are the query times; default is used before the first timestamp.
    Returns: the last value at or before each query time, or the default.
    """
    series_t_ns, values = np.asarray(series_t_ns, dtype=np.int64), np.asarray(values, dtype=float)
    if series_t_ns.size and np.any(series_t_ns[1:] < series_t_ns[:-1]):
        raise ValueError("input series timestamps must be non-decreasing")
    if not np.all(np.isfinite(values)):
        raise ValueError("input series values must be finite")
    index = np.searchsorted(series_t_ns, query_t_ns, side="right") - 1
    result = np.where(index >= 0, values[np.maximum(index, 0)] if values.size else np.nan, default)
    return result.astype(float)
