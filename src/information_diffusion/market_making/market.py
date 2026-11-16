"""NBBO and trade arrays normalised to integer micro-dollars and nanoseconds for the backtest."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.market_making.params import PRICE_SCALE

_NS_PER_S = 1_000_000_000


def _to_ns(values, unit: str) -> np.ndarray:
    array = np.asarray(values)
    if unit == "ns":
        return array.astype(np.int64)
    if unit == "s":
        return np.rint(array.astype(float) * _NS_PER_S).astype(np.int64)
    raise ValueError(f"unknown timestamp unit {unit!r}")


def _to_units(prices) -> np.ndarray:
    return np.rint(np.asarray(prices, dtype=float) * PRICE_SCALE).astype(np.int64)


def _require_sorted(times: np.ndarray, label: str) -> None:
    if times.size and np.any(times[1:] < times[:-1]):
        raise ValueError(f"{label} timestamps must be non-decreasing")


@dataclass(frozen=True)
class MarketData:
    """Usable NBBO updates and trades of one instrument.

    Fields: quote_t, bid, ask, bid_size and ask_size of the usable NBBO updates (ask > bid > 0); trade_t, trade_price and
    trade_size of the trades; dropped_quotes counts the crossed, locked or empty updates removed. Times are in
    nanoseconds, prices in micro-dollars and sizes in shares.
    """

    quote_t: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    bid_size: np.ndarray
    ask_size: np.ndarray
    trade_t: np.ndarray
    trade_price: np.ndarray
    trade_size: np.ndarray
    dropped_quotes: int = 0

    @classmethod
    def from_arrays(cls, quote_t, bid, ask, bid_size, ask_size, trade_t, trade_price, trade_size, *, ts_unit: str = "ns") -> "MarketData":
        """Build a MarketData from NBBO updates and trades, dropping unusable quotes.

        Arguments: quote_t, bid, ask, bid_size and ask_size are the NBBO updates; trade_t, trade_price and trade_size
        are the trades; prices are in dollars, sizes in shares, and timestamps non-decreasing in ts_unit ("ns" or "s").
        Returns: a MarketData.
        """
        q_t, t_t = _to_ns(quote_t, ts_unit), _to_ns(trade_t, ts_unit)
        q_bid, q_ask = _to_units(bid), _to_units(ask)
        q_bs, q_as = np.asarray(bid_size).astype(np.int64), np.asarray(ask_size).astype(np.int64)
        t_px, t_sz = _to_units(trade_price), np.asarray(trade_size).astype(np.int64)
        if not (len(q_t) == len(q_bid) == len(q_ask) == len(q_bs) == len(q_as)):
            raise ValueError("NBBO arrays must have equal lengths")
        if not (len(t_t) == len(t_px) == len(t_sz)):
            raise ValueError("trade arrays must have equal lengths")
        _require_sorted(q_t, "NBBO")
        _require_sorted(t_t, "trade")
        if np.any(q_bs < 0) or np.any(q_as < 0) or np.any(t_sz < 0):
            raise ValueError("sizes must be non-negative")
        usable = (q_bid > 0) & (q_ask > q_bid)
        return cls(q_t[usable], q_bid[usable], q_ask[usable], q_bs[usable], q_as[usable], t_t, t_px, t_sz, int(len(q_t) - usable.sum()))

    @property
    def mid_doubled(self) -> np.ndarray:
        """Return twice the mid in micro-dollars, an exact integer."""
        return self.bid + self.ask

    @property
    def mid_dollars(self) -> np.ndarray:
        """Return the NBBO mid in dollars."""
        return (self.bid + self.ask) / (2 * PRICE_SCALE)
