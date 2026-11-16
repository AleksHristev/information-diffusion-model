"""Event-driven market-making backtest for one quoting rule, one fill model and one latency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from information_diffusion.market_making.fill_optimistic import OptimisticFills
from information_diffusion.market_making.fill_pessimistic import PessimisticFills
from information_diffusion.market_making.market import MarketData
from information_diffusion.market_making.markouts import markouts, mid_doubled_at
from information_diffusion.market_making.params import MARKOUT_HORIZONS_S, EngineParams
from information_diffusion.market_making.quote_terms import QuoteTerms, quote_ticks
from information_diffusion.market_making.results import Fills, RunResult

_NO_ORDER = int(np.iinfo(np.int64).min)
_FIRST_CHUNK = 4096
_SMALLEST_CHUNK = 256
_LARGEST_CHUNK = 1 << 20
_ZERO_INDEX = np.zeros(1, dtype=np.int64)
_NO_INDEX = np.zeros(0, dtype=np.int64)

FILL_MODELS = {"pessimistic": PessimisticFills(), "optimistic": OptimisticFills()}


class QuotingRule(Protocol):
    """Interface of a quoting rule: a name and a quote_terms method."""

    name: str

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Compute the quote terms at the decision times.

        Arguments: t_ns are the decision times in nanoseconds; mid is the NBBO mid in dollars at those times;
        window_end_ns is the end of the backtest window in nanoseconds.
        Returns: a QuoteTerms with one entry per decision time.
        """
        ...


@dataclass(slots=True)
class _Order:
    px: int
    rem: int
    ahead: float = 0
    scanned_to: int = -1


@dataclass(slots=True)
class _Side:
    sign: int
    best: np.ndarray
    best_size: np.ndarray
    far: np.ndarray
    far_size: np.ndarray
    last_target: int = _NO_ORDER
    live: _Order | None = None
    pending: list = field(default_factory=list)


def default_window(market: MarketData) -> tuple[int, int]:
    """Return the whole-sample window in nanoseconds, from the first usable NBBO update to just after the last."""
    return int(market.quote_t[0]), int(market.quote_t[-1]) + 1


def _decision_range(market: MarketData, window: tuple[int, int]) -> tuple[int, int]:
    lo, hi = np.searchsorted(market.quote_t, list(window), side="left")
    return int(lo), int(hi)


def rule_terms(market: MarketData, rule: QuotingRule, window: tuple[int, int]) -> QuoteTerms:
    """Evaluate a rule at every decision time in the window.

    Arguments: market is the MarketData; rule is the quoting rule; window is (start, end) in nanoseconds.
    Returns: the QuoteTerms of the rule, checked to have one entry per decision.
    """
    lo, hi = _decision_range(market, window)
    terms = rule.quote_terms(market.quote_t[lo:hi], market.mid_dollars[lo:hi], window[1])
    if not (len(terms.center) == len(terms.skew) == len(terms.half_spread) == hi - lo):
        raise ValueError(f"{rule.name} returned arrays of the wrong length")
    return terms


def run_backtest(market: MarketData, rule: QuotingRule, fill_model: str, latency_ns: int, params: EngineParams = EngineParams(), *, window: tuple[int, int] | None = None, record_orders: bool = False) -> RunResult:
    """Evaluate a rule over a window and run the backtest.

    Arguments: market is the MarketData; rule is the quoting rule; fill_model is "pessimistic" or "optimistic";
    latency_ns is the one-way latency in nanoseconds; params are the EngineParams; window is (start, end) in
    nanoseconds (the whole sample by default); record_orders keeps a log of order arrivals.
    Returns: a RunResult.
    """
    window = default_window(market) if window is None else window
    return simulate(market, rule_terms(market, rule, window), rule.name, fill_model, latency_ns, params, window, record_orders=record_orders)


def simulate(market: MarketData, terms: QuoteTerms, rule_name: str, fill_model: str, latency_ns: int, params: EngineParams, window: tuple[int, int], *, record_orders: bool = False) -> RunResult:
    """Run the event-driven backtest from precomputed quote terms.

    At each NBBO update a changed target bid or ask sends an order, or a cancel at the inventory limit, which arrives
    after latency_ns. At equal timestamps NBBO updates come first, then trades, then order arrivals. An order
    marketable on arrival takes the far NBBO up to its displayed size; other fills are maker fills at the resting
    price, decided by the fill model, and every fill is capped by inventory headroom.
    Arguments: market is the MarketData; terms are the QuoteTerms at the decision times; rule_name names the rule;
    fill_model is "pessimistic" or "optimistic"; latency_ns is the one-way latency in nanoseconds; params are the
    EngineParams; window is (start, end) in nanoseconds; record_orders keeps a log of order arrivals.
    Returns: a RunResult.
    """
    if latency_ns < 0:
        raise ValueError("latency must be non-negative")
    model = FILL_MODELS[fill_model]
    start, end = window
    quote_t = market.quote_t
    lo, hi = _decision_range(market, window)
    decision_t = quote_t[lo:hi]
    n_decisions = len(decision_t)
    if len(terms.center) != n_decisions:
        raise ValueError("the quote terms do not match the decision times")
    first_trade, last_trade = np.searchsorted(market.trade_t, [start, end], side="left")
    trade_t = market.trade_t[first_trade:last_trade]
    trade_px = market.trade_price[first_trade:last_trade]
    trade_sz = market.trade_size[first_trade:last_trade]

    buy = _Side(1, market.bid, market.bid_size, market.ask, market.ask_size)
    sell = _Side(-1, -market.ask, market.ask_size, -market.bid, market.bid_size)
    sides = (buy, sell)
    limit, tick, tick_units = params.inventory_limit, params.tick, params.tick_units
    take_fee, add_rebate, order_size = params.take_fee_units, params.add_rebate_units, params.order_size

    inventory = 0
    cash = 0
    ledger: list[tuple] = []
    order_log: list[tuple] = []
    n_orders = 0
    i0, j0 = 0, 0
    chunk = _FIRST_CHUNK

    def record_fill(t: int, side: _Side, qty: int, price: int, maker: bool) -> None:
        nonlocal inventory, cash
        flow = qty * (add_rebate if maker else -take_fee)
        inventory += side.sign * qty
        cash += -side.sign * qty * price + flow
        ledger.append((t, side.sign, qty, price, maker, flow, inventory))

    while True:
        i1 = min(n_decisions, i0 + chunk)
        horizon = int(decision_t[i1]) - 1 if i1 < n_decisions else end - 1
        targets = quote_ticks(QuoteTerms(terms.center[i0:i1], terms.skew[i0:i1], terms.half_spread[i0:i1]), inventory, tick)
        side_targets = [targets[0] * tick_units, -(targets[1] * tick_units)]

        columns: list[list[np.ndarray]] = [[], [], [], [], []]
        timelines: list[tuple[np.ndarray, np.ndarray]] = []
        for side_index, side in enumerate(sides):
            size = min(order_size, limit - side.sign * inventory)
            target = side_targets[side_index] if size > 0 else np.full(i1 - i0, _NO_ORDER, dtype=np.int64)
            side_targets[side_index] = target
            if len(target):
                changes = (target[1:] != target[:-1]).nonzero()[0] + 1
                first_new = int(target[0]) != side.last_target or (side.live is None and not side.pending and int(target[0]) != _NO_ORDER)
                sent = np.concatenate((_ZERO_INDEX, changes)) if first_new else changes
            else:
                sent = _NO_INDEX
            carried = np.array(side.pending, dtype=np.int64).reshape(-1, 4)
            side_px = np.concatenate((carried[:, 1], target[sent]))
            side_t = np.concatenate((carried[:, 0], decision_t[i0 + sent] + latency_ns))
            new_size = np.where(target[sent] == _NO_ORDER, 0, size)
            for column, values in zip(columns, (side_t, side_px, np.concatenate((carried[:, 2], new_size)), np.concatenate((carried[:, 3], i0 + sent)), np.full(len(side_t), side_index, dtype=np.int64))):
                column.append(values)
            live_px = _NO_ORDER if side.live is None else side.live.px
            timelines.append((np.concatenate(((_NO_ORDER,), side_t)), np.concatenate(((live_px,), side_px))))

        a_t, a_px, a_qty, a_dec, a_side = (np.concatenate(column) for column in columns)
        arrival_order = np.lexsort((a_side, a_dec, a_t))
        a_t, a_px, a_qty, a_dec, a_side = a_t[arrival_order], a_px[arrival_order], a_qty[arrival_order], a_dec[arrival_order], a_side[arrival_order]
        n_arrivals = len(a_t)

        j1 = int(trade_t.searchsorted(horizon, side="right"))
        window_t, window_px = trade_t[j0:j1], trade_px[j0:j1]
        touches = np.zeros(j1 - j0, dtype=bool)
        for side, (timeline_t, timeline_px) in zip(sides, timelines):
            live_px_at_trade = timeline_px[timeline_t.searchsorted(window_t, side="left") - 1]
            touches |= (side.sign * window_px) <= live_px_at_trade
        candidates = touches.nonzero()[0]
        in_horizon = (a_t <= horizon).nonzero()[0]

        n_candidates = len(candidates)
        event_times = np.concatenate((window_t[candidates], a_t[in_horizon]))
        event_kinds = np.concatenate((np.zeros(n_candidates, dtype=np.int8), np.ones(len(in_horizon), dtype=np.int8)))
        order = np.lexsort((np.concatenate((candidates, in_horizon)), event_kinds, event_times))
        trade_nbbo = quote_t.searchsorted(window_t[candidates], side="right") - 1
        arrival_nbbo = quote_t.searchsorted(a_t[in_horizon], side="right") - 1
        processed = np.zeros(n_arrivals, dtype=bool)

        restart_t = None
        restart_trade = None
        for event in order.tolist():
            if event < n_candidates:
                j = j0 + int(candidates[event])
                t, px, sz, k = int(trade_t[j]), int(trade_px[j]), int(trade_sz[j]), int(trade_nbbo[event])
                filled = False
                for side in sides:
                    live = side.live
                    mirrored = side.sign * px
                    if live is None or mirrored > live.px:
                        continue
                    available = model.shares_available(live, k, side.best, side.best_size, mirrored, sz)
                    quantity = min(live.rem, available, limit - side.sign * inventory)
                    if quantity > 0:
                        record_fill(t, side, quantity, side.sign * live.px, True)
                        live.rem -= quantity
                        if live.rem == 0:
                            side.live = None
                        filled = True
                        break
                if filled:
                    restart_t, restart_trade = t, j + 1
                    break
            else:
                a = int(in_horizon[event - n_candidates])
                processed[a] = True
                side = sides[int(a_side[a])]
                px = int(a_px[a])
                k = int(arrival_nbbo[event - n_candidates])
                if record_orders:
                    order_log.append((int(a_t[a]), side.sign, None if px == _NO_ORDER else side.sign * px, int(a_qty[a])))
                if px == _NO_ORDER:
                    side.live = None
                    continue
                n_orders += 1
                far = int(side.far[k])
                if px >= far:
                    quantity = min(int(a_qty[a]), int(side.far_size[k]), limit - side.sign * inventory)
                    side.live = None
                    if quantity > 0:
                        record_fill(int(a_t[a]), side, quantity, side.sign * far, False)
                    restart_t = int(a_t[a])
                    restart_trade = int(np.searchsorted(trade_t, restart_t, side="right"))
                    break
                placed = _Order(px, int(a_qty[a]))
                model.on_arrival(placed, k, side.best, side.best_size)
                side.live = placed

        if restart_t is None:
            leftover = ~processed
            for side_index, side in enumerate(sides):
                keep = leftover & (a_side == side_index)
                side.pending = [(int(t), int(px), int(qty), int(dec)) for t, px, qty, dec in zip(a_t[keep], a_px[keep], a_qty[keep], a_dec[keep])]
                if i1 > i0:
                    side.last_target = int(side_targets[side_index][-1])
            j0 = j1
            if i1 >= n_decisions:
                break
            i0 = i1
            chunk = min(_LARGEST_CHUNK, chunk * 2)
        else:
            i_next = int(np.searchsorted(decision_t, restart_t, side="right"))
            leftover = ~processed & (a_dec < i_next)
            for side_index, side in enumerate(sides):
                keep = leftover & (a_side == side_index)
                side.pending = [(int(t), int(px), int(qty), int(dec)) for t, px, qty, dec in zip(a_t[keep], a_px[keep], a_qty[keep], a_dec[keep])]
                if i_next > i0 and i1 > i0:
                    side.last_target = int(side_targets[side_index][i_next - 1 - i0])
            chunk = max(_SMALLEST_CHUNK, min(_LARGEST_CHUNK, 4 * (i_next - i0)))
            i0, j0 = i_next, int(restart_trade)

    return _assemble(market, rule_name, model, latency_ns, params, window, ledger, inventory, cash, order_log, n_orders, int(trade_sz.sum()))


def _assemble(market, rule_name, model, latency_ns, params, window, ledger, inventory, cash, order_log, n_orders, market_volume) -> RunResult:
    rows = np.array(ledger, dtype=np.int64).reshape(-1, 7)
    t_ns, side, qty, price, maker, flow, inventory_after = (rows[:, column] for column in range(7))
    mid_doubled = mid_doubled_at(market, t_ns) if len(t_ns) else np.empty(0, dtype=np.int64)
    fills = Fills(t_ns, side, qty, price, maker.astype(bool), flow, mid_doubled, inventory_after)
    ticks, bps = {}, {}
    for horizon in MARKOUT_HORIZONS_S:
        ticks[horizon], bps[horizon] = markouts(market, t_ns, side, horizon, params.tick_units)
    end_index = int(np.searchsorted(market.quote_t, window[1], side="left")) - 1
    end_mid = int(market.mid_doubled[max(end_index, 0)]) if market.quote_t.size else 0
    return RunResult(rule_name, model.name, model.label, latency_ns, window, params, fills, ticks, bps, market_volume, inventory, cash, end_mid, order_log, n_orders)
