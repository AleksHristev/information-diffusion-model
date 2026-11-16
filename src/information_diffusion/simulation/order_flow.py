"""Synthetic order flow for verifying the market-making engine: hand-derived scenarios and an independent reference replay."""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.params import MARKOUT_HORIZONS_S, EngineParams
from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule1_fixed_spread import FixedSpread
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, AvellanedaStoikov
from information_diffusion.market_making.rule3_hawkes_intensity import HawkesIntensity
from information_diffusion.market_making.rule4_network_fair_value import NetworkFairValue
from information_diffusion.market_making.rule5_hawkes_network import HawkesNetwork
from information_diffusion.market_making.rule6_scheduled_avellaneda_stoikov import ScheduledAvellanedaStoikov

US, MS, S = 1_000, 1_000_000, 1_000_000_000
MICRO = 1_000_000
BUY, SELL = 1, -1




@dataclass
class Flow:
    """NBBO updates and trades, built up with quote() and trade(); times in integer nanoseconds, prices in dollars.

    Fields: quotes, rows (t_ns, bid, ask, bid_size, ask_size); trades, rows (t_ns, price, size).
    """

    quotes: list[tuple[int, float, float, int, int]] = field(default_factory=list)
    trades: list[tuple[int, float, int]] = field(default_factory=list)

    def quote(self, t_ns: int, bid: float, ask: float, bid_size: int = 500, ask_size: int = 500) -> "Flow":
        """Add an NBBO update at t_ns with bid and ask in dollars and sizes in shares (default 500). Returns the flow."""
        self.quotes.append((int(t_ns), bid, ask, bid_size, ask_size))
        return self

    def trade(self, t_ns: int, price: float, size: int) -> "Flow":
        """Add a trade at t_ns with price in dollars and size in shares. Returns the flow."""
        self.trades.append((int(t_ns), price, size))
        return self

    def as_arrays(self) -> dict:
        """Keyword arguments for MarketData.from_arrays, quotes and trades each sorted by time, timestamps in nanoseconds."""
        quotes, trades = sorted(self.quotes, key=lambda row: row[0]), sorted(self.trades, key=lambda row: row[0])
        return {"quote_t": np.array([row[0] for row in quotes], dtype=np.int64), "bid": np.array([row[1] for row in quotes], dtype=float),
                "ask": np.array([row[2] for row in quotes], dtype=float), "bid_size": np.array([row[3] for row in quotes], dtype=np.int64),
                "ask_size": np.array([row[4] for row in quotes], dtype=np.int64), "trade_t": np.array([row[0] for row in trades], dtype=np.int64),
                "trade_price": np.array([row[1] for row in trades], dtype=float), "trade_size": np.array([row[2] for row in trades], dtype=np.int64)}

    def end_ns(self) -> int:
        """One nanosecond after the last quote or trade time."""
        return max(max((row[0] for row in self.quotes), default=0), max((row[0] for row in self.trades), default=0)) + 1


@dataclass(frozen=True)
class FixedQuotes:
    """Test quoting rule with constant bid and ask prices (dollars), so a queue path can be derived by hand."""

    bid: float
    ask: float
    name: ClassVar[str] = "fixed_quotes"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Quote terms at each decision time: centre (bid + ask) / 2, zero inventory skew and half-spread (ask - bid) / 2; mid and window_end_ns are not used."""
        n = len(t_ns)
        return QuoteTerms(np.full(n, (self.bid + self.ask) / 2.0), np.zeros(n), np.full(n, (self.ask - self.bid) / 2.0))


@dataclass(frozen=True)
class Scenario:
    """A flow, an engine configuration and the hand-derived result.

    Fields: name; prop, the property tested; flow; rule; fill_model ("optimistic" or "pessimistic"); latency_ns;
    fills, rows (t_ns, side, qty, price in dollars, maker); params; window (default from the first quote to the end of
    the flow); markouts, expected mark-outs in ticks by horizon in seconds (None where undefined); cash_units, the
    expected cash in micro-dollars; expected_orders, rows (t_ns, side, price or None for a cancel, qty).
    """

    name: str
    prop: str
    flow: Flow
    rule: object
    fill_model: str
    latency_ns: int
    fills: tuple[tuple[int, int, int, float, bool], ...]
    params: EngineParams = EngineParams()
    window: tuple[int, int] | None = None
    markouts: dict[int, tuple[float | None, ...]] | None = None
    cash_units: int | None = None
    expected_orders: tuple[tuple[int, int, float | None, int], ...] | None = None

    def resolved_window(self) -> tuple[int, int]:
        """Trading window (start_ns, end_ns): the given window, or from the first quote to the end of the flow."""
        return self.window if self.window is not None else (self.flow.quotes[0][0], self.flow.end_ns())

    def final_inventory(self) -> int:
        """Inventory in shares after all expected fills."""
        return sum(side * qty for _, side, qty, _, _ in self.fills)


def _touch_flow() -> Flow:
    return Flow().quote(0, 100.00, 100.02)


def scenarios() -> list[Scenario]:
    """All hand-derived scenarios: queue position, fills inside the spread, latency, inventory limits, mark-outs and fees; each prop states the property tested."""
    cases: list[Scenario] = []
    r1 = FixedSpread(0.01)

    flow = _touch_flow().trade(1 * MS, 100.00, 200).trade(2 * MS, 100.00, 400)
    cases.append(Scenario("queue_touch_fills_optimistic", "queue: optimistic fills on touch", flow, r1, "optimistic", 0, ((1 * MS, BUY, 100, 100.00, True),)))
    cases.append(Scenario("queue_behind_displayed_pessimistic", "queue: pessimistic waits for the 500 displayed ahead", flow, r1, "pessimistic", 0, ((2 * MS, BUY, 100, 100.00, True),)))

    flow = _touch_flow().trade(1 * MS, 100.00, 499)
    cases.append(Scenario("queue_not_exhausted_pessimistic", "queue: 499 of 500 ahead consumed, no pessimistic fill", flow, r1, "pessimistic", 0, ()))
    cases.append(Scenario("queue_not_exhausted_optimistic", "queue: same flow fills optimistic", flow, r1, "optimistic", 0, ((1 * MS, BUY, 100, 100.00, True),)))

    inside = FixedSpread(0.005)
    flow = Flow().quote(0, 100.00, 100.03).trade(1 * MS, 100.01, 30).trade(2 * MS, 100.00, 50).trade(3 * MS, 100.01, 100)
    expected = ((1 * MS, BUY, 30, 100.01, True), (2 * MS, BUY, 50, 100.01, True), (3 * MS, BUY, 20, 100.01, True))
    cases.append(Scenario("inside_spread_partial_pessimistic", "fills inside the spread at our price, no queue", flow, inside, "pessimistic", 0, expected))
    cases.append(Scenario("inside_spread_partial_optimistic", "fills inside the spread at our price, no queue", flow, inside, "optimistic", 0, expected))

    behind = FixedQuotes(100.01, 100.05)
    flow = (Flow().quote(0, 100.02, 100.04).trade(1 * MS, 100.01, 100).quote(2 * MS, 100.01, 100.03, 300, 500).trade(3 * MS, 100.01, 250).trade(4 * MS, 100.01, 100).trade(5 * MS, 100.00, 1000))
    cases.append(Scenario("behind_touch_level_reached_pessimistic", "behind the touch: unknown queue, level reached, partial, then through", flow, behind, "pessimistic", 0, ((4 * MS, BUY, 50, 100.01, True), (5 * MS, BUY, 50, 100.01, True))))
    cases.append(Scenario("behind_touch_optimistic", "behind the touch fills optimistic on touch, then a new order at the next NBBO update", flow, behind, "optimistic", 0, ((1 * MS, BUY, 100, 100.01, True), (3 * MS, BUY, 100, 100.01, True))))

    drain = FixedQuotes(100.00, 100.05)
    flow = (Flow().quote(0, 100.00, 100.03).trade(1 * MS, 100.00, 200).quote(2 * MS, 99.99, 100.01, 400, 400).trade(3 * MS, 99.99, 150))
    cases.append(Scenario("level_cleared_pessimistic", "NBBO through the level clears the queue", flow, drain, "pessimistic", 0, ((3 * MS, BUY, 100, 100.00, True),)))
    cases.append(Scenario("level_cleared_optimistic", "same flow, optimistic fills on touch and again after the next order", flow, drain, "optimistic", 0, ((1 * MS, BUY, 100, 100.00, True), (3 * MS, BUY, 100, 100.00, True))))

    return cases + _skew_scenarios() + _latency_scenarios() + _inventory_scenarios() + _markout_scenarios() + _fee_scenarios()


def _skew_scenarios() -> list[Scenario]:
    params = ASParams(gamma=0.01, sigma=0.01, k=0.01 / math.expm1(0.01 * 0.02))
    flow = Flow().quote(0, 99.99, 100.01).trade(1 * S, 99.97, 100).quote(2 * S, 99.99, 100.01)
    orders = ((0, BUY, 99.97, 100), (0, SELL, 100.03, 100), (2 * S, BUY, 99.97, 100), (2 * S, SELL, 100.02, 100))
    return [Scenario("avellaneda_stoikov_inventory_skew", "rule 2: inventory skews both quotes after a fill", flow, AvellanedaStoikov(params), "optimistic", 0, ((1 * S, BUY, 100, 99.97, True),), window=(0, 100 * S), expected_orders=orders)]


def _latency_scenarios() -> list[Scenario]:
    cases: list[Scenario] = []
    grid = (0, 100 * US, 500 * US, 1 * MS, 5 * MS, 10 * MS)
    r1 = FixedSpread(0.01)
    for model in ("pessimistic", "optimistic"):
        for latency in grid:
            after = _touch_flow().trade(latency + 1 * US, 100.00, 1000)
            cases.append(Scenario(f"latency_trade_after_arrival_{model}_{latency}", f"latency {latency} ns: trade after arrival fills", after, r1, model, latency, ((latency + 1 * US, BUY, 100, 100.00, True),)))
            at_arrival = _touch_flow().trade(latency, 100.00, 1000)
            cases.append(Scenario(f"latency_trade_at_arrival_{model}_{latency}", f"latency {latency} ns: trade at arrival time misses", at_arrival, r1, model, latency, ()))
            if latency > 0:
                before = _touch_flow().trade(latency - 1 * US, 100.00, 1000)
                cases.append(Scenario(f"latency_trade_before_arrival_{model}_{latency}", f"latency {latency} ns: trade before arrival misses", before, r1, model, latency, ()))

        moved = _touch_flow().trade(500 * US, 100.00, 1000).quote(1 * MS, 100.02, 100.04)
        for latency in grid:
            if latency < 500 * US:
                expected = ((500 * US, BUY, 100, 100.00, True),)
            elif latency == 500 * US:
                expected = ()
            else:
                expected = ((latency, SELL, 100, 100.02, False),)
            cases.append(Scenario(f"latency_before_price_move_{model}_{latency}", f"latency {latency} ns: order placed before a price move", moved, r1, model, latency, expected, window=(0, 30 * MS)))

        stale = _touch_flow().quote(20 * MS, 99.90, 99.92).trade(21 * MS, 99.91, 100)
        picked_off_from = {1 * MS, 5 * MS, 10 * MS}
        for latency in grid:
            expected = ((21 * MS, BUY, 100, 100.00, True),) if latency in picked_off_from else ()
            cases.append(Scenario(f"latency_stale_pickoff_{model}_{latency}", f"latency {latency} ns: stale bid picked off", stale, r1, model, latency, expected))
    return cases


def _inventory_scenarios() -> list[Scenario]:
    cases: list[Scenario] = []
    r1 = FixedSpread(0.01)
    limit_200 = EngineParams(inventory_limit=200, order_size=100)
    flow = (_touch_flow().trade(1 * MS, 100.00, 100).quote(2 * MS, 100.00, 100.02).trade(3 * MS, 100.00, 150).quote(4 * MS, 100.00, 100.02).trade(5 * MS, 100.00, 100).trade(6 * MS, 100.02, 100))
    expected = ((1 * MS, BUY, 100, 100.00, True), (3 * MS, BUY, 100, 100.00, True), (6 * MS, SELL, 100, 100.02, True))
    cases.append(Scenario("limit_long_stops_bids", "inventory limit: bid withdrawn at +200, ask keeps quoting", flow, r1, "optimistic", 0, expected, limit_200, expected_orders=((0, BUY, 100.00, 100), (0, SELL, 100.02, 100), (2 * MS, BUY, 100.00, 100), (4 * MS, BUY, None, 0))))

    limit_150 = EngineParams(inventory_limit=150, order_size=100)
    flow = _touch_flow().trade(1 * MS, 100.00, 100).quote(2 * MS, 100.00, 100.02).trade(3 * MS, 100.00, 150).quote(4 * MS, 100.00, 100.02).trade(5 * MS, 100.00, 100)
    cases.append(Scenario("limit_partial_headroom", "inventory limit: order sized down to headroom, fill capped at limit", flow, r1, "optimistic", 0, ((1 * MS, BUY, 100, 100.00, True), (3 * MS, BUY, 50, 100.00, True)), limit_150))

    limit_100 = EngineParams(inventory_limit=100, order_size=100)
    flow = _touch_flow().quote(1500 * US, 100.01, 100.03).trade(2 * MS, 100.00, 100).trade(3 * MS, 100.01, 100).quote(4 * MS, 100.01, 100.03).trade(5 * MS, 100.01, 100)
    cases.append(Scenario("limit_in_flight_order_capped", "in-flight order cannot breach the limit", flow, r1, "optimistic", 1 * MS, ((2 * MS, BUY, 100, 100.00, True),), limit_100))

    flow = _touch_flow().trade(1 * MS, 100.02, 100).quote(2 * MS, 100.00, 100.02).trade(3 * MS, 100.02, 100).trade(4 * MS, 100.00, 100)
    cases.append(Scenario("limit_short_stops_asks", "inventory limit: ask withdrawn at -100, bid keeps quoting", flow, r1, "optimistic", 0, ((1 * MS, SELL, 100, 100.02, True), (4 * MS, BUY, 100, 100.00, True)), limit_100, expected_orders=((0, BUY, 100.00, 100), (0, SELL, 100.02, 100), (2 * MS, SELL, None, 0))))
    return cases


def _markout_flow(extra: Flow | None = None) -> Flow:
    flow = Flow().quote(0, 100.00, 100.02).quote(1500 * MS, 100.05, 100.07).quote(2500 * MS, 100.10, 100.12).quote(6500 * MS, 99.90, 99.92).quote(31 * S, 99.80, 99.82)
    if extra is not None:
        flow.trades.extend(extra.trades)
        flow.quotes.extend(extra.quotes)
    return flow


def _markout_scenarios() -> list[Scenario]:
    cases: list[Scenario] = []
    r1 = FixedSpread(0.01)
    flow = _markout_flow(Flow().trade(1 * S, 100.00, 100))
    cases.append(Scenario("markout_buy", "mark-outs: buy, positive when the mid rises", flow, r1, "optimistic", 0, ((1 * S, BUY, 100, 100.00, True),), markouts={1: (5.0,), 5: (10.0,), 30: (-20.0,)}))
    flow = _markout_flow(Flow().trade(1 * S, 100.02, 100))
    cases.append(Scenario("markout_sell", "mark-outs: sell, positive when the mid falls", flow, r1, "optimistic", 0, ((1 * S, SELL, 100, 100.02, True),), markouts={1: (-5.0,), 5: (-10.0,), 30: (20.0,)}))
    flow = Flow().quote(0, 100.00, 100.02).quote(6500 * MS, 99.90, 99.92).trade(28 * S, 99.90, 100).quote(31 * S, 99.80, 99.82)
    cases.append(Scenario("markout_past_end_of_data", "mark-outs: horizons past the last quote are undefined", flow, r1, "optimistic", 0, ((28 * S, BUY, 100, 99.90, True),), window=(0, 40 * S), markouts={1: (0.0,), 5: (None,), 30: (None,)}))
    flow = Flow().quote(0, 100.00, 100.02).quote(6500 * MS, 99.90, 99.92).trade(28 * S, 99.90, 100).quote(31 * S, 99.80, 99.82).quote(60 * S, 99.70, 99.72)
    cases.append(Scenario("markout_past_window_end_defined", "mark-outs: quotes after the window end are used", flow, r1, "optimistic", 0, ((28 * S, BUY, 100, 99.90, True),), window=(0, 30 * S), markouts={1: (0.0,), 5: (-10.0,), 30: (-10.0,)}))
    return cases


def _fee_scenarios() -> list[Scenario]:
    cases: list[Scenario] = []
    r1 = FixedSpread(0.01)
    flow = _touch_flow().trade(1 * MS, 100.00, 100).trade(2 * MS, 100.02, 100)
    expected = ((1 * MS, BUY, 100, 100.00, True), (2 * MS, SELL, 100, 100.02, True))
    cases.append(Scenario("fees_maker_base", "maker fills earn the add rebate", flow, r1, "optimistic", 0, expected, cash_units=-10_000_000_000 + 200_000 + 10_002_000_000 + 200_000))
    cases.append(Scenario("fees_maker_sensitivity", "maker fills at the 0.25 cent rebate", flow, r1, "optimistic", 0, expected, EngineParams(add_rebate_cents=0.25), cash_units=-10_000_000_000 + 250_000 + 10_002_000_000 + 250_000))

    crossing_buy = FixedQuotes(100.03, 100.05)
    flow = _touch_flow()
    cases.append(Scenario("fees_taker_buy", "a marketable bid takes at the ask and pays the take fee", flow, crossing_buy, "pessimistic", 0, ((0, BUY, 100, 100.02, False),), window=(0, 1 * MS), cash_units=-10_002_000_000 - 300_000))
    crossing_sell = FixedQuotes(99.95, 99.99)
    cases.append(Scenario("fees_taker_sell", "a marketable ask takes at the bid and pays the take fee", flow, crossing_sell, "optimistic", 0, ((0, SELL, 100, 100.00, False),), window=(0, 1 * MS), cash_units=10_000_000_000 - 300_000))
    flow = Flow().quote(0, 100.00, 100.02, 500, 50)
    cases.append(Scenario("fees_taker_capped_by_displayed_size", "a taker is limited to the displayed size", flow, crossing_buy, "optimistic", 0, ((0, BUY, 50, 100.02, False),), window=(0, 1 * MS), cash_units=-5_001_000_000 - 150_000))
    flow = _touch_flow()
    cases.append(Scenario("fees_taker_sensitivity", "the 0.25 cent take fee", flow, crossing_buy, "pessimistic", 0, ((0, BUY, 100, 100.02, False),), EngineParams(take_fee_cents=0.25), window=(0, 1 * MS), cash_units=-10_002_000_000 - 250_000))
    return cases




def _floor_ticks(value: float, tick: float) -> int:
    return math.floor(value / tick + 1e-9)


def _ceil_ticks(value: float, tick: float) -> int:
    return math.ceil(value / tick - 1e-9)


def reference_quote_fn(rule, params: EngineParams, window_end_ns: int):
    """Independent scalar version of a quoting rule's formulas, for the reference replay.

    Arguments: rule is one of the six quoting rules or FixedQuotes; params the engine parameters; window_end_ns the end
    of the trading window.
    Returns: a function (t_ns, doubled mid in micro-dollars, inventory) -> (bid, ask) in micro-dollars, rounded outwards
    to the tick; raises TypeError for an unknown rule.
    """
    tick = params.tick
    tick_units = params.tick_units
    if isinstance(rule, FixedSpread):
        half = rule.half_spread

        def terms(t_ns, mid):
            return mid, 0.0, half
    elif isinstance(rule, (AvellanedaStoikov, HawkesIntensity, NetworkFairValue, HawkesNetwork, ScheduledAvellanedaStoikov)):
        p: ASParams = rule.params

        def asof(times: list[int], values: list[float], t_ns: int, default: float) -> float:
            index = bisect.bisect_right(times, t_ns) - 1
            return default if index < 0 else values[index]

        intensity_t, intensity_v, baseline, cap, exponent = None, None, None, None, None
        if isinstance(rule, (HawkesIntensity, HawkesNetwork)):
            intensity_t, intensity_v, baseline = list(map(int, rule.intensity_t_ns)), list(map(float, rule.intensity)), float(rule.baseline_intensity)
            cap, exponent = rule.cap, rule.exponent
        fv_t, fv_v = None, None
        if isinstance(rule, (NetworkFairValue, HawkesNetwork)):
            fv_t, fv_v = list(map(int, rule.fair_value_t_ns)), list(map(float, rule.fair_value))

        def schedule(t_ns: int) -> float:
            best = 1.0
            for t0 in rule.t0s_ns:
                u = (t_ns - t0) / 1e9
                if -rule.pre_s <= u < rule.post_s:
                    best = max(best, rule.factor)
                elif rule.post_s <= u < rule.decay_end_s:
                    best = max(best, rule.factor - (rule.factor - 1.0) * (u - rule.post_s) / (rule.decay_end_s - rule.post_s))
            return best

        def terms(t_ns, mid):
            time_left = max((window_end_ns - t_ns) / 1e9, 0.0)
            variance_term = p.gamma * p.sigma * p.sigma * time_left
            half = 0.5 * variance_term + math.log1p(p.gamma / p.k) / p.gamma
            center = mid
            if fv_t is not None:
                center = asof(fv_t, fv_v, t_ns, mid)
            if intensity_t is not None:
                level = asof(intensity_t, intensity_v, t_ns, baseline)
                half = half * min(max((level / baseline) ** exponent, 1.0), cap)
            if isinstance(rule, ScheduledAvellanedaStoikov):
                half = half * schedule(t_ns)
            return center, variance_term, half
    elif isinstance(rule, FixedQuotes):
        def terms(t_ns, mid):
            return (rule.bid + rule.ask) / 2.0, 0.0, (rule.ask - rule.bid) / 2.0
    else:
        raise TypeError(f"no reference formula for {type(rule).__name__}")

    def quote(t_ns: int, mid_doubled: int, inventory: int) -> tuple[int, int]:
        center, skew, half = terms(t_ns, mid_doubled / (2.0 * MICRO))
        reservation = center - inventory * skew
        half = max(half, tick / 2.0)
        return _floor_ticks(reservation - half, tick) * tick_units, _ceil_ticks(reservation + half, tick) * tick_units

    return quote


@dataclass
class ReferenceResult:
    """Result of reference_replay. Fields: fills, rows (t_ns, side, qty, price in micro-dollars, maker); inventory after each fill; cash_units in micro-dollars; orders, rows (t_ns, side, price or None, qty)."""

    fills: list[tuple[int, int, int, int, bool]]
    inventory: list[int]
    cash_units: int
    orders: list[tuple[int, int, int | None, int]]


def reference_replay(flow: Flow, quote_fn, fill_model: str, latency_ns: int, params: EngineParams, window: tuple[int, int]) -> ReferenceResult:
    """Simple event-by-event replay of a quoting strategy, sharing no matching code with the engine.

    Events at equal times are processed as NBBO update, then trade, then order arrival. Orders arrive latency_ns after
    the quote that triggered them; marketable orders take the displayed size, and resting orders fill under the given
    fill model with queue position tracked from displayed size.
    Arguments: flow is the Flow; quote_fn from reference_quote_fn; fill_model "optimistic" or "pessimistic"; latency_ns
    the order latency in nanoseconds; params the engine parameters; window (start_ns, end_ns).
    Returns: a ReferenceResult.
    """
    start, end = window
    to_units = lambda x: round(x * MICRO)
    quotes = [(t, to_units(b), to_units(a), bs, az) for t, b, a, bs, az in flow.quotes if to_units(a) > to_units(b) > 0]
    queue = []
    counter = 0
    for row in sorted(quotes, key=lambda r: r[0]):
        queue.append((row[0], 0, counter, "quote", row))
        counter += 1
    for t, price, size in sorted(flow.trades, key=lambda r: r[0]):
        queue.append((t, 1, counter, "trade", (to_units(price), size)))
        counter += 1
    heapq.heapify(queue)

    inf = float("inf")
    bid = ask = bid_size = ask_size = 0
    inventory, cash = 0, 0
    live: dict[int, dict | None] = {BUY: None, SELL: None}
    in_flight = {BUY: 0, SELL: 0}
    last_target: dict[int, int | None] = {BUY: None, SELL: None}
    fills: list[tuple[int, int, int, int, bool]] = []
    path: list[int] = []
    orders: list[tuple[int, int, int | None, int]] = []
    rebate, fee = params.add_rebate_units, params.take_fee_units

    def apply_fill(t: int, side: int, qty: int, price: int, maker: bool) -> None:
        nonlocal inventory, cash
        inventory += side * qty
        cash += -side * qty * price + qty * (rebate if maker else -fee)
        fills.append((t, side, qty, price, maker))
        path.append(inventory)

    while queue:
        t, _, _, kind, row = heapq.heappop(queue)
        if kind == "quote":
            _, bid, ask, bid_size, ask_size = row
            for side, order in live.items():
                if order is None:
                    continue
                best = bid if side == BUY else ask
                size = bid_size if side == BUY else ask_size
                if (side == BUY and best < order["price"]) or (side == SELL and best > order["price"]):
                    order["ahead"] = 0
                elif order["ahead"] == inf and best == order["price"]:
                    order["ahead"] = size
            if not (start <= t < end):
                continue
            bid_quote, ask_quote = quote_fn(t, bid + ask, inventory)
            for side, target_price in ((BUY, bid_quote), (SELL, ask_quote)):
                size = min(params.order_size, params.inventory_limit - side * inventory)
                target = target_price if size > 0 else None
                needs_order = target != last_target[side] or (target is not None and live[side] is None and in_flight[side] == 0)
                if needs_order:
                    arrival = t + latency_ns
                    heapq.heappush(queue, (arrival, 2, counter, "arrival", (side, target, size if target is not None else 0)))
                    counter += 1
                    in_flight[side] += 1
                last_target[side] = target
        elif kind == "trade":
            if not (start <= t < end):
                continue
            price, size = row
            for side in (BUY, SELL):
                order = live[side]
                if order is None:
                    continue
                touched = price <= order["price"] if side == BUY else price >= order["price"]
                if not touched:
                    continue
                if fill_model == "optimistic":
                    available = size
                elif order["ahead"] == inf:
                    available = 0
                elif price == order["price"]:
                    used = min(order["ahead"], size)
                    order["ahead"] -= used
                    available = size - used
                else:
                    available = size if order["ahead"] == 0 else 0
                quantity = min(order["remaining"], available, params.inventory_limit - side * inventory)
                if quantity > 0:
                    apply_fill(t, side, quantity, order["price"], True)
                    order["remaining"] -= quantity
                    if order["remaining"] == 0:
                        live[side] = None
                    break
        else:
            side, price, size = row
            in_flight[side] -= 1
            if t >= end:
                continue
            orders.append((t, side, price, size))
            if price is None:
                live[side] = None
                continue
            marketable = price >= ask if side == BUY else price <= bid
            if marketable:
                displayed = ask_size if side == BUY else bid_size
                quantity = min(size, displayed, params.inventory_limit - side * inventory)
                live[side] = None
                if quantity > 0:
                    apply_fill(t, side, quantity, ask if side == BUY else bid, False)
                continue
            best = bid if side == BUY else ask
            displayed = bid_size if side == BUY else ask_size
            ahead = 0 if (price > best if side == BUY else price < best) else (displayed if price == best else inf)
            live[side] = {"price": price, "remaining": size, "ahead": ahead}
    return ReferenceResult(fills, path, cash, orders)


def reference_markouts(flow: Flow, fills: list[tuple[int, int, int, int, bool]], params: EngineParams) -> dict[int, list[float | None]]:
    """Mark-outs of each fill in ticks, side times the midpoint change from the fill to each horizon, computed independently of the engine.

    Arguments: flow is the Flow; fills rows as in ReferenceResult; params the engine parameters.
    Returns: a dict from horizon (seconds) to one value per fill, None when the horizon is past the last quote.
    """
    to_units = lambda x: round(x * MICRO)
    quotes = [(t, to_units(b) + to_units(a)) for t, b, a, _, _ in sorted(flow.quotes, key=lambda r: r[0]) if to_units(a) > to_units(b) > 0]
    times = [row[0] for row in quotes]
    result: dict[int, list[float | None]] = {h: [] for h in MARKOUT_HORIZONS_S}
    for t, side, _, _, _ in fills:
        m0 = quotes[bisect.bisect_right(times, t) - 1][1]
        for horizon in MARKOUT_HORIZONS_S:
            target = t + horizon * S
            if not times or target > times[-1]:
                result[horizon].append(None)
            else:
                result[horizon].append(side * (quotes[bisect.bisect_right(times, target) - 1][1] - m0) / (2.0 * params.tick_units))
    return result


@dataclass(frozen=True)
class RuleQuoteCase:
    """Decision times and mids for one rule, and the hand-derived quotes at one inventory level.

    Fields: name; rule; t_ns, the decision times; mid, the midpoints in dollars; window_end_ns; inventory in shares;
    bid and ask, the expected quotes in dollars.
    """

    name: str
    rule: object
    t_ns: tuple[int, ...]
    mid: tuple[float, ...]
    window_end_ns: int
    inventory: int
    bid: tuple[float, ...]
    ask: tuple[float, ...]


def _derived_as_params(h: float = 0.03) -> ASParams:
    return ASParams(gamma=1.0, sigma=0.01, k=1.0 / math.expm1(h))


def rule_quote_cases() -> list[RuleQuoteCase]:
    """Hand-worked quote cases for the six quoting rules, with variance term gamma sigma^2 tau = 1e-4 tau. Returns a list of RuleQuoteCase."""
    params = _derived_as_params()
    sec = (lambda *times: tuple(int(round(x * S)) for x in times))
    cases = [
        RuleQuoteCase("rule1 fixed spread", FixedSpread(0.03), sec(0, 10), (100.01, 100.015), 60 * S, 0, (99.98, 99.98), (100.04, 100.05)),
        RuleQuoteCase("rule1 inventory has no effect", FixedSpread(0.03), sec(0), (100.01,), 60 * S, 500, (99.98,), (100.04,)),
        RuleQuoteCase("rule2 at q = 0", AvellanedaStoikov(params), sec(900, 990), (100.0, 100.0), 1000 * S, 0, (99.96, 99.96), (100.04, 100.04)),
        RuleQuoteCase("rule2 at q = +2", AvellanedaStoikov(params), sec(900), (100.0,), 1000 * S, 2, (99.94,), (100.02,)),
        RuleQuoteCase("rule2 at q = -1", AvellanedaStoikov(params), sec(900), (100.0,), 1000 * S, -1, (99.97,), (100.05,)),
    ]
    intensity_t, intensity = sec(900, 950, 980), (4.0, 100.0, 0.25)
    rule3 = HawkesIntensity(params, np.array(intensity_t, dtype=np.int64), np.array(intensity), baseline_intensity=1.0)
    cases.append(RuleQuoteCase("rule3 intensity multiplier", rule3, sec(850, 900, 950, 980), (100.0, 100.0, 100.0, 100.0), 1000 * S, 0,
                               (99.96, 99.93, 99.90, 99.96), (100.04, 100.07, 100.10, 100.04)))
    fv_t, fv = sec(900, 950), (100.25, 99.75)
    rule4 = NetworkFairValue(params, np.array(fv_t, dtype=np.int64), np.array(fv))
    cases.append(RuleQuoteCase("rule4 network fair value", rule4, sec(850, 900, 950), (100.0, 100.0, 100.0), 1000 * S, 0, (99.96, 100.21, 99.71), (100.04, 100.29, 99.79)))
    rule5 = HawkesNetwork(params, np.array(sec(900, 950), dtype=np.int64), np.array([4.0, 100.0]), baseline_intensity=1.0, fair_value_t_ns=np.array(fv_t, dtype=np.int64), fair_value=np.array(fv))
    cases.append(RuleQuoteCase("rule5 both inputs", rule5, sec(850, 900, 950), (100.0, 100.0, 100.0), 1000 * S, 0, (99.96, 100.18, 99.65), (100.04, 100.32, 99.85)))
    rule6 = ScheduledAvellanedaStoikov(params, (500 * S,))
    times = sec(439, 440, 619, 620, 860, 1099, 1100)
    cases.append(RuleQuoteCase("rule6 schedule", rule6, times, tuple(100.0 for _ in times), 1200 * S, 0,
                               (99.93, 99.86, 99.88, 99.88, 99.92, 99.96, 99.96), (100.07, 100.14, 100.12, 100.12, 100.08, 100.04, 100.04)))
    sensitivity = ScheduledAvellanedaStoikov(ASParams(gamma=1.0, sigma=0.0, k=1.0 / math.expm1(0.03)), (500 * S,), factor=3.0, decay_end_s=900.0)
    cases.append(RuleQuoteCase("rule6 sensitivity factor 3 decay end 900", sensitivity, sec(559, 1300), (100.0, 100.0), 2000 * S, 0, (99.91, 99.96), (100.09, 100.04)))
    return cases


FOMC_T0_EST_NS = 1_769_626_800 * S
FOMC_T0_EDT_NS = 1_781_719_200 * S


def fomc_schedule_cases() -> list[dict]:
    """Test cases for the scheduled rule around a decision at t0 with a second shock 30 minutes later, on one standard-time and one daylight-time date.

    The dates are Federal Open Market Committee (FOMC) decision days, with t0 the 14:00 New York statement and the
    second shock the 14:30 press conference. sigma = 0, so the base half-spread is 0.03.
    Returns: one dict per date with label, params, t0_ns, times, expected_multiplier, bid, ask and repeat_ns, the time
    of the second shock.
    """
    params = ASParams(gamma=1.0, sigma=0.0, k=1.0 / math.expm1(0.03))
    rows = []
    for label, t0_ns, repeat_utc_offset_s in (("EST", FOMC_T0_EST_NS, 19 * 3600 + 30 * 60), ("EDT", FOMC_T0_EDT_NS, 18 * 3600 + 30 * 60)):
        offsets_from_t0 = (-60, 600, 30 * 60 - 60, 30 * 60, 32 * 60, 35 * 60, 40 * 60)
        times = tuple(t0_ns + offset * S for offset in offsets_from_t0)
        expected_multiplier = (2.0, 1.0, 2.0, 2.0, 2.0, 1.625, 1.0)
        bid = (99.94, 99.97, 99.94, 99.94, 99.94, 99.95, 99.97)
        ask = (100.06, 100.03, 100.06, 100.06, 100.06, 100.05, 100.03)
        date_utc_midnight = t0_ns - (t0_ns % (86400 * S))
        rows.append({"label": label, "params": params, "t0_ns": t0_ns, "times": times, "expected_multiplier": expected_multiplier, "bid": bid, "ask": ask, "repeat_ns": date_utc_midnight + repeat_utc_offset_s * S})
    return rows


def check_scenario(result, scenario: Scenario) -> dict[str, bool]:
    """Exact comparison of one engine run with its hand-derived scenario; mark-outs allow rounding error only.

    Arguments: result is the engine result; scenario the Scenario.
    Returns: a dict of named checks (fills, inventory_path, final_inventory, accounting, no_fill_beyond_limit, and
    markouts, ledger and orders when the scenario states them), each True when it passes.
    """
    fills = result.fills
    got_fills = [(int(t), int(side), int(qty), int(price), bool(maker)) for t, side, qty, price, maker in zip(fills.t_ns, fills.side, fills.qty, fills.price, fills.maker)]
    expected_fills = [(t, side, qty, round(price * MICRO), maker) for t, side, qty, price, maker in scenario.fills]
    expected_inventory, running = [], 0
    for _, side, qty, _, _ in scenario.fills:
        running += side * qty
        expected_inventory.append(running)
    checks = {"fills": got_fills == expected_fills, "inventory_path": [int(x) for x in fills.inventory_after] == expected_inventory,
              "final_inventory": result.final_inventory == scenario.final_inventory(), "accounting": result.reconciles(), "no_fill_beyond_limit": (not expected_inventory) or int(np.abs(fills.inventory_after).max()) <= scenario.params.inventory_limit}
    if scenario.markouts is not None:
        markouts_ok = True
        for horizon, expected in scenario.markouts.items():
            observed = result.markout_ticks[horizon]
            markouts_ok &= len(observed) == len(expected)
            for got, want in zip(observed, expected):
                markouts_ok &= (bool(np.isnan(got)) if want is None else bool(np.isfinite(got)) and abs(float(got) - want) <= 1e-9)
        checks["markouts"] = bool(markouts_ok)
    if scenario.cash_units is not None:
        checks["ledger"] = result.cash_units == scenario.cash_units
    if scenario.expected_orders is not None:
        expected_orders = [(t, side, None if price is None else round(price * MICRO), qty) for t, side, price, qty in scenario.expected_orders]
        checks["orders"] = [tuple(row) for row in result.orders] == expected_orders
    return checks


def reference_agrees(result, flow: Flow, rule, params: EngineParams) -> dict[str, bool]:
    """Compare an engine run with the independent reference replay on the same flow.

    Arguments: result is the engine result; flow the Flow; rule the quoting rule; params the engine parameters.
    Returns: a dict of checks fills, inventory_path, ledger and markouts, each True when engine and reference agree.
    """
    reference = reference_replay(flow, reference_quote_fn(rule, params, result.window[1]), result.fill_model, result.latency_ns, params, result.window)
    fills = result.fills
    got = [(int(t), int(side), int(qty), int(price), bool(maker)) for t, side, qty, price, maker in zip(fills.t_ns, fills.side, fills.qty, fills.price, fills.maker)]
    checks = {"fills": got == reference.fills, "inventory_path": [int(x) for x in fills.inventory_after] == reference.inventory, "ledger": result.cash_units == reference.cash_units}
    expected_markouts = reference_markouts(flow, reference.fills, params)
    markouts_ok = True
    for horizon, values in expected_markouts.items():
        observed = result.markout_ticks[horizon]
        markouts_ok &= len(observed) == len(values)
        for got_value, want in zip(observed, values):
            markouts_ok &= bool(np.isnan(got_value)) if want is None else bool(np.isfinite(got_value)) and abs(float(got_value) - want) <= 1e-9
    checks["markouts"] = bool(markouts_ok)
    return checks




def random_flow(seed: int, *, n_quotes: int = 3_000, tick: float = 0.01, start_price: float = 100.00) -> Flow:
    """Seeded random NBBO path and trades, including time ties, sub-penny prints, locked quotes and prints through the quotes.

    Arguments: seed the random seed; n_quotes the number of NBBO updates; tick the tick size in dollars; start_price in dollars.
    Returns: a Flow.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    flow = Flow()
    bid = round(start_price / tick)
    spread = 2
    t = 0
    for _ in range(n_quotes):
        t += int(rng.choice([0, 1, 50, 300, 1500, 4000, 20000]) * US)
        draw = rng.random()
        if draw < 0.25:
            bid += int(rng.choice([-2, -1, 1, 2]))
        elif draw < 0.35:
            spread = int(rng.integers(1, 5))
        locked = rng.random() < 0.01
        bid_price = bid * tick
        ask_price = (bid + (0 if locked else spread)) * tick
        flow.quote(t, round(bid_price, 2), round(ask_price, 2), int(rng.integers(1, 10)) * 100, int(rng.integers(1, 10)) * 100)
        if rng.random() < 0.6:
            trade_t = t + int(rng.choice([0, 1, 100, 900])) * US
            where = rng.random()
            if where < 0.4:
                price = bid * tick
            elif where < 0.8:
                price = (bid + spread) * tick
            elif where < 0.9:
                price = (bid + spread / 2.0) * tick
            elif where < 0.95:
                price = (bid - int(rng.integers(1, 4))) * tick
            else:
                price = (bid + spread + int(rng.integers(1, 4))) * tick
            flow.trade(trade_t, round(price, 4), int(rng.integers(1, 25)) * 10)
    return flow


def synthetic_rule_inputs(flow: Flow) -> dict:
    """Deterministic intensity and fair-value series on a one-second grid derived from the flow's NBBO. Returns a dict with grid_ns, intensity and fair_value."""
    quote_t = np.array([row[0] for row in sorted(flow.quotes, key=lambda r: r[0])], dtype=np.int64)
    bid = np.array([row[1] for row in sorted(flow.quotes, key=lambda r: r[0])])
    ask = np.array([row[2] for row in sorted(flow.quotes, key=lambda r: r[0])])
    grid = np.arange(int(quote_t[0]), int(quote_t[-1]) + S, S, dtype=np.int64)
    index = np.searchsorted(quote_t, grid, side="right") - 1
    mid = (bid[index] + ask[index]) / 2.0
    intensity = 1.0 + 0.5 * np.arange(len(grid))
    fair_value = mid + 0.01 * np.where(np.arange(len(grid)) % 2 == 0, 1.0, -1.0)
    return {"grid_ns": grid, "intensity": intensity, "fair_value": fair_value}


def standard_rules(flow: Flow, t0_ns: int) -> list:
    """The six quoting rules with fixed Avellaneda-Stoikov parameters (gamma 1e-4, sigma 0.05, k 100) and synthetic inputs from the flow.

    Arguments: flow is the Flow; t0_ns the shock time of the scheduled rule.
    Returns: a list of the six rule objects.
    """
    params = ASParams(gamma=1e-4, sigma=0.05, k=100.0)
    inputs = synthetic_rule_inputs(flow)
    return [FixedSpread(0.01), AvellanedaStoikov(params),
            HawkesIntensity(params, inputs["grid_ns"], inputs["intensity"], baseline_intensity=2.0),
            NetworkFairValue(params, inputs["grid_ns"], inputs["fair_value"]),
            HawkesNetwork(params, inputs["grid_ns"], inputs["intensity"], baseline_intensity=2.0, fair_value_t_ns=inputs["grid_ns"], fair_value=inputs["fair_value"]),
            ScheduledAvellanedaStoikov(params, (t0_ns,))]


def large_flow(n_quotes: int, *, seed: int = 11, tick: float = 0.01, start_price: float = 500.0) -> dict:
    """Large random NBBO and trade arrays for speed runs, about one update every 1.8 ms as in a liquid ETF.

    About 12% of updates change the best bid or ask; the rest change displayed size only.
    Arguments: n_quotes the number of NBBO updates; seed the random seed; tick in dollars; start_price in dollars.
    Returns: keyword arguments for MarketData.from_arrays.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    t = np.cumsum(rng.integers(1, 3_600_000, n_quotes))
    bid_ticks = int(round(start_price / tick)) + np.cumsum((rng.random(n_quotes) < 0.1) * rng.choice([-1, 1], n_quotes))
    spread_changes = np.flatnonzero(rng.random(n_quotes) < 0.02)
    spread_levels = rng.integers(1, 4, len(spread_changes) + 1)
    spread = spread_levels[np.searchsorted(spread_changes, np.arange(n_quotes), side="right")]
    bid = bid_ticks * tick
    ask = (bid_ticks + spread) * tick
    n_trades = n_quotes // 8
    trade_index = np.sort(rng.integers(0, n_quotes, n_trades))
    side = rng.random(n_trades) < 0.5
    trade_price = np.where(side, bid[trade_index], ask[trade_index])
    trade_t = (t[trade_index] + rng.integers(1, 500, n_trades)).astype(np.int64)
    in_time_order = np.argsort(trade_t, kind="stable")
    return {"quote_t": t.astype(np.int64), "bid": np.round(bid, 2), "ask": np.round(ask, 2), "bid_size": rng.integers(1, 40, n_quotes) * 100,
            "ask_size": rng.integers(1, 40, n_quotes) * 100, "trade_t": trade_t[in_time_order],
            "trade_price": np.round(trade_price, 2)[in_time_order], "trade_size": (rng.integers(1, 12, n_trades) * 100)[in_time_order]}
