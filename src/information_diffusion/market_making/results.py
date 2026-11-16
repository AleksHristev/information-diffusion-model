"""Output of one backtest run: the fill ledger, mark-outs, exact integer accounting and a summary in ticks, basis points and shares."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from information_diffusion.market_making.params import EngineParams, MARKOUT_HORIZONS_S, PRICE_SCALE


@dataclass(frozen=True)
class Fills:
    """Fill ledger with one row per fill.

    Fields: t_ns (nanoseconds), side (+1 buy, -1 sell), qty (shares), price (integer micro-dollars), maker flag,
    fee_flow (micro-dollars, positive for a rebate), mid_doubled (twice the mid at the fill) and inventory_after.
    """

    t_ns: np.ndarray
    side: np.ndarray
    qty: np.ndarray
    price: np.ndarray
    maker: np.ndarray
    fee_flow: np.ndarray
    mid_doubled: np.ndarray
    inventory_after: np.ndarray

    def __len__(self) -> int:
        return len(self.t_ns)

    @property
    def price_dollars(self) -> np.ndarray:
        """Return the fill prices in dollars."""
        return self.price / PRICE_SCALE


@dataclass(frozen=True)
class RunResult:
    """Result of one backtest run.

    Fields: rule, fill_model and fill_label; latency_ns; window in nanoseconds; params; fills; mark-outs in ticks and
    basis points by horizon in seconds; market volume in shares over the window; final inventory; cash in micro-dollars
    (kept for the internal reconciliation only); twice the end mid; the optional order log and the number of orders.
    """

    rule: str
    fill_model: str
    fill_label: str
    latency_ns: int
    window: tuple[int, int]
    params: EngineParams
    fills: Fills
    markout_ticks: dict[int, np.ndarray]
    markout_bps: dict[int, np.ndarray]
    market_volume: int
    final_inventory: int
    cash_units: int
    end_mid_doubled: int
    orders: list = field(default_factory=list)
    n_orders: int = 0

    def spread_captured_ticks(self) -> np.ndarray:
        """Return the spread captured s (m_fill - p) of each fill in ticks, positive for a maker capturing the half-spread."""
        f = self.fills
        return f.side * (f.mid_doubled - 2 * f.price) / (2.0 * self.params.tick_units)

    def spread_captured_bps(self) -> np.ndarray:
        """Return the spread captured 1e4 s (m_fill - p) / m_fill of each fill in basis points."""
        f = self.fills
        return 1e4 * f.side * (f.mid_doubled - 2 * f.price) / f.mid_doubled

    def inventory_path(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the inventory step function as the fill times and the inventory after each fill (starting from zero)."""
        return self.fills.t_ns, self.fills.inventory_after

    def time_weighted_inventory(self) -> tuple[float, float]:
        """Return the time-weighted mean and variance of inventory, in shares, over the window."""
        start, end = self.window
        duration = float(end - start)
        times = np.concatenate(([start], self.fills.t_ns, [end])).astype(float)
        levels = np.concatenate(([0], self.fills.inventory_after)).astype(float)
        widths = np.diff(times)
        mean = float((levels * widths).sum() / duration)
        return mean, float(((levels - mean) ** 2 * widths).sum() / duration)

    def reconcile(self) -> dict[str, int]:
        """Recompute the exact integer accounting from the ledger, for internal checks only.

        All amounts are integer micro-dollars. Twice the final wealth, 2 cash + final inventory x twice the end mid,
        equals spread + carry + 2 fees, where spread sums s q (twice the mid at the fill - 2 p), carry sums
        s q (twice the end mid - twice the mid at the fill) and fees sums the fee flows over the fills.
        Returns: a dict with the ledger inventory and cash, spread, carry, fees and twice the wealth (key "wealth").
        """
        f = self.fills
        side = f.side.astype(np.int64)
        qty = f.qty.astype(np.int64)
        signed = side * qty
        ledger_inventory = int(signed.sum())
        ledger_cash = int((-signed * f.price + f.fee_flow).sum())
        spread = int((signed * (f.mid_doubled - 2 * f.price)).sum())
        carry = int((signed * (self.end_mid_doubled - f.mid_doubled)).sum())
        fees = int(f.fee_flow.sum())
        wealth = 2 * self.cash_units + self.final_inventory * self.end_mid_doubled
        return {"ledger_inventory": ledger_inventory, "ledger_cash": ledger_cash, "spread": spread, "carry": carry, "fees": fees, "wealth": wealth}

    def reconciles(self) -> bool:
        """Check the integer accounting of the run.

        Returns: True when the ledger matches the running inventory and cash, the wealth decomposition of reconcile is
        exact, the inventory path ends at the final inventory and the inventory limit is respected.
        """
        check = self.reconcile()
        running_matches = check["ledger_inventory"] == self.final_inventory and check["ledger_cash"] == self.cash_units
        decomposition_matches = check["wealth"] == check["spread"] + check["carry"] + 2 * check["fees"]
        path_matches = len(self.fills) == 0 or int(self.fills.inventory_after[-1]) == self.final_inventory
        limit_respected = len(self.fills) == 0 or int(np.abs(self.fills.inventory_after).max()) <= self.params.inventory_limit
        return running_matches and decomposition_matches and path_matches and limit_respected

    def summary(self) -> dict:
        """Summarise the run in ticks, basis points and shares only, with no cash amounts.

        The fill rate is filled shares divided by market volume in the window; spread captured and mark-outs are
        share-weighted means.
        Returns: a dict with one entry per summary measure.
        """
        f = self.fills
        qty = f.qty.astype(float)
        row = {"rule": self.rule, "fill_model": self.fill_label, "latency_ms": self.latency_ns / 1e6,
               "fills": len(f), "shares": int(f.qty.sum()), "maker_shares": int(f.qty[f.maker].sum()),
               "taker_shares": int(f.qty[~f.maker].sum()),
               "fill_rate": (float(f.qty.sum()) / self.market_volume) if self.market_volume else float("nan"),
               "spread_captured_ticks": _weighted_mean(self.spread_captured_ticks(), qty),
               "spread_captured_bps": _weighted_mean(self.spread_captured_bps(), qty),
               "net_fee_bps_of_notional": 1e4 * float(f.fee_flow.sum()) / float((f.qty * f.price).sum()) if len(f) else float("nan"),
               "final_inventory": self.final_inventory}
        mean, variance = self.time_weighted_inventory()
        row["inventory_mean"] = mean
        row["inventory_variance"] = variance
        row["inventory_max_abs"] = int(np.abs(f.inventory_after).max()) if len(f) else 0
        for horizon in MARKOUT_HORIZONS_S:
            ticks, bps = self.markout_ticks[horizon], self.markout_bps[horizon]
            row[f"markout_{horizon}s_ticks"] = _weighted_mean(ticks, qty)
            row[f"markout_{horizon}s_bps"] = _weighted_mean(bps, qty)
            row[f"markout_{horizon}s_defined_shares"] = int(f.qty[np.isfinite(ticks)].sum())
        return row


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    usable = np.isfinite(values)
    return float(np.average(values[usable], weights=weights[usable])) if usable.any() and weights[usable].sum() > 0 else float("nan")
