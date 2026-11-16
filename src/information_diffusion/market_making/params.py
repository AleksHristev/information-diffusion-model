"""Backtest parameters: price scale, latency grid, mark-out horizons, fees and sensitivity ranges."""

from __future__ import annotations

from dataclasses import dataclass

PRICE_SCALE = 1_000_000
CENTS_TO_UNITS = 10_000

LATENCY_GRID_NS = (0, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)
PRIMARY_LATENCY_NS = 1_000_000
MARKOUT_HORIZONS_S = (1, 5, 30)

SENSITIVITY_INVENTORY_LIMITS = (500, 2_000)
SENSITIVITY_TAKE_FEES_CENTS = (0.25, 0.30)
SENSITIVITY_ADD_REBATES_CENTS = (0.15, 0.25)
SENSITIVITY_RULE6_FACTORS = (1.5, 3.0)
SENSITIVITY_RULE6_DECAY_ENDS_S = (300.0, 900.0)


@dataclass(frozen=True)
class EngineParams:
    """Engine parameters.

    Fields: tick in dollars (1 cent by default); order_size in shares (100 by default); inventory_limit in shares
    either way (1,000 by default); take_fee_cents and add_rebate_cents per share (0.30 and 0.20 by default).
    """

    tick: float = 0.01
    order_size: int = 100
    inventory_limit: int = 1_000
    take_fee_cents: float = 0.30
    add_rebate_cents: float = 0.20

    def __post_init__(self) -> None:
        if self.tick <= 0 or self.order_size <= 0 or self.inventory_limit <= 0:
            raise ValueError("tick, order size and inventory limit must be positive")
        if self.take_fee_cents < 0 or self.add_rebate_cents < 0:
            raise ValueError("the take fee and the add rebate are stated as non-negative amounts")

    @property
    def tick_units(self) -> int:
        """Return the tick in micro-dollars."""
        return int(round(self.tick * PRICE_SCALE))

    @property
    def take_fee_units(self) -> int:
        """Return the per-share take fee in micro-dollars."""
        return int(round(self.take_fee_cents * CENTS_TO_UNITS))

    @property
    def add_rebate_units(self) -> int:
        """Return the per-share add rebate in micro-dollars."""
        return int(round(self.add_rebate_cents * CENTS_TO_UNITS))
