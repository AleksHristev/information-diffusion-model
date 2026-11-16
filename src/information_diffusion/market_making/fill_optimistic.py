"""Optimistic price-touch fill model, an upper bound on maker fills that ignores queue position."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass(frozen=True)
class OptimisticFills:
    """Price-touch fill model: any trade at or through a resting order's price fills it, up to the trade size.

    Prices are mirrored by the engine (sell prices negated), so one comparison serves both sides.
    Reference: Gould, Porter, Williams, McDonald, Fenn and Howison (2013).
    """

    name: ClassVar[str] = "optimistic"
    label: ClassVar[str] = "optimistic (price touch)"

    def on_arrival(self, order, k: int, best: np.ndarray, best_size: np.ndarray) -> None:
        """Do nothing on arrival, since queue position is ignored; returns None."""
        return None

    def shares_available(self, order, k: int, best: np.ndarray, best_size: np.ndarray, trade_px: int, trade_size: int) -> int:
        """Return the shares a trade makes available to the order: the whole trade size at or through its price, else 0.

        Arguments: order is the resting order; k, best and best_size are unused; trade_px is the mirrored trade price;
        trade_size is the trade size in shares.
        """
        return trade_size if trade_px <= order.px else 0
