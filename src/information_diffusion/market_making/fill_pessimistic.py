"""Pessimistic queue-position fill model behind the displayed NBBO size, which is taken to be single-venue size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

UNKNOWN = float("inf")


@dataclass(frozen=True)
class PessimisticFills:
    """Fill model that places the order behind all displayed NBBO size at its price.

    The displayed NBBO size is taken to be the size at one venue, an assumption about the data that the model does not
    check. With Q the shares ahead, a trade at the order's price first consumes Q, a trade through the price fills the
    order only when Q is zero, and only trades reduce Q (cancellations are assumed to come from behind). Prices are
    mirrored by the engine for sell orders.
    Reference: Moallemi and Yuan (2017); Gould et al. (2013).
    """

    name: ClassVar[str] = "pessimistic"
    label: ClassVar[str] = "behind single-venue NBBO size, inferred"

    def on_arrival(self, order, k: int, best: np.ndarray, best_size: np.ndarray) -> None:
        """Set the shares ahead of an arriving order: 0 above the best, the displayed size at the best, unknown below.

        Arguments: order is the arriving order; k is the index of the current NBBO update; best and best_size are the
        mirrored best price and displayed size series.
        Returns: None; the order is updated in place.
        """
        touch = int(best[k])
        order.ahead = 0 if order.px > touch else (int(best_size[k]) if order.px == touch else UNKNOWN)
        order.scanned_to = k

    def _advance(self, order, k: int, best: np.ndarray, best_size: np.ndarray) -> None:
        if k <= order.scanned_to:
            return
        segment = best[order.scanned_to + 1:k + 1]
        if order.ahead > 0:
            if (segment < order.px).any():
                order.ahead = 0
            elif order.ahead == UNKNOWN:
                touching = np.flatnonzero(segment == order.px)
                if touching.size:
                    order.ahead = int(best_size[order.scanned_to + 1 + int(touching[0])])
        order.scanned_to = k

    def shares_available(self, order, k: int, best: np.ndarray, best_size: np.ndarray, trade_px: int, trade_size: int) -> int:
        """Update the queue for the NBBO updates up to k and return the shares a trade makes available to the order.

        The best moving strictly through the price clears the queue; the best first reaching the price with the queue
        unknown sets it to the displayed size.
        Arguments: order is the resting order; k is the index of the NBBO update at the trade; best and best_size are
        the mirrored best price and displayed size series; trade_px is the mirrored trade price; trade_size is the
        trade size in shares.
        Returns: the shares available to the order.
        """
        self._advance(order, k, best, best_size)
        if trade_px > order.px or order.ahead == UNKNOWN:
            return 0
        if trade_px == order.px:
            used = min(order.ahead, trade_size)
            order.ahead -= used
            return trade_size - used
        return trade_size if order.ahead == 0 else 0
