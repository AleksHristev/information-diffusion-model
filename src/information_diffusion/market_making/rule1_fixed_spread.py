"""Quoting rule 1: a fixed half-spread around the NBBO mid, with no inventory skew."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms


@dataclass(frozen=True)
class FixedSpread:
    """Rule 1: quote the NBBO mid -/+ a constant half-spread, whatever the inventory.

    Fields: half_spread in dollars.
    Reference: Avellaneda and Stoikov (2008), symmetric benchmark.
    """

    half_spread: float
    name: ClassVar[str] = "rule1_fixed_spread"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with zero skew and the constant half-spread at each decision time t_ns."""
        return QuoteTerms(mid, np.zeros(len(t_ns)), np.full(len(t_ns), self.half_spread))
