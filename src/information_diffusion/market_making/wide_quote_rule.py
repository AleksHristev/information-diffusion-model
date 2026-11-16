"""Rule 2 quotes widened by a multiplier after each anchor time, then optionally pulled until the span ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.pause_rule import paused
from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

PULL_HALF_SPREAD_DOLLARS = 1000000.0


@dataclass(frozen=True)
class WideRule:
    """Rule 2 with its half-spread widened after each anchor and optionally pulled for the rest of the span.

    The half-spread is multiplied by multiplier over [a, a + end_ns) after each anchor a and, when pause_from_ns is
    given, the quotes are pulled PULL_HALF_SPREAD_DOLLARS from the centre over [a + pause_from_ns, a + end_ns). With
    multiplier 1 and no pause it equals rule 2; with multiplier 1 and a pause it equals PauseRule with from_ns equal
    to pause_from_ns and to_ns equal to end_ns.
    Fields: params, the ASParams; anchors_ns, the anchor times in nanoseconds; multiplier, the widening factor; end_ns,
    the span length in nanoseconds; pause_from_ns, the optional pause start in nanoseconds after the anchor.
    Reference: Avellaneda and Stoikov (2008), Quantitative Finance, for the base quotes.
    """

    params: ASParams
    anchors_ns: tuple[int, ...]
    multiplier: float
    end_ns: int
    pause_from_ns: int | None = None
    name: ClassVar[str] = "wide_rule"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the widened or pulled half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        t = np.asarray(t_ns, dtype=np.int64)
        span = paused(t, self.anchors_ns, 0, self.end_ns)
        half = np.where(span, self.multiplier * half_spread, half_spread) if self.multiplier != 1.0 else half_spread
        if self.pause_from_ns is not None:
            half = np.where(paused(t, self.anchors_ns, self.pause_from_ns, self.end_ns), PULL_HALF_SPREAD_DOLLARS, half)
        return QuoteTerms(mid, skew, half)
