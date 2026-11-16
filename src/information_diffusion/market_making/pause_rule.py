"""Rule 2 quotes pulled for a fixed interval after each anchor time, compared in integer nanoseconds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

PULL_HALF_SPREAD_DOLLARS = 1000000.0


def paused(t_ns: np.ndarray, anchors_ns, from_ns: int, to_ns: int) -> np.ndarray:
    """Test which times are paused, a + from_ns <= t < a + to_ns for some anchor a, in integer nanoseconds.

    Arguments: t_ns are the times; anchors_ns are the anchor times; from_ns and to_ns bound the pause after an anchor.
    Returns: a boolean mask over the times.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    a = np.sort(np.asarray(anchors_ns, dtype=np.int64))
    if a.size == 0 or t.size == 0:
        return np.zeros(t.size, dtype=bool)
    k = np.searchsorted(a, t - int(from_ns), side="right") - 1
    ok = k >= 0
    return ok & (t < a[np.maximum(k, 0)] + int(to_ns))


@dataclass(frozen=True)
class PauseRule:
    """Rule 2 with its quotes pulled for [from_ns, to_ns) nanoseconds after each anchor.

    A pulled quote is placed PULL_HALF_SPREAD_DOLLARS from the centre, a price no trade reaches; the engine applies it
    after its latency.
    Fields: params, the ASParams; anchors_ns, the anchor times; from_ns and to_ns, the pause bounds.
    Reference: Avellaneda and Stoikov (2008), Quantitative Finance, for the base quotes.
    """

    params: ASParams
    anchors_ns: tuple[int, ...]
    from_ns: int
    to_ns: int
    name: ClassVar[str] = "pause_rule"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the half-spread pulled when paused."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        half = np.where(paused(t_ns, self.anchors_ns, self.from_ns, self.to_ns), PULL_HALF_SPREAD_DOLLARS, half_spread)
        return QuoteTerms(mid, skew, half)
