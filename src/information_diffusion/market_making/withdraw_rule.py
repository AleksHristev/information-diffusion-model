"""Rule 2 quotes pulled for a window of any length around each scheduled release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

PULL_FROM_S = -1.0
PULL_HALF_SPREAD_DOLLARS = 1000000.0

NS = 1_000_000_000


def pulled(t_ns: np.ndarray, anchors_ns: tuple[int, ...], pull_from_s: float, pull_to_s: float) -> np.ndarray:
    """Test which times are pulled, u = (t - a) / 1e9 in [pull_from_s, pull_to_s) for some anchor a.

    Arguments: t_ns are the times in nanoseconds; anchors_ns are the anchor times; pull_from_s and pull_to_s bound the
    pull in seconds.
    Returns: a boolean mask over the times.
    """
    t_ns = np.asarray(t_ns, dtype=np.int64)
    out = np.zeros(len(t_ns), dtype=bool)
    for a in anchors_ns:
        u = (t_ns - a) / NS
        out |= (u >= pull_from_s) & (u < pull_to_s)
    return out


@dataclass(frozen=True)
class PullRule:
    """Rule 2 with its quotes pulled for [pull_from_s, pull_to_s) seconds around each anchor.

    With pull_from_s = -1 and pull_to_s = 7 it equals rule 7. A pulled quote is placed PULL_HALF_SPREAD_DOLLARS from
    the centre, a price no trade reaches.
    Fields: params, the ASParams; anchors_ns, the anchor times; pull_to_s; pull_from_s (-1 by default).
    Reference: Avellaneda and Stoikov (2008), Quantitative Finance, for the base quotes.
    """

    params: ASParams
    anchors_ns: tuple[int, ...]
    pull_to_s: float
    pull_from_s: float = PULL_FROM_S
    name: ClassVar[str] = "pull_rule"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the half-spread pulled in the window."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        half = np.where(pulled(t_ns, self.anchors_ns, self.pull_from_s, self.pull_to_s), PULL_HALF_SPREAD_DOLLARS, half_spread)
        return QuoteTerms(mid, skew, half)
