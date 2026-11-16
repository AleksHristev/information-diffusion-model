"""Rule 2 quotes restricted to one side in given intervals according to a direction series, with optional pauses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms, as_of
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

ONE_SIDED_DOLLARS = 1000.0
PULL_DOLLARS = 1000000.0


def inside(t_ns: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Test which times lie in some half-open interval [start, end).

    Arguments: t_ns are the times in nanoseconds; starts and ends are the interval bounds, in any order and possibly
    overlapping.
    Returns: a boolean mask over the times.
    """
    t = np.asarray(t_ns, dtype=np.int64)
    s, e = np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
    if s.size == 0 or t.size == 0:
        return np.zeros(t.size, dtype=bool)
    order = np.argsort(s, kind="mergesort")
    s, e = s[order], np.maximum.accumulate(e[order])
    k = np.searchsorted(s, t, side="right") - 1
    return (k >= 0) & (t < e[np.maximum(k, 0)])


@dataclass(frozen=True)
class OneSidedRule:
    """Rule 2 quoting only one side inside the one-sided intervals.

    With d the direction read as of each decision time (0 before its first value): for d > 0 only the rule 2 bid is
    live and the ask is placed 2 ONE_SIDED_DOLLARS above it, out of any trade's reach; for d < 0 only the rule 2 ask
    is live and the bid is placed 2 ONE_SIDED_DOLLARS below it; for d = 0 or outside the intervals the rule 2 quotes
    apply. Inside a pause both quotes are pulled PULL_DOLLARS from the reservation price, which takes precedence.
    Fields: params, the ASParams; one_sided_starts and one_sided_ends, the one-sided intervals in nanoseconds;
    direction_t and direction_v, the times in nanoseconds and the values of the direction step series; pause_starts
    and pause_ends, the paused intervals in nanoseconds (none by default).
    Reference: Avellaneda and Stoikov (2008).
    """

    params: ASParams
    one_sided_starts: np.ndarray
    one_sided_ends: np.ndarray
    direction_t: np.ndarray
    direction_v: np.ndarray
    pause_starts: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    pause_ends: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    name: ClassVar[str] = "one_sided_rule"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms with the centre and half-spread shifted for one-sided and paused times, skew unchanged."""
        skew, half = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        t = np.asarray(t_ns, dtype=np.int64)
        mid = np.asarray(mid, dtype=float)
        d = as_of(self.direction_t, self.direction_v, t, 0.0) if len(self.direction_t) else np.zeros(t.size)
        one = inside(t, self.one_sided_starts, self.one_sided_ends)
        pause = inside(t, self.pause_starts, self.pause_ends)
        p = ONE_SIDED_DOLLARS
        up, down = one & ~pause & (d > 0), one & ~pause & (d < 0)
        center = np.where(up, mid - half + p, np.where(down, mid + half - p, mid))
        half_out = np.where(up | down, p, half)
        half_out = np.where(pause, PULL_DOLLARS, half_out)
        center = np.where(pause, mid, center)
        return QuoteTerms(center, np.asarray(skew, dtype=float) * np.ones(t.size), half_out)
