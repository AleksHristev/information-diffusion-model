"""Quoting rule 6: constant-intensity Avellaneda-Stoikov quotes with the spread widened on a fixed release schedule."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar
from zoneinfo import ZoneInfo

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

NEW_YORK = ZoneInfo("America/New_York")
_NS_PER_S = 1_000_000_000


def fomc_repeat_ns(t0_ns: int) -> int:
    """Return 14:30 New York time on the New York calendar date of t0_ns, in UTC nanoseconds."""
    date = datetime.fromtimestamp(t0_ns // _NS_PER_S, tz=NEW_YORK).date()
    return int(datetime(date.year, date.month, date.day, 14, 30, tzinfo=NEW_YORK).timestamp()) * _NS_PER_S


def schedule_times(t0_ns: int, *, is_fomc: bool) -> tuple[int, ...]:
    """Return the schedule times of a release: t0_ns, plus 14:30 New York time on the same date when is_fomc is true."""
    return (t0_ns, fomc_repeat_ns(t0_ns)) if is_fomc else (t0_ns,)


def schedule_multiplier(t_ns: np.ndarray, t0s_ns: Sequence[int], *, factor: float = 2.0, pre_s: float = 60.0, post_s: float = 120.0, decay_end_s: float = 600.0) -> np.ndarray:
    """Compute the schedule multiplier s(u), u = t - t0 in seconds, taking the largest across the listed release times.

    s(u) = factor for -pre_s <= u < post_s, falls linearly from factor to 1 for post_s <= u < decay_end_s, and is 1
    otherwise.
    Arguments: t_ns are the decision times in nanoseconds; t0s_ns are the release times; factor is 2 by default;
    pre_s, post_s and decay_end_s are 60, 120 and 600 seconds by default.
    Returns: the multiplier at each decision time.
    """
    if factor < 1 or not (-pre_s <= 0 <= post_s < decay_end_s):
        raise ValueError("the schedule needs factor >= 1 and -pre <= 0 <= post < decay end")
    t_ns = np.asarray(t_ns, dtype=np.int64)
    multiplier = np.ones(len(t_ns))
    for t0 in t0s_ns:
        offset = (t_ns - t0) / 1e9
        at_full = (offset >= -pre_s) & (offset < post_s)
        decaying = (offset >= post_s) & (offset < decay_end_s)
        this_release = np.ones(len(t_ns))
        this_release[at_full] = factor
        this_release[decaying] = factor - (factor - 1.0) * (offset[decaying] - post_s) / (decay_end_s - post_s)
        multiplier = np.maximum(multiplier, this_release)
    return multiplier


@dataclass(frozen=True)
class ScheduledAvellanedaStoikov:
    """Rule 6: the rule 2 quotes with the half-spread multiplied by the release schedule; quotes never withdraw.

    Fields: params, the ASParams; t0s_ns, the schedule times in nanoseconds; factor, pre_s, post_s and decay_end_s of
    the schedule.
    Reference: Avellaneda and Stoikov (2008).
    """

    params: ASParams
    t0s_ns: tuple[int, ...]
    factor: float = 2.0
    pre_s: float = 60.0
    post_s: float = 120.0
    decay_end_s: float = 600.0
    name: ClassVar[str] = "rule6_scheduled_avellaneda_stoikov"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the schedule-scaled half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        multiplier = schedule_multiplier(t_ns, self.t0s_ns, factor=self.factor, pre_s=self.pre_s, post_s=self.post_s, decay_end_s=self.decay_end_s)
        return QuoteTerms(mid, skew, half_spread * multiplier)
