"""Quoting rule 4: Avellaneda-Stoikov quotes centred on a network fair value in place of the NBBO mid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms, as_of
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread


def network_center(t_ns: np.ndarray, mid: np.ndarray, fair_value_t_ns: np.ndarray, fair_value: np.ndarray) -> np.ndarray:
    """Compute the quote centre F_t, the last fair value at or before each decision time.

    Arguments: t_ns are the decision times in nanoseconds; mid is the NBBO mid in dollars, used before the first fair
    value; fair_value_t_ns and fair_value are the fair value series in dollars.
    Returns: the quote centre at each decision time.
    """
    return as_of(fair_value_t_ns, fair_value, t_ns, default=mid)


@dataclass(frozen=True)
class NetworkFairValue:
    """Rule 4: reservation price r = F_t - q gamma sigma^2 (T - t), with the rule 2 half-spread unchanged.

    A fair value far enough from the mid can make a quote marketable, which the engine executes as a taker.
    Fields: params, the ASParams; fair_value_t_ns and fair_value, the lead-lag fair value series in dollars.
    Reference: Avellaneda and Stoikov (2008); Hoffmann, Rosenbaum and Yoshida (2013); Stoikov (2018).
    """

    params: ASParams
    fair_value_t_ns: np.ndarray
    fair_value: np.ndarray
    name: ClassVar[str] = "rule4_network_fair_value"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on the network fair value with the Avellaneda-Stoikov skew and half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        return QuoteTerms(network_center(t_ns, mid, self.fair_value_t_ns, self.fair_value), skew, half_spread)
