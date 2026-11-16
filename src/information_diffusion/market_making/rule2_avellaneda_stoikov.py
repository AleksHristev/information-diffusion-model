"""Quoting rule 2: Avellaneda-Stoikov reservation price and spread with constant arrival intensity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms


@dataclass(frozen=True)
class ASParams:
    """Avellaneda-Stoikov parameters.

    Fields: gamma, risk aversion in 1/dollars; sigma, mid volatility in dollars per root second; k, decay of the fill
    intensity A exp(-k delta) in the distance from the mid, in 1/dollars.
    """

    gamma: float
    sigma: float
    k: float

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.k <= 0 or self.sigma < 0:
            raise ValueError("gamma and k must be positive and sigma non-negative")


def as_skew_and_half_spread(t_ns: np.ndarray, window_end_ns: int, params: ASParams) -> tuple[np.ndarray, np.ndarray]:
    """Compute the skew gamma sigma^2 (T - t) and half-spread 0.5 gamma sigma^2 (T - t) + ln(1 + gamma / k) / gamma.

    Arguments: t_ns are the decision times in nanoseconds; window_end_ns is the window end T in nanoseconds; params
    are the ASParams.
    Returns: the inventory skew per share and the half-spread, in dollars, at each time.
    Reference: Avellaneda and Stoikov (2008).
    """
    time_left = np.maximum((window_end_ns - np.asarray(t_ns, dtype=np.int64)) / 1e9, 0.0)
    variance_term = params.gamma * params.sigma * params.sigma * time_left
    return variance_term, 0.5 * variance_term + np.log1p(params.gamma / params.k) / params.gamma


@dataclass(frozen=True)
class AvellanedaStoikov:
    """Rule 2: reservation price r = m - q gamma sigma^2 (T - t), quoted -/+ the Avellaneda-Stoikov half-spread.

    Fields: params, the ASParams.
    Reference: Avellaneda and Stoikov (2008).
    """

    params: ASParams
    name: ClassVar[str] = "rule2_avellaneda_stoikov"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and half-spread up to window_end_ns."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        return QuoteTerms(mid, skew, half_spread)
