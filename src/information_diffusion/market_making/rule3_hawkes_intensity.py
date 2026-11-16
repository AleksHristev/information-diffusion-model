"""Quoting rule 3: the Avellaneda-Stoikov half-spread widened by a multiplier driven by the Hawkes intensity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms, as_of
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread

DEFAULT_EXPONENT = 0.5
DEFAULT_CAP = 3.0


def intensity_multiplier(t_ns: np.ndarray, intensity_t_ns: np.ndarray, intensity: np.ndarray, baseline: float, *, exponent: float = DEFAULT_EXPONENT, cap: float = DEFAULT_CAP) -> np.ndarray:
    """Compute the spread multiplier clip((lambda_t / lambda_bar)^exponent, 1, cap).

    lambda_t is the last intensity at or before t; the multiplier is 1 before the first intensity value, and the floor
    at 1 means low intensity never narrows the spread.
    Arguments: t_ns are the decision times in nanoseconds; intensity_t_ns and intensity are the intensity series;
    baseline is the positive baseline lambda_bar; exponent is 0.5 by default; cap is 3 by default.
    Returns: the multiplier at each decision time.
    Reference: Hawkes (1971); Clark (1973); Ane and Geman (2000); Wyart et al. (2008).
    """
    if baseline <= 0:
        raise ValueError("the baseline intensity must be positive")
    current = as_of(intensity_t_ns, intensity, t_ns, default=baseline)
    if np.any(current < 0):
        raise ValueError("intensity values must be non-negative")
    return np.clip((current / baseline) ** exponent, 1.0, cap)


@dataclass(frozen=True)
class HawkesIntensity:
    """Rule 3: the rule 2 reservation price, with the half-spread multiplied by the intensity multiplier.

    Fields: params, the ASParams; intensity_t_ns and intensity, the Hawkes intensity series; baseline_intensity, the
    fixed baseline lambda_bar; exponent and cap of the multiplier.
    Reference: Hawkes (1971), Biometrika; Bacry, Mastromatteo and Muzy (2015), Market Microstructure and Liquidity;
    Avellaneda and Stoikov (2008), Quantitative Finance.
    """

    params: ASParams
    intensity_t_ns: np.ndarray
    intensity: np.ndarray
    baseline_intensity: float
    exponent: float = DEFAULT_EXPONENT
    cap: float = DEFAULT_CAP
    name: ClassVar[str] = "rule3_hawkes_intensity"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on mid with the Avellaneda-Stoikov skew and the intensity-scaled half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        multiplier = intensity_multiplier(t_ns, self.intensity_t_ns, self.intensity, self.baseline_intensity, exponent=self.exponent, cap=self.cap)
        return QuoteTerms(mid, skew, half_spread * multiplier)
