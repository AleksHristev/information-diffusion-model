"""Quoting rule 5: network fair value centre and Hawkes intensity spread multiplier combined."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from information_diffusion.market_making.quote_terms import QuoteTerms
from information_diffusion.market_making.rule2_avellaneda_stoikov import ASParams, as_skew_and_half_spread
from information_diffusion.market_making.rule3_hawkes_intensity import DEFAULT_CAP, DEFAULT_EXPONENT, intensity_multiplier
from information_diffusion.market_making.rule4_network_fair_value import network_center


@dataclass(frozen=True)
class HawkesNetwork:
    """Rule 5: the centre moves to the fair value as in rule 4 and the half-spread widens with intensity as in rule 3.

    Fields: params, the ASParams; intensity_t_ns, intensity and baseline_intensity, the Hawkes intensity inputs;
    fair_value_t_ns and fair_value, the fair value series in dollars; exponent and cap of the intensity multiplier.
    Reference: Avellaneda and Stoikov (2008).
    """

    params: ASParams
    intensity_t_ns: np.ndarray
    intensity: np.ndarray
    baseline_intensity: float
    fair_value_t_ns: np.ndarray
    fair_value: np.ndarray
    exponent: float = DEFAULT_EXPONENT
    cap: float = DEFAULT_CAP
    name: ClassVar[str] = "rule5_hawkes_network"

    def quote_terms(self, t_ns: np.ndarray, mid: np.ndarray, window_end_ns: int) -> QuoteTerms:
        """Return QuoteTerms centred on the fair value with the Avellaneda-Stoikov skew and intensity-scaled half-spread."""
        skew, half_spread = as_skew_and_half_spread(t_ns, window_end_ns, self.params)
        multiplier = intensity_multiplier(t_ns, self.intensity_t_ns, self.intensity, self.baseline_intensity, exponent=self.exponent, cap=self.cap)
        center = network_center(t_ns, mid, self.fair_value_t_ns, self.fair_value)
        return QuoteTerms(center, skew, half_spread * multiplier)
