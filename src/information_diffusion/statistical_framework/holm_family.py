"""Holm decisions in a family of directional tests, under the standard rule and the sign-first rule."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from information_diffusion.statistical_framework import holm

STANDARD, SIGN_FIRST = "standard", "sign_first"


def entered_pvalues(p_values: Sequence[float], sign_holds: Sequence[bool], removed: Sequence[bool] | None = None, *, rule: str = STANDARD) -> np.ndarray:
    """Compute the p-value with which each test enters the step-down.

    Under the standard rule a removed test enters with p = 1 and every other test with its own p-value; under the
    sign-first rule a test whose estimate has the wrong sign also enters with p = 1.
    Arguments: p_values are the p-values; sign_holds flags estimates with the stated sign; removed flags removed tests
    (none by default); rule is STANDARD or SIGN_FIRST.
    Returns: the entered p-values in the input order.
    """
    if rule not in (STANDARD, SIGN_FIRST):
        raise ValueError(f"rule must be {STANDARD!r} or {SIGN_FIRST!r}")
    p = np.asarray(p_values, dtype=float)
    sign = np.asarray(sign_holds, dtype=bool)
    out = np.zeros(p.size, dtype=bool) if removed is None else np.asarray(removed, dtype=bool)
    if not (p.size == sign.size == out.size):
        raise ValueError("p_values, sign_holds and removed must have one entry per test")
    keep = ~out if rule == STANDARD else (sign & ~out)
    entered = np.where(keep, p, 1.0)
    if np.any(~np.isfinite(entered)):
        raise ValueError("a test that enters with its own p-value needs a finite p-value")
    return entered


def decide(p_values: Sequence[float], sign_holds: Sequence[bool], removed: Sequence[bool] | None = None, alpha: float = 0.05, *, rule: str = STANDARD) -> dict[str, np.ndarray]:
    """Apply Holm's step-down to the entered p-values and the sign condition after the adjustment.

    A test is supported when its adjusted p-value is at most alpha, its sign holds and it is not removed.
    Arguments: p_values, sign_holds, removed and rule are as in entered_pvalues; alpha is the family-wise level.
    Returns: a dict with the entered p-values, the adjusted p-values and the supported mask, in the input order.
    Reference: Holm (1979).
    """
    entered = entered_pvalues(p_values, sign_holds, removed, rule=rule)
    adjusted = holm.adjusted_pvalues(entered)
    out = np.zeros(entered.size, dtype=bool) if removed is None else np.asarray(removed, dtype=bool)
    supported = (adjusted <= alpha) & np.asarray(sign_holds, dtype=bool) & ~out
    return {"entered": entered, "adjusted": adjusted, "supported": supported}


def compare(p_values: Sequence[float], sign_holds: Sequence[bool], removed: Sequence[bool] | None = None, alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Apply the standard and sign-first rules side by side.

    Arguments: p_values, sign_holds, removed and alpha are as in decide.
    Returns: a dict with the entered and adjusted p-values and supported masks of both rules, and a mask of the tests
    on which the decisions disagree.
    """
    standard = decide(p_values, sign_holds, removed, alpha, rule=STANDARD)
    sensitivity = decide(p_values, sign_holds, removed, alpha, rule=SIGN_FIRST)
    return {"entered_standard": standard["entered"], "adjusted_standard": standard["adjusted"], "supported_standard": standard["supported"],
            "entered_sign_first": sensitivity["entered"], "adjusted_sign_first": sensitivity["adjusted"], "supported_sign_first": sensitivity["supported"],
            "decisions_disagree": standard["supported"] != sensitivity["supported"]}
