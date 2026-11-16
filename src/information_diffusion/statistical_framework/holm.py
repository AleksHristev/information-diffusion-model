"""Holm's step-down adjustment of p-values for family-wise error control."""

from __future__ import annotations

import numpy as np


def adjusted_pvalues(pvalues) -> np.ndarray:
    """Adjust p-values for multiple testing with Holm's step-down method.

    The adjusted value of the i-th smallest is max_{j <= i} min(1, (m - j + 1) p_(j)).
    Arguments: pvalues is a sequence of p-values in [0, 1].
    Returns: the adjusted p-values in the input order, capped at one.
    Reference: Holm (1979).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    scaled = p[order] * (m - np.arange(m))
    monotone = np.minimum(np.maximum.accumulate(scaled), 1.0)
    adjusted = np.empty(m)
    adjusted[order] = monotone
    return adjusted


def reject(pvalues, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of the tests rejected at family-wise level alpha (0.05 by default)."""
    return adjusted_pvalues(pvalues) <= alpha
