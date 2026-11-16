"""Benjamini-Yekutieli false discovery rate control under arbitrary dependence."""

from __future__ import annotations

import numpy as np


def adjusted_pvalues(pvalues) -> np.ndarray:
    """Adjust p-values with the Benjamini-Yekutieli procedure.

    The adjusted value of the i-th smallest p-value is min(1, min_{j >= i} p_(j) m c(m) / j), with c(m) = sum_{k<=m} 1/k.
    Arguments: pvalues is a sequence of p-values in [0, 1].
    Returns: the adjusted p-values in the input order.
    Reference: Benjamini and Yekutieli (2001).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    harmonic = np.sum(1.0 / np.arange(1, m + 1))
    scaled = p[order] * m * harmonic / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(monotone, 1.0)
    return adjusted


def reject(pvalues, q: float = 0.05) -> np.ndarray:
    """Return a boolean mask of the tests rejected at false discovery rate q (0.05 by default)."""
    return adjusted_pvalues(pvalues) <= q
