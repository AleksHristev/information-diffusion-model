"""Benjamini-Yekutieli false-discovery control over the ordered links of a network."""

from __future__ import annotations

import numpy as np

from information_diffusion.statistical_framework import benjamini_yekutieli

DEFAULT_Q = 0.05


def reject_links(link_pvalues, q: float = DEFAULT_Q) -> np.ndarray:
    """Links declared significant by the Benjamini-Yekutieli rule p_(k) <= k q / (m c(m)), c(m) = sum_{i=1}^m 1/i.

    The rule controls the false discovery rate at q under arbitrary dependence.
    Arguments: link_pvalues are the p-values of the m ordered pairs; q the false discovery rate (default 0.05).
    Returns: a boolean mask over the p-values.
    Reference: Benjamini and Yekutieli (2001), Annals of Statistics.
    """
    return benjamini_yekutieli.reject(link_pvalues, q)


def off_diagonal(matrix: np.ndarray) -> np.ndarray:
    """Off-diagonal entries of an (N, N) array in row-major order, as the N (N - 1) ordered pairs."""
    return matrix[~np.eye(matrix.shape[0], dtype=bool)]
