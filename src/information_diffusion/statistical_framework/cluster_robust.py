"""Cluster-robust (CR1) standard errors for OLS coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ClusterRobust:
    """OLS fit with CR1 cluster-robust inference.

    Fields: coefficients, covariance, standard errors, t-statistics, two-sided p-values from Student t, the number of
    clusters G and the degrees of freedom G - 1.
    """

    coefficients: np.ndarray
    covariance: np.ndarray
    standard_errors: np.ndarray
    statistics: np.ndarray
    p_values: np.ndarray
    clusters: int
    degrees_of_freedom: int


def cr1(y, X, clusters) -> ClusterRobust:
    """Fit OLS with the CR1 covariance [G / (G - 1)] [(n - 1) / (n - k)] (X'X)^-1 (sum_g s_g s_g') (X'X)^-1.

    Arguments: y is the response; X is the n by k design, including any intercept; clusters labels each row.
    Returns: a ClusterRobust.
    Reference: Liang and Zeger (1986); Cameron and Miller (2015).
    """
    response = np.asarray(y, dtype=float)
    design = np.asarray(X, dtype=float)
    groups = np.asarray(clusters)
    if design.ndim != 2 or design.shape[0] != response.shape[0] or groups.shape[0] != response.shape[0]:
        raise ValueError("X must be n by k and y and clusters must have length n")
    n, k = design.shape
    labels, index = np.unique(groups, return_inverse=True)
    G = labels.size
    if G < 2:
        raise ValueError("at least two clusters are required")
    if np.linalg.matrix_rank(design) < k:
        raise ValueError("X is rank deficient")
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    residuals = response - design @ coefficients
    cluster_scores = np.zeros((G, k))
    np.add.at(cluster_scores, index, design * residuals[:, None])
    meat = cluster_scores.T @ cluster_scores
    bread = np.linalg.inv(design.T @ design)
    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    covariance = correction * bread @ meat @ bread
    standard_errors = np.sqrt(np.diag(covariance))
    t_stats = coefficients / standard_errors
    return ClusterRobust(coefficients, covariance, standard_errors, t_stats, 2 * stats.t.sf(np.abs(t_stats), G - 1), G, G - 1)
