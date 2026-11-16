"""Diebold-Yilmaz connectedness by generalised forecast-error variance decomposition of a VAR."""

from __future__ import annotations

import numpy as np

DEFAULT_HORIZON = 12


def moving_average(coefficients: np.ndarray, horizon: int) -> np.ndarray:
    """Moving-average matrices of a VAR, Psi_0 = I and Psi_h = sum_l A_l Psi_{h-l}.

    Arguments: coefficients are the VAR matrices A_l, shape (lags, nodes, nodes); horizon is the number of matrices.
    Returns: Psi_0, ..., Psi_{horizon-1}, shape (horizon, nodes, nodes).
    """
    lags, nodes = coefficients.shape[0], coefficients.shape[1]
    psi = np.zeros((horizon, nodes, nodes))
    psi[0] = np.eye(nodes)
    for step in range(1, horizon):
        for lag in range(1, min(step, lags) + 1):
            psi[step] += coefficients[lag - 1] @ psi[step - lag]
    return psi


def generalised_fevd(coefficients: np.ndarray, covariance: np.ndarray, horizon: int = DEFAULT_HORIZON) -> np.ndarray:
    """Row-normalised generalised forecast-error variance decomposition.

    theta_ij = sigma_jj^(-1) sum_{h<H} (e_i' Psi_h Sigma e_j)^2 / sum_{h<H} e_i' Psi_h Sigma Psi_h' e_i, then each row
    is divided by its sum. It is invariant to a common rescaling of Sigma.
    Arguments: coefficients are the VAR matrices, shape (lags, nodes, nodes); covariance is the innovation covariance
    Sigma; horizon is H (default 12).
    Returns: the (nodes, nodes) matrix of shares, row i giving the share of i's forecast-error variance due to each j.
    Reference: Koop, Pesaran and Potter (1996), J. Econometrics; Pesaran and Shin (1998), Economics Letters; Diebold and Yilmaz (2012),
    Int. J. Forecasting.
    """
    psi = moving_average(coefficients, horizon)
    numerator = np.sum((psi @ covariance) ** 2, axis=0) / np.diag(covariance)
    denominator = np.sum(np.einsum("hij,hij->hi", psi @ covariance, psi), axis=0)
    theta = numerator / denominator[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


def total_connectedness(theta: np.ndarray) -> float:
    """Total connectedness 100 (1 / N) sum_{i != j} theta_ij, in percent, of a row-normalised decomposition theta."""
    return float(100.0 * (theta.sum() - np.trace(theta)) / theta.shape[0])
