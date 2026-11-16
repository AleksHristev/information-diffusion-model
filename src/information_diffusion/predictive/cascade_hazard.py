"""Discrete-time logistic hazard of the cascade models, fitted by penalised likelihood on binomial cells."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


@dataclass(frozen=True)
class HazardFit:
    """Fitted logistic hazard.

    Fields: coefficients, the log-likelihood and penalised log-likelihood at the estimate, the convergence flag and the
    number of iterations.
    """

    coefficients: np.ndarray
    log_likelihood: float
    penalised_log_likelihood: float
    converged: bool
    iterations: int


def _validate(design, events, trials, prior_sd, lower):
    design = sparse.csr_matrix(design, dtype=float)
    events, trials = np.asarray(events, dtype=float), np.asarray(trials, dtype=float)
    prior_sd, lower = np.asarray(prior_sd, dtype=float), np.asarray(lower, dtype=float)
    rows, columns = design.shape
    if events.shape != (rows,) or trials.shape != (rows,):
        raise ValueError("events and trials need one value per row")
    if np.any(trials <= 0) or np.any(events < 0) or np.any(events > trials):
        raise ValueError("each row needs trials > 0 and 0 <= events <= trials")
    if prior_sd.shape != (columns,) or lower.shape != (columns,) or np.any(prior_sd <= 0):
        raise ValueError("prior_sd (positive, inf for none) and lower need one value per column")
    return design, events, trials, prior_sd, lower


def log_likelihood(design, events, trials, coefficients) -> float:
    """Binomial log-likelihood sum(y eta - n log(1 + exp(eta))) of the cells, with eta = design @ coefficients.

    Arguments: design is the (rows, columns) design; events and trials are the binomial cells; coefficients is beta.
    Returns: the log-likelihood.
    """
    eta = sparse.csr_matrix(design) @ np.asarray(coefficients, dtype=float)
    return float(np.asarray(events) @ eta - np.asarray(trials) @ np.logaddexp(0.0, eta))


def fit(design, events, trials, prior_sd, lower=None, start=None, *, tolerance: float = 1e-9, max_iterations: int = 5000) -> HazardFit:
    """Fit a ridge logistic hazard by L-BFGS-B, subject to beta >= lower.

    Maximises sum_i [y_i x_i' beta - n_i log(1 + exp(x_i' beta))] - sum_j beta_j^2 / (2 sd_j^2).
    Arguments: design is the (rows, columns) sparse design; events and trials are the binomial cells of each row;
    prior_sd is the prior standard deviation of each coefficient (inf for none); lower is the lower bound of each
    coefficient (none by default); start is the starting value; tolerance is the gradient tolerance; max_iterations is the
    iteration limit.
    Returns: a HazardFit.
    Reference: Allison (1982), Sociological Methodology; le Cessie and van Houwelingen (1992); Byrd, Lu, Nocedal and Zhu
    (1995).
    """
    columns = sparse.csr_matrix(design).shape[1]
    lower = np.full(columns, -np.inf) if lower is None else lower
    design, events, trials, prior_sd, lower = _validate(design, events, trials, prior_sd, lower)
    precision = 1.0 / prior_sd**2
    transposed = design.T.tocsr()

    def objective(beta):
        eta = design @ beta
        value = trials @ np.logaddexp(0.0, eta) - events @ eta + 0.5 * precision @ beta**2
        gradient = transposed @ (trials * expit(eta) - events) + precision * beta
        return float(value), gradient

    x0 = np.zeros(columns) if start is None else np.maximum(np.asarray(start, dtype=float), lower)
    bounds = [(None if not np.isfinite(b) else float(b), None) for b in lower]
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": max_iterations, "ftol": 1e-15, "gtol": tolerance, "maxcor": 30})
    beta = np.asarray(result.x, dtype=float)
    ll = log_likelihood(design, events, trials, beta)
    return HazardFit(beta, ll, ll - 0.5 * float(precision @ beta**2), bool(result.success), int(result.nit))


def probability(design, coefficients) -> np.ndarray:
    """Return the logistic activation probability Lambda(design @ coefficients) of each row."""
    return expit(sparse.csr_matrix(design) @ np.asarray(coefficients, dtype=float))


def cell_log_likelihood(probabilities, events, trials, floor: float) -> np.ndarray:
    """Compute the Bernoulli log-likelihood of each binomial cell.

    Arguments: probabilities, events and trials of each cell; floor keeps the probability within [floor, 1 - floor].
    Returns: the log-likelihood of each cell.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), floor, 1.0 - floor)
    events, trials = np.asarray(events, dtype=float), np.asarray(trials, dtype=float)
    return events * np.log(p) + (trials - events) * np.log1p(-p)
