"""Linear-Gaussian transfer entropy pooled over events, with circular-shift surrogate p-values."""

from __future__ import annotations

import numpy as np
from scipy import linalg

MINIMUM_SHIFT = 10


def gaussian_closed_form(a: float, b: float, c: float, var_eps: float = 1.0, var_eta: float = 1.0) -> float:
    """Closed-form transfer entropy from X to Y, TE = 0.5 log(1 + c^2 Var(X_t | Y_t) / Var(eps)), in nats.

    The process is X_{t+1} = b X_t + eta, Y_{t+1} = a Y_t + c X_t + eps, with the stationary covariance from the Lyapunov equation.
    Arguments: a, b and c are the coefficients; var_eps and var_eta the innovation variances.
    Returns: the transfer entropy in nats.
    """
    transition = np.array([[a, c], [0.0, b]])
    stationary = linalg.solve_discrete_lyapunov(transition, np.diag([var_eps, var_eta]))
    var_x_given_y = stationary[1, 1] - stationary[0, 1] ** 2 / stationary[0, 0]
    return float(0.5 * np.log1p(c**2 * var_x_given_y / var_eps))


def event_moments(x_events: list[np.ndarray], y_events: list[np.ndarray]) -> np.ndarray:
    """Per-event cross-products of (Y_{t+1}, Y_t, X_t), each column demeaned within the event, so no pair crosses an event.

    Arguments: x_events and y_events are lists of per-event series of equal length.
    Returns: shape (events, 3, 3).
    """
    moments = np.empty((len(x_events), 3, 3))
    for position, (x, y) in enumerate(zip(x_events, y_events, strict=True)):
        future = y[1:] - y[1:].mean()
        past = y[:-1] - y[:-1].mean()
        source = x[:-1] - x[:-1].mean()
        stack = np.column_stack([future, past, source])
        moments[position] = stack.T @ stack
    return moments


def _te_from_moments(moment: np.ndarray) -> float:
    rss_restricted = moment[0, 0] - moment[0, 1] ** 2 / moment[1, 1]
    determinant = moment[1, 1] * moment[2, 2] - moment[1, 2] ** 2
    beta = np.array([moment[0, 1] * moment[2, 2] - moment[0, 2] * moment[1, 2], moment[0, 2] * moment[1, 1] - moment[0, 1] * moment[1, 2]]) / determinant
    rss_full = moment[0, 0] - beta[0] * moment[0, 1] - beta[1] * moment[0, 2]
    return float(0.5 * np.log(rss_restricted / rss_full))


def transfer_entropy(moments: np.ndarray, weights=None) -> float:
    """Gaussian transfer entropy TE_{X->Y} = 0.5 log(RSS_restricted / RSS_full) in nats, from pooled event moments.

    RSS_restricted regresses Y_{t+1} on Y_t and RSS_full on Y_t and X_t; the estimator assumes linear-Gaussian dynamics.
    Arguments: moments from event_moments; weights optional event multiplicities (for a bootstrap).
    Returns: the transfer entropy.
    Reference: Schreiber (2000), Phys. Rev. Lett.; Barnett, Barrett and Seth (2009), Phys. Rev. Lett.
    """
    pooled = moments.sum(axis=0) if weights is None else np.einsum("e,eij->ij", weights, moments)
    return _te_from_moments(pooled)


def surrogate_pvalue(x_events: list[np.ndarray], y_events: list[np.ndarray], surrogates: int, rng: np.random.Generator) -> float:
    """Surrogate p-value (1 + #{TE_s >= TE}) / (1 + surrogates) of TE_{X->Y}.

    Each surrogate shifts X circularly within each event by an independent offset at least 10 steps from zero, which keeps
    X's autocorrelation and breaks the coupling.
    Arguments: x_events and y_events are lists of per-event series; surrogates the number of surrogates; rng the generator.
    Returns: the p-value.
    """
    events = len(x_events)
    observed_moments = event_moments(x_events, y_events)
    observed = transfer_entropy(observed_moments)
    exceed = 0
    aligned = []
    for x, y in zip(x_events, y_events, strict=True):
        future = y[1:] - y[1:].mean()
        past = y[:-1] - y[:-1].mean()
        aligned.append((x - x.mean(), future, past))
    for _ in range(surrogates):
        pooled = observed_moments.copy()
        for position in range(events):
            x, future, past = aligned[position]
            size = x.size
            offset = int(rng.integers(MINIMUM_SHIFT, size - MINIMUM_SHIFT))
            shifted = np.roll(x, offset)[:-1]
            shifted = shifted - shifted.mean()
            pooled[position, 0, 2] = pooled[position, 2, 0] = float(future @ shifted)
            pooled[position, 1, 2] = pooled[position, 2, 1] = float(past @ shifted)
            pooled[position, 2, 2] = float(shifted @ shifted)
        if _te_from_moments(pooled.sum(axis=0)) >= observed:
            exceed += 1
    return (1.0 + exceed) / (1.0 + surrogates)
