"""L1-penalised (Lasso) VAR from per-event moments, with EBIC penalty and lag selection and a post-Lasso refit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

DEFAULT_RATIOS = tuple(np.geomspace(1.0, 1e-3, 16))
TOLERANCE = 1e-5
EBIC_GAMMA = 1.0
MAX_SWEEPS = 2000


@dataclass(frozen=True)
class EventMoments:
    """Per-event cross-products for regressors at lags 1 to max_lag, on one sample shared by every lag order.

    Fields: max_lag; nodes; xx, xy and yy, the per-event X'X, X'Y and Y'Y; n, the number of rows per event.
    """

    max_lag: int
    nodes: int
    xx: np.ndarray
    xy: np.ndarray
    yy: np.ndarray
    n: np.ndarray


def build_moments(events: list[np.ndarray], max_lag: int) -> EventMoments:
    """Per-event moments for a VAR, each event demeaned on its own so that no lag crosses an event boundary.

    Arguments: events is a list of (steps, nodes) arrays of returns; max_lag the largest lag order (at least 1).
    Returns: an EventMoments.
    """
    if max_lag < 1:
        raise ValueError("max_lag must be at least 1")
    nodes = events[0].shape[1]
    regressors = nodes * max_lag
    xx = np.empty((len(events), regressors, regressors))
    xy = np.empty((len(events), regressors, nodes))
    yy = np.empty((len(events), nodes, nodes))
    n = np.empty(len(events))
    for position, values in enumerate(events):
        demeaned = values - values.mean(axis=0)
        target = demeaned[max_lag:]
        lagged = np.hstack([demeaned[max_lag - lag : -lag] for lag in range(1, max_lag + 1)])
        xx[position] = lagged.T @ lagged
        xy[position] = lagged.T @ target
        yy[position] = target.T @ target
        n[position] = target.shape[0]
    return EventMoments(max_lag, nodes, xx, xy, yy, n)


def _aggregate(moments: EventMoments, lag_order: int, weights=None):
    m = moments.nodes * lag_order
    if weights is None:
        weights = np.ones(moments.n.size)
    xx = np.einsum("e,eij->ij", weights, moments.xx[:, :m, :m])
    xy = np.einsum("e,eij->ij", weights, moments.xy[:, :m, :])
    return xx, xy, np.einsum("e,eij->ij", weights, moments.yy), float(weights @ moments.n)


def _standardised(xx: np.ndarray, xy: np.ndarray, yy: np.ndarray, n: float):
    scale_x = np.sqrt(np.maximum(np.diag(xx) / n, 1e-300))
    scale_y = np.sqrt(np.maximum(np.diag(yy) / n, 1e-300))
    return xx / (n * np.outer(scale_x, scale_x)), xy / (n * np.outer(scale_x, scale_y)), scale_x, scale_y


def coordinate_descent(gram: np.ndarray, cross: np.ndarray, penalties: np.ndarray, init=None, *, tol: float = TOLERANCE, max_sweeps: int = MAX_SWEEPS) -> np.ndarray:
    """Lasso min_b (1 / 2n) ||y - X b||^2 + lambda ||b||_1 by cyclic coordinate descent on moments, for all equations at once.

    Each update is the soft threshold b_j = S(c_j - sum_{k != j} G_jk b_k, lambda) / G_jj.
    Arguments: gram is the standardised X'X / n; cross the standardised X'Y / n; penalties one lambda per equation;
    init optional starting coefficients; tol the largest coefficient change at convergence; max_sweeps the sweep limit.
    Returns: coefficients of shape (m, equations).
    Reference: Tibshirani (1996), JRSS B; Friedman, Hastie and Tibshirani (2010), J. Statistical Software.
    """
    m, equations = cross.shape
    b = np.zeros((m, equations)) if init is None else init.copy()
    residual = cross - gram @ b
    diagonal = np.diag(gram)
    for _ in range(max_sweeps):
        change = 0.0
        for j in range(m):
            old = b[j]
            excluding = residual[j] + diagonal[j] * old
            new = np.sign(excluding) * np.maximum(np.abs(excluding) - penalties, 0.0) / diagonal[j]
            delta = new - old
            if np.any(delta):
                residual -= np.outer(gram[:, j], delta)
                b[j] = new
                change = max(change, float(np.max(np.abs(delta))))
        if change < tol:
            break
    return b


def _post_lasso(xx: np.ndarray, xy: np.ndarray, yy: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, float]:
    beta = np.zeros(xy.size)
    if support.any():
        beta_s = np.linalg.solve(xx[np.ix_(support, support)], xy[support])
        beta[support] = beta_s
        rss = yy - float(beta_s @ xy[support])
    else:
        rss = float(yy)
    return beta, float(max(rss, 1e-300))


@dataclass
class VarFit:
    """Sparse VAR fit.

    Fields: lag_order; nodes; coefficients, the post-Lasso OLS coefficients (nodes * lag_order, nodes); lasso_standardised,
    the Lasso coefficients on the standardised scale; penalties, the chosen lambda per equation; bic, the EBIC per equation.
    """

    lag_order: int
    nodes: int
    coefficients: np.ndarray
    lasso_standardised: np.ndarray
    penalties: np.ndarray
    bic: np.ndarray

    def coefficient_matrices(self, coefficients=None) -> np.ndarray:
        """Coefficients (the fit's, or those given) as (lags, nodes, nodes): entry [l, i, j] is the effect of node j at lag l + 1 on node i."""
        b = self.coefficients if coefficients is None else coefficients
        return np.stack([b[lag * self.nodes : (lag + 1) * self.nodes].T for lag in range(self.lag_order)])


def fit_ebic(moments: EventMoments, lag_order: int, ratios=DEFAULT_RATIOS) -> VarFit:
    """Sparse VAR of a given lag order, with lambda chosen per equation by the extended BIC and a post-Lasso OLS refit.

    Regressors and responses are standardised. lambda_i is a ratio of max_j |c_j| from the grid, chosen to minimise
    EBIC_i = n log(RSS_i / n) + df_i log n + 2 log C(M, df_i), with RSS_i from the OLS refit on the Lasso support and
    M = nodes * lag_order the number of regressors.
    Arguments: moments are the EventMoments; lag_order the VAR order; ratios the penalty grid (default 16 ratios from 1 to 0.001).
    Returns: a VarFit.
    Reference: Chen and Chen (2008), Biometrika; Zou, Hastie and Tibshirani (2007), Annals of Statistics; Belloni and
    Chernozhukov (2013), Bernoulli.
    """
    xx, xy, yy, n = _aggregate(moments, lag_order)
    diagonal_yy = np.diag(yy)
    m, equations = xy.shape
    gram, cross, scale_x, scale_y = _standardised(xx, xy, yy, n)
    largest = np.max(np.abs(cross), axis=0)
    best_bic = np.full(equations, np.inf)
    best_beta = np.zeros((m, equations))
    best_lasso = np.zeros((m, equations))
    best_penalty = np.zeros(equations)
    b = None
    for ratio in ratios:
        penalties = ratio * largest
        b = coordinate_descent(gram, cross, penalties, b)
        for equation in range(equations):
            support = b[:, equation] != 0.0
            beta, rss = _post_lasso(xx, xy[:, equation], float(diagonal_yy[equation]), support)
            df = int(support.sum())
            criterion = n * np.log(rss / n) + df * np.log(n) + 2.0 * EBIC_GAMMA * (gammaln(m + 1.0) - gammaln(df + 1.0) - gammaln(m - df + 1.0))
            if criterion < best_bic[equation]:
                best_bic[equation], best_penalty[equation] = criterion, penalties[equation]
                best_beta[:, equation], best_lasso[:, equation] = beta, b[:, equation]
    return VarFit(lag_order, moments.nodes, best_beta, best_lasso, best_penalty, best_bic)


def select_lag_order(moments: EventMoments, max_order: int | None = None, ratios=DEFAULT_RATIOS) -> tuple[VarFit, np.ndarray]:
    """Choose the lag order minimising the total EBIC over equations, fitting orders 1 to max_order on the common sample.

    Arguments: moments are the EventMoments; max_order the largest order (default moments.max_lag); ratios the penalty grid.
    Returns: (the chosen VarFit, total EBIC of each order).
    """
    top = moments.max_lag if max_order is None else max_order
    fits = [fit_ebic(moments, order, ratios) for order in range(1, top + 1)]
    totals = np.array([fit.bic.sum() for fit in fits])
    return fits[int(np.argmin(totals))], totals


def refit_weighted(moments: EventMoments, fit: VarFit, weights: np.ndarray) -> np.ndarray:
    """Refit on weighted event moments (for an event bootstrap): Lasso at the fit's penalties, then post-Lasso OLS.

    Arguments: moments are the EventMoments; fit the original VarFit; weights one multiplicity per event.
    Returns: coefficients of shape (m, nodes).
    """
    xx, xy, yy, n = _aggregate(moments, fit.lag_order, weights)
    gram, cross, _, _ = _standardised(xx, xy, yy, n)
    b = coordinate_descent(gram, cross, fit.penalties, fit.lasso_standardised)
    coefficients = np.zeros(b.shape)
    diagonal_yy = np.diag(yy)
    for equation in range(b.shape[1]):
        coefficients[:, equation] = _post_lasso(xx, xy[:, equation], float(diagonal_yy[equation]), b[:, equation] != 0.0)[0]
    return coefficients


def residual_covariance(moments: EventMoments, fit: VarFit, coefficients=None, weights=None) -> np.ndarray:
    """Pooled residual covariance E'E / n = (Y'Y - B'X'Y - Y'X B + B'X'X B) / n from event moments.

    Arguments: moments are the EventMoments; fit the VarFit; coefficients optional B (default the fit's); weights
    optional event multiplicities.
    Returns: the (nodes, nodes) covariance.
    """
    xx, xy, yy, n = _aggregate(moments, fit.lag_order, weights)
    b = fit.coefficients if coefficients is None else coefficients
    cross = b.T @ xy
    return (yy - cross - cross.T + b.T @ xx @ b) / n


def lasso_support(moments: EventMoments, fit: VarFit, weights: np.ndarray) -> np.ndarray:
    """Lasso support at the fit's penalties on weighted event moments, for stability selection.

    Arguments: moments are the EventMoments; fit the VarFit; weights one multiplicity per event.
    Returns: a boolean (m, nodes) array of non-zero coefficients.
    """
    xx, xy, yy, n = _aggregate(moments, fit.lag_order, weights)
    gram, cross, _, _ = _standardised(xx, xy, yy, n)
    return coordinate_descent(gram, cross, fit.penalties, fit.lasso_standardised) != 0.0
