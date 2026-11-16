"""Linear Threshold model with logistic noise and fixed node thresholds, fitted by maximum likelihood."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import chi2

NEVER = -1
_CUTOFF = 0.5 * float(chi2.ppf(0.95, 1))
_WEIGHT_BOUND = 30.0
_THRESHOLD_BOUND = 60.0


@dataclass(frozen=True)
class LinearThresholdFit:
    """Fitted Linear Threshold model.

    Fields: edges is (m, 2) parent and child; weight, weight_lower and weight_upper are the edge weights and their 95%
    profile-likelihood bounds; threshold, threshold_lower and threshold_upper are the node thresholds and bounds;
    identified flags edges with a finite unique estimate; log_likelihood and converged describe the fit. Parameters that
    are not identified or are separated are NaN.
    """

    edges: np.ndarray
    weight: np.ndarray
    weight_lower: np.ndarray
    weight_upper: np.ndarray
    threshold: np.ndarray
    threshold_lower: np.ndarray
    threshold_upper: np.ndarray
    identified: np.ndarray
    log_likelihood: float
    converged: bool


def _validate(activation: np.ndarray, adjacency: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    activation = np.asarray(activation, dtype=np.int64)
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if activation.ndim != 2 or activation.shape[1] != adjacency.shape[0]:
        raise ValueError("activation must be (cascades, nodes) with one column per node")
    if adjacency.diagonal().any():
        raise ValueError("self-links are not allowed")
    if activation.max(initial=-1) >= n_steps or activation.min(initial=-1) < -1:
        raise ValueError("activation steps must lie in {-1, 0, ..., n_steps - 1}")
    return activation, adjacency


def _risk_table(activation: np.ndarray, parents: np.ndarray, child: int, n_steps: int, window: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = parents.size
    code_blocks: list[np.ndarray] = []
    event_blocks: list[np.ndarray] = []
    for c in range(activation.shape[0]):
        activated_at = int(activation[c, child])
        if activated_at == 0:
            continue
        last = n_steps - 1 if activated_at < 0 else activated_at
        if last < 1:
            continue
        codes = np.zeros(last + 1, dtype=np.int64)
        for j, parent in enumerate(parents):
            parent_step = int(activation[c, parent])
            if parent_step < 0:
                continue
            start = parent_step + 1
            stop = last + 1 if window is None else min(last + 1, parent_step + window + 1)
            if start < stop:
                codes[start:stop] |= np.int64(1) << j
        events = np.zeros(last, dtype=float)
        if activated_at > 0:
            events[-1] = 1.0
        code_blocks.append(codes[1:])
        event_blocks.append(events)
    if not code_blocks:
        return np.zeros((0, k + 1)), np.zeros(0), np.zeros(0)
    codes = np.concatenate(code_blocks)
    events = np.concatenate(event_blocks)
    distinct, inverse = np.unique(codes, return_inverse=True)
    trials = np.bincount(inverse, minlength=distinct.size).astype(float)
    event_counts = np.bincount(inverse, weights=events, minlength=distinct.size)
    exposure = ((distinct[:, None] >> np.arange(k, dtype=np.int64)) & 1).astype(float)
    return np.column_stack([exposure, -np.ones(len(distinct))]), event_counts, trials


def _log_likelihood(design: np.ndarray, events: np.ndarray, trials: np.ndarray, beta: np.ndarray) -> float:
    eta = design @ beta
    return float(events @ eta - trials @ np.logaddexp(0.0, eta))


def _fit_child(design: np.ndarray, events: np.ndarray, trials: np.ndarray, lower: np.ndarray, upper: np.ndarray, start: np.ndarray, fixed: np.ndarray | None = None, *, tolerance: float = 1e-8, max_iterations: int = 200) -> tuple[np.ndarray, float, bool]:
    beta = np.clip(start, lower, upper)
    current = _log_likelihood(design, events, trials, beta)
    fixed = np.zeros(beta.size, dtype=bool) if fixed is None else fixed
    converged = False
    for _ in range(max_iterations):
        mu = expit(design @ beta)
        gradient = design.T @ (events - trials * mu)
        weight = trials * mu * (1.0 - mu)
        active = fixed | ((beta <= lower) & (gradient < 0)) | ((beta >= upper) & (gradient > 0))
        free = ~active
        if not free.any() or float(np.max(np.abs(gradient[free]))) < tolerance:
            converged = True
            break
        design_free = design[:, free]
        hessian = (design_free.T * weight) @ design_free
        hessian += np.eye(hessian.shape[0]) * max(1e-9 * float(np.trace(hessian)) / hessian.shape[0], 1e-10)
        step = np.linalg.solve(hessian, gradient[free])
        rate = 1.0
        while True:
            candidate = beta.copy()
            candidate[free] = np.clip(beta[free] + rate * step, lower[free], upper[free])
            updated = _log_likelihood(design, events, trials, candidate)
            if updated >= current or rate < 1e-8:
                break
            rate *= 0.5
        if updated <= current + 1e-12 and float(np.max(np.abs(candidate - beta))) < 1e-10:
            beta, current = candidate, updated
            converged = True
            break
        beta, current = candidate, updated
    return beta, current, converged


def _null_space_flags(design: np.ndarray) -> np.ndarray:
    if design.shape[0] == 0:
        return np.ones(design.shape[1], dtype=bool)
    _, singular, right = np.linalg.svd(design, full_matrices=True)
    rank = int((singular > 1e-8 * max(singular.max(), 1.0)).sum())
    if rank == design.shape[1]:
        return np.zeros(design.shape[1], dtype=bool)
    return np.abs(right[rank:]).max(axis=0) > 1e-8


def _independent_columns(design: np.ndarray, flagged: np.ndarray) -> np.ndarray:
    order = list(np.argsort(flagged.astype(int), kind="stable"))
    basis: list[np.ndarray] = []
    keep = np.zeros(design.shape[1], dtype=bool)
    for j in order:
        column = design[:, j].astype(float)
        norm = float(np.linalg.norm(column))
        if norm == 0.0:
            continue
        residual = column / norm
        for vector in basis:
            residual = residual - (vector @ residual) * vector
        size = float(np.linalg.norm(residual))
        if size > 1e-8:
            basis.append(residual / size)
            keep[j] = True
    return keep


def _profile_bounds(design, events, trials, lower, upper, solution, index, level) -> tuple[float, float]:
    estimate = float(solution[index])
    state = {"beta": solution.copy()}
    fixed = np.zeros(solution.size, dtype=bool)
    fixed[index] = True

    def excess(value: float) -> float:
        start = state["beta"].copy()
        start[index] = value
        beta, loglik, _ = _fit_child(design, events, trials, lower, upper, start, fixed, tolerance=1e-6)
        state["beta"] = beta
        return loglik - level

    def expand(direction: float) -> float:
        bound = upper[index] if direction > 0 else lower[index]
        reach = max(abs(estimate) * 0.25, 0.5)
        previous = estimate
        while True:
            trial = estimate + direction * reach
            trial = min(trial, bound) if direction > 0 else max(trial, bound)
            if excess(trial) < 0:
                return float(brentq(excess, min(previous, trial), max(previous, trial), xtol=1e-5, maxiter=200))
            if trial == bound:
                return np.nan
            previous = trial
            reach *= 1.8

    high = expand(+1.0)
    if lower[index] == 0.0:
        if estimate <= 0.0 or excess(0.0) >= 0:
            low = 0.0
        else:
            low = float(brentq(excess, 0.0, estimate, xtol=1e-5, maxiter=200))
    else:
        low = expand(-1.0)
    return low, high


def fit(activation: np.ndarray, adjacency: np.ndarray, n_steps: int, *, window: int | None = 4, intervals: bool = True) -> LinearThresholdFit:
    """Fit edge weights b_vw >= 0 and node thresholds theta_w by projected Newton, one child at a time.

    A node w inactive at the end of step t - 1 activates in step t with probability Lambda(S_w(t) - theta_w), where
    S_w(t) = sum_v b_vw 1{1 <= t - t_v <= L}. Nodes with no at-risk step get NaN.
    Arguments: activation is an integer (cascades, nodes) array of activation steps, -1 if never; adjacency is a boolean
    (parent, child) matrix; n_steps is the number of steps observed; window is the influence window L in steps (4 by
    default; None gives the classical form, in which seed-parent weights are not identified); intervals requests 95%
    profile-likelihood intervals.
    Returns: a LinearThresholdFit.
    Reference: Granovetter (1978); Kempe, Kleinberg and Tardos (2003); Goyal, Bonchi and Lakshmanan (2010).
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    n_nodes = adjacency.shape[0]
    parents_of, children_of = np.nonzero(adjacency)
    edges = np.column_stack([parents_of, children_of])
    edge_index = np.full(adjacency.shape, -1, dtype=np.int64)
    edge_index[parents_of, children_of] = np.arange(len(edges))
    n_edges = len(edges)
    weight = np.full(n_edges, np.nan)
    weight_lower = np.full(n_edges, np.nan)
    weight_upper = np.full(n_edges, np.nan)
    identified = np.zeros(n_edges, dtype=bool)
    threshold = np.full(n_nodes, np.nan)
    threshold_lower = np.full(n_nodes, np.nan)
    threshold_upper = np.full(n_nodes, np.nan)
    total_likelihood = 0.0
    all_converged = True
    for child in range(n_nodes):
        parents = np.flatnonzero(adjacency[:, child])
        k = parents.size
        design, events, trials = _risk_table(activation, parents, child, n_steps, window)
        if design.shape[0] == 0 or events.sum() == 0 and k == 0:
            continue
        lower = np.array([0.0] * k + [-_THRESHOLD_BOUND])
        upper = np.array([_WEIGHT_BOUND] * k + [_THRESHOLD_BOUND])
        flagged = _null_space_flags(design)
        keep = _independent_columns(design, flagged)
        reduced_design = design[:, keep]
        start = np.zeros(int(keep.sum()))
        if keep[-1]:
            rate = float(np.clip(events.sum() / trials.sum(), 1e-6, 1.0 - 1e-6))
            start[-1] = -np.log(rate / (1.0 - rate))
        solution, loglik, converged = _fit_child(reduced_design, events, trials, lower[keep], upper[keep], start)
        if not converged:
            all_converged = False
            continue
        total_likelihood += loglik
        estimate = np.full(design.shape[1], np.nan)
        estimate[keep] = solution
        separated = np.isfinite(estimate) & ((estimate >= upper) | ((estimate <= lower) & (lower <= -_THRESHOLD_BOUND)))
        valid = ~flagged & ~separated & np.isfinite(estimate)
        where = edge_index[parents, child] if k else np.zeros(0, dtype=np.int64)
        identified[where] = valid[:k]
        weight[where[valid[:k]]] = estimate[:k][valid[:k]]
        if valid[-1]:
            threshold[child] = estimate[-1]
        if intervals:
            level = loglik - _CUTOFF
            positions = np.cumsum(keep) - 1
            reduced_estimate = solution
            for j in np.flatnonzero(valid):
                low, high = _profile_bounds(reduced_design, events, trials, lower[keep], upper[keep], reduced_estimate, positions[j], level)
                if j < k:
                    weight_lower[where[j]] = low
                    weight_upper[where[j]] = high
                else:
                    threshold_lower[child] = low
                    threshold_upper[child] = high
    return LinearThresholdFit(edges=edges, weight=weight, weight_lower=weight_lower, weight_upper=weight_upper, threshold=threshold, threshold_lower=threshold_lower, threshold_upper=threshold_upper, identified=identified, log_likelihood=total_likelihood, converged=all_converged)


def log_likelihood(activation: np.ndarray, adjacency: np.ndarray, weights: np.ndarray, thresholds: np.ndarray, n_steps: int, *, window: int | None = 4) -> float:
    """Compute the log-likelihood at given weights and thresholds.

    Arguments: activation, adjacency, n_steps and window are as in fit; weights is a (parent, child) matrix; thresholds
    is a vector over nodes.
    Returns: the log-likelihood.
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    total = 0.0
    for child in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, child])
        design, events, trials = _risk_table(activation, parents, child, n_steps, window)
        if design.shape[0]:
            beta = np.append(weights[parents, child], thresholds[child])
            total += _log_likelihood(design, events, trials, beta)
    return total


def score(activation: np.ndarray, adjacency: np.ndarray, weights: np.ndarray, thresholds: np.ndarray, n_steps: int, *, window: int | None = 4) -> tuple[np.ndarray, np.ndarray]:
    """Compute the gradient of log_likelihood.

    Arguments: as in log_likelihood.
    Returns: a (parent, child) gradient matrix for the weights and a gradient vector for the thresholds.
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    weight_gradient = np.zeros(adjacency.shape)
    threshold_gradient = np.zeros(adjacency.shape[0])
    for child in range(adjacency.shape[0]):
        parents = np.flatnonzero(adjacency[:, child])
        design, events, trials = _risk_table(activation, parents, child, n_steps, window)
        if design.shape[0]:
            beta = np.append(weights[parents, child], thresholds[child])
            gradient = design.T @ (events - trials * expit(design @ beta))
            weight_gradient[parents, child] = gradient[:-1]
            threshold_gradient[child] = gradient[-1]
    return weight_gradient, threshold_gradient
