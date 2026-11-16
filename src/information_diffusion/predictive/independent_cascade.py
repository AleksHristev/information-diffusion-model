"""Independent Cascade model fitted by maximum likelihood with EM, with and without a background hazard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

NEVER = -1
_CUTOFF = 0.5 * float(chi2.ppf(0.95, 1))
_CEILING = 1.0 - 1e-12
_LOWER_SEARCH = 1e-9
_UNIDENTIFIED_GAP = 1e-3


@dataclass(frozen=True)
class IndependentCascadeFit:
    """Fitted Independent Cascade.

    Fields: edges is (m, 2) parent and child; probability, lower and upper are the estimates and 95% profile-likelihood
    bounds (NaN for links whose estimate is not unique or that have no trials); identified flags unique estimates;
    trials counts the attempts of each link; log_likelihood and converged describe the fit.
    """

    edges: np.ndarray
    probability: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    identified: np.ndarray
    trials: np.ndarray
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


def _child_tables(activation: np.ndarray, parents: np.ndarray, child: int, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_child = activation[:, child]
    t_parents = activation[:, parents]
    k = parents.size
    attempt = t_parents + 1
    observed = (t_parents >= 0) & (attempt <= n_steps - 1)
    still_inactive = (t_child[:, None] < 0) | (t_child[:, None] > t_parents)
    trial = observed & still_inactive
    positive = trial & (t_child[:, None] == attempt)
    failures = (trial & ~positive).sum(axis=0).astype(float)
    rows = positive.any(axis=1)
    if rows.any():
        codes = positive[rows].astype(np.int64) @ (np.int64(1) << np.arange(k, dtype=np.int64))
        distinct, counts = np.unique(codes, return_counts=True)
        patterns = ((distinct[:, None] >> np.arange(k, dtype=np.int64)) & 1).astype(float)
        multiplicity = counts.astype(float)
    else:
        patterns = np.zeros((0, k))
        multiplicity = np.zeros(0)
    return patterns, multiplicity, failures, trial.sum(axis=0)


def _log_likelihood(patterns: np.ndarray, multiplicity: np.ndarray, failures: np.ndarray, p: np.ndarray) -> float:
    if patterns.size:
        fired = -np.expm1(patterns @ np.log1p(-np.minimum(p, _CEILING)))
        positive = float(multiplicity @ np.log(np.maximum(fired, 1e-300)))
    else:
        positive = 0.0
    return positive + float(failures @ np.log1p(-np.minimum(p, _CEILING)))


def _em(patterns: np.ndarray, multiplicity: np.ndarray, failures: np.ndarray, denominator: np.ndarray, start: np.ndarray, *, fixed: int | None = None, tolerance: float = 1e-9, max_iterations: int = 20000) -> tuple[np.ndarray, bool]:
    p = np.clip(start, 0.0, _CEILING)
    free = denominator > 0
    if fixed is not None:
        free[fixed] = False
    if not free.any():
        return p, True
    for _ in range(max_iterations):
        if patterns.size:
            fired = np.maximum(-np.expm1(patterns @ np.log1p(-p)), 1e-300)
            responsibility = (patterns * (p / fired[:, None])).T @ multiplicity
        else:
            responsibility = np.zeros_like(p)
        updated = p.copy()
        updated[free] = np.minimum(responsibility[free] / denominator[free], _CEILING)
        change = float(np.max(np.abs(updated - p)))
        p = updated
        if change < tolerance:
            return p, True
    return p, False


def _profile_bound(patterns, multiplicity, failures, denominator, solution, index, level, *, upper_side: bool) -> float:
    estimate = float(solution[index])
    state = {"p": solution.copy()}

    def excess(value: float) -> float:
        start = state["p"].copy()
        start[index] = value
        p, _ = _em(patterns, multiplicity, failures, denominator, start, fixed=index, tolerance=1e-7, max_iterations=3000)
        state["p"] = p
        return _log_likelihood(patterns, multiplicity, failures, p) - level

    if upper_side:
        boundary = 1.0 - _LOWER_SEARCH
        if estimate >= boundary or excess(boundary) >= 0:
            return 1.0
        return float(brentq(excess, estimate, boundary, xtol=1e-5, maxiter=200))
    if estimate <= _LOWER_SEARCH or excess(_LOWER_SEARCH) >= 0:
        return 0.0
    return float(brentq(excess, _LOWER_SEARCH, estimate, xtol=1e-5, maxiter=200))


def fit(activation: np.ndarray, adjacency: np.ndarray, n_steps: int, *, intervals: bool = True, tolerance: float = 1e-9, max_iterations: int = 20000) -> IndependentCascadeFit:
    """Fit the edge probabilities of an Independent Cascade on the links of an adjacency by EM.

    A node activated in step s attempts each inactive out-neighbour once, in step s + 1. The log-likelihood is
    sum_w [log(1 - prod_{v in B_w} (1 - p_vw)) + sum_{v in F_w} log(1 - p_vw)], with B_w the parents that attempted w in
    its activation step and F_w those that failed earlier. The likelihood is fitted child by child; two EM starts flag
    links whose estimate is not unique.
    Arguments: activation is an integer (cascades, nodes) array of activation steps, -1 if never; adjacency is a
    boolean (parent, child) matrix; n_steps is the number of steps observed (later attempts are censored); intervals
    requests 95% profile-likelihood intervals; tolerance and max_iterations control EM.
    Returns: an IndependentCascadeFit.
    Reference: Kempe, Kleinberg and Tardos (2003); Saito, Nakano and Kimura (2008); Dempster, Laird and Rubin (1977).
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    parents_of, children_of = np.nonzero(adjacency)
    edges = np.column_stack([parents_of, children_of])
    edge_index = np.full(adjacency.shape, -1, dtype=np.int64)
    edge_index[parents_of, children_of] = np.arange(len(edges))
    estimate = np.full(len(edges), np.nan)
    lower = np.full(len(edges), np.nan)
    upper = np.full(len(edges), np.nan)
    identified = np.zeros(len(edges), dtype=bool)
    trials_out = np.zeros(len(edges), dtype=np.int64)
    log_likelihood = 0.0
    converged = True
    for child in np.flatnonzero(adjacency.any(axis=0)):
        parents = np.flatnonzero(adjacency[:, child])
        where = edge_index[parents, child]
        k = parents.size
        patterns, multiplicity, failures, trials = _child_tables(activation, parents, child, n_steps)
        trials_out[where] = trials
        denominator = patterns.T @ multiplicity + failures
        first, ok_first = _em(patterns, multiplicity, failures, denominator, np.full(k, 0.5), tolerance=tolerance, max_iterations=max_iterations)
        unique = denominator > 0
        if k > 1:
            second, ok_second = _em(patterns, multiplicity, failures, denominator, np.linspace(0.15, 0.85, k), tolerance=tolerance, max_iterations=max_iterations)
            ok_first = ok_first and ok_second
            unique &= np.abs(first - second) <= _UNIDENTIFIED_GAP
        if not ok_first:
            unique[:] = False
            converged = False
        log_likelihood += _log_likelihood(patterns, multiplicity, failures, first)
        identified[where] = unique
        estimate[where[unique]] = first[unique]
        if intervals:
            level = _log_likelihood(patterns, multiplicity, failures, first) - _CUTOFF
            for index in np.flatnonzero(unique):
                lower[where[index]] = _profile_bound(patterns, multiplicity, failures, denominator, first, index, level, upper_side=False)
                upper[where[index]] = _profile_bound(patterns, multiplicity, failures, denominator, first, index, level, upper_side=True)
    return IndependentCascadeFit(edges=edges, probability=estimate, lower=lower, upper=upper, identified=identified, trials=trials_out, log_likelihood=log_likelihood, converged=converged)


def log_likelihood(activation: np.ndarray, adjacency: np.ndarray, probabilities: np.ndarray, n_steps: int) -> float:
    """Compute the Independent Cascade log-likelihood at given edge probabilities.

    Arguments: activation, adjacency and n_steps are as in fit; probabilities is a (parent, child) matrix.
    Returns: the log-likelihood.
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    total = 0.0
    for child in np.flatnonzero(adjacency.any(axis=0)):
        parents = np.flatnonzero(adjacency[:, child])
        patterns, multiplicity, failures, _ = _child_tables(activation, parents, child, n_steps)
        total += _log_likelihood(patterns, multiplicity, failures, np.minimum(probabilities[parents, child], _CEILING))
    return total


def score(activation: np.ndarray, adjacency: np.ndarray, probabilities: np.ndarray, n_steps: int) -> np.ndarray:
    """Compute the gradient of log_likelihood with respect to each edge probability.

    Arguments: activation, adjacency and n_steps are as in fit; probabilities is a (parent, child) matrix.
    Returns: a (parent, child) gradient matrix, zero outside the adjacency.
    """
    activation, adjacency = _validate(activation, adjacency, n_steps)
    gradient = np.zeros(adjacency.shape)
    for child in np.flatnonzero(adjacency.any(axis=0)):
        parents = np.flatnonzero(adjacency[:, child])
        patterns, multiplicity, failures, _ = _child_tables(activation, parents, child, n_steps)
        p = np.minimum(probabilities[parents, child], _CEILING)
        grad = -failures / (1.0 - p)
        if patterns.size:
            survive = np.exp(patterns @ np.log1p(-p))
            fired = np.maximum(1.0 - survive, 1e-300)
            for pattern, m, s, f in zip(patterns, multiplicity, survive, fired):
                members = pattern.astype(bool)
                grad[members] += m * (s / (1.0 - p[members])) / f
        gradient[parents, child] = grad
    return gradient




@dataclass(frozen=True)
class BackgroundCascadeFit:
    """Fitted Independent Cascade with a known background hazard.

    Fields: probability is the EM estimate of each edge (0 for an edge with no attempt); identified flags edges that were
    attempted and whose two EM starts agree within 1e-3; attempts counts the attempts of each edge; log_likelihood and
    converged describe the fit.
    """

    probability: np.ndarray
    identified: np.ndarray
    attempts: np.ndarray
    log_likelihood: float
    converged: bool


def noisy_or(incidence, background, probability) -> np.ndarray:
    """Compute the noisy-OR activation probability F = 1 - (1 - h) prod_{v in B} (1 - p_v) of each trial.

    Arguments: incidence is a (trials, edges) 0/1 matrix of attempting edges; background is the hazard h of each trial;
    probability is the probability of each edge.
    Returns: the activation probability of each trial.
    """
    from scipy import sparse

    incidence = sparse.csr_matrix(incidence, dtype=float)
    p = np.minimum(np.asarray(probability, dtype=float), _CEILING)
    return -np.expm1(np.log1p(-np.asarray(background, dtype=float)) + incidence @ np.log1p(-p))


def _background_log_likelihood(incidence, background, activated, p) -> float:
    fired = np.clip(noisy_or(incidence, background, p), 1e-300, 1.0 - 1e-16)
    return float(activated @ np.log(fired) + (1.0 - activated) @ np.log1p(-fired))


def _background_em(incidence, transposed, background, activated, attempts, start, tolerance, max_iterations) -> tuple[np.ndarray, bool]:
    p = np.where(attempts > 0, np.clip(start, 0.0, _CEILING), 0.0)
    free = attempts > 0
    for _ in range(max_iterations):
        fired = np.maximum(noisy_or(incidence, background, p), 1e-300)
        responsibility = (transposed @ (activated / fired)) * p
        updated = np.zeros_like(p)
        updated[free] = np.minimum(responsibility[free] / attempts[free], _CEILING)
        change = float(np.max(np.abs(updated - p))) if p.size else 0.0
        p = updated
        if change < tolerance:
            return p, True
    return p, False


def fit_with_background(incidence, background, activated, *, tolerance: float = 1e-9, max_iterations: int = 20000) -> BackgroundCascadeFit:
    """Fit the edge probabilities of the one-attempt cascade combined by noisy-OR with a known background hazard, by EM.

    Arguments: incidence is a (trials, edges) 0/1 matrix of the edges attempting in each trial; background is the hazard
    h in [0, 1) of each trial; activated is the 0/1 outcome of each trial; tolerance and max_iterations control EM.
    Returns: a BackgroundCascadeFit.
    """
    from scipy import sparse

    incidence = sparse.csr_matrix(incidence, dtype=float)
    background = np.asarray(background, dtype=float)
    activated = np.asarray(activated, dtype=float)
    if background.shape != (incidence.shape[0],) or activated.shape != background.shape:
        raise ValueError("background and activated need one value per trial")
    if np.any((background < 0) | (background >= 1)) or np.any((activated != 0) & (activated != 1)):
        raise ValueError("background must lie in [0, 1) and activated in {0, 1}")
    transposed = incidence.T.tocsr()
    attempts = np.asarray(incidence.sum(axis=0)).ravel()
    edges = incidence.shape[1]
    first, ok_first = _background_em(incidence, transposed, background, activated, attempts, np.full(edges, 0.5), tolerance, max_iterations)
    second, ok_second = _background_em(incidence, transposed, background, activated, attempts, np.linspace(0.15, 0.85, edges) if edges > 1 else np.full(edges, 0.3), tolerance, max_iterations)
    identified = (attempts > 0) & (np.abs(first - second) <= _UNIDENTIFIED_GAP)
    return BackgroundCascadeFit(first, identified, attempts.astype(np.int64), _background_log_likelihood(incidence, background, activated, first), bool(ok_first and ok_second))
