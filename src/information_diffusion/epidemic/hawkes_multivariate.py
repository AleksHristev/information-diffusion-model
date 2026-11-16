"""Multivariate Hawkes process with a shared exponential decay: weighted L1 link selection, refit on the support and intervals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize
from scipy.stats import norm

from information_diffusion.epidemic.hawkes_exponential import ExponentialHawkes, exp_states_by_column
from information_diffusion.epidemic.spline_baseline import ConstantBaseline

MERGE_RESOLUTION_SECONDS = 0.001

TARGET_LEVEL = 0.05
SHARE_FLOOR = 1e-3
BETA_GOLDEN_STEPS = 18
BETA_BRACKET = (0.3, 3.3)


@dataclass
class MultivariateFit:
    """Fit of a multivariate Hawkes model.

    Fields: gamma, the D x D branching matrix (Gamma_ij, expected children in i of one event in j); mu, the baseline
    rates (per second); beta, the shared decay (per second); support, the selected off-diagonal links; cov, the
    observed-information covariance of the refit (None when not available); index_map from parameter keys ("mu", i),
    ("gamma", i, j) and ("beta",) to positions in cov; log_likelihood; log_beta_se, the standard error of log beta;
    converged; message.
    """

    gamma: np.ndarray
    mu: np.ndarray
    beta: float
    support: np.ndarray
    cov: np.ndarray | None
    index_map: dict
    log_likelihood: float
    log_beta_se: float
    converged: bool
    message: str = ""

    @property
    def spectral_radius(self) -> float:
        """Spectral radius rho(Gamma) of the branching matrix."""
        return float(np.max(np.abs(linalg.eigvals(self.gamma))))

    def spectral_interval(self, level: float = 0.95):
        """Delta-method interval for the spectral radius from the refit covariance, with gradient u_i v_j / (u' v).

        u and v are the left and right eigenvectors of the leading eigenvalue. The interval is conditional on the
        selected support.
        Arguments: level is the coverage (default 0.95).
        Returns: (lower, upper), NaN when not available.
        """
        if self.cov is None:
            return np.nan, np.nan
        values, left, right = linalg.eig(self.gamma, left=True, right=True)
        index = np.argmax(np.abs(values))
        u = left[:, index].real
        v = right[:, index].real
        denominator = u @ v
        if abs(denominator) < 1e-12:
            return np.nan, np.nan
        gradient = np.zeros(self.cov.shape[0])
        for (key, position) in self.index_map.items():
            if key[0] == "gamma":
                _, i, j = key
                gradient[position] = u[i] * v[j] / denominator
        variance = float(gradient @ self.cov @ gradient)
        if not np.isfinite(variance) or variance <= 0:
            return np.nan, np.nan
        half = float(norm.ppf(0.5 + level / 2)) * np.sqrt(variance)
        centre = self.spectral_radius
        return centre - half, centre + half

    def link_interval(self, i: int, j: int, level: float = 0.95):
        """Wald interval for Gamma_ij from the refit covariance.

        Arguments: i is the target node and j the source node; level is the coverage (default 0.95).
        Returns: (lower, upper), NaN when (i, j) is not in the support or the covariance is not available.
        """
        key = ("gamma", i, j)
        if self.cov is None or key not in self.index_map:
            return np.nan, np.nan
        position = self.index_map[key]
        variance = self.cov[position, position]
        if not np.isfinite(variance) or variance <= 0:
            return np.nan, np.nan
        half = float(norm.ppf(0.5 + level / 2)) * np.sqrt(variance)
        estimate = self.gamma[i, j]
        return float(estimate - half), float(estimate + half)


class _Pooled:

    def __init__(self, times_list, window: float, resolution: float | None):
        self.times_list = [np.asarray(t, dtype=float) for t in times_list]
        self.window = float(window)
        self.resolution = resolution
        self.lag = 0.0 if resolution is None else float(resolution)
        self.D = len(self.times_list)
        pooled = np.concatenate(self.times_list)
        self.grid, inverse = np.unique(pooled, return_inverse=True)
        sizes = [t.size for t in self.times_list]
        offsets = np.cumsum([0] + sizes)
        self.node_index = [inverse[offsets[i]:offsets[i + 1]] for i in range(self.D)]
        self.counts = np.array(sizes)
        self.weight_columns = np.column_stack([np.bincount(self.node_index[j], minlength=self.grid.size).astype(float) for j in range(self.D)])

    def states(self, beta: float):
        compensators = np.array([self._compensator(self.times_list[j], beta) for j in range(self.D)])
        return exp_states_by_column(self.grid, beta, self.weight_columns), compensators

    def states_and_moments(self, beta: float):
        columns, moments = exp_states_by_column(self.grid, beta, self.weight_columns, deriv=True)
        comp = np.empty(self.D)
        dcomp = np.empty(self.D)
        for j in range(self.D):
            comp[j], dcomp[j] = self._compensator_and_grad(self.times_list[j], beta)
        return columns, moments, comp, dcomp

    def _compensator(self, t, beta):
        remaining = np.maximum(self.window - t, self.lag)
        return float(np.sum(np.exp(-beta * self.lag) - np.exp(-beta * remaining)))

    def _compensator_and_grad(self, t, beta):
        remaining = np.maximum(self.window - t, self.lag)
        head = np.exp(-beta * self.lag)
        tail = np.exp(-beta * remaining)
        return float(np.sum(head - tail)), float(np.sum(remaining * tail - self.lag * head))


def _prepare_target(pooled: _Pooled, R, i: int, active: np.ndarray):
    return R[pooled.node_index[i]][:, active]


def _objective_target(y, X, compensators, beta, window, resolution, penalty_weights):
    mu, gamma = y[0], y[1:]
    lam = mu + beta * (X @ gamma)
    if np.any(lam <= 0):
        return np.inf, np.full(y.size, np.nan)
    if resolution is None:
        log_rate = np.log(lam)
        weight = 1.0 / lam
    else:
        r = np.minimum(resolution * lam, 500.0)
        log_rate = np.log(np.expm1(r) / resolution)
        weight = resolution / (-np.expm1(-r))
    value = -float(np.sum(log_rate)) + mu * window + float(compensators @ gamma) + float(penalty_weights @ gamma)
    gradient = np.empty(y.size)
    gradient[0] = window - float(np.sum(weight))
    gradient[1:] = compensators + penalty_weights - beta * (X.T @ weight)
    return value, gradient


def _solve_target(X, compensators, beta, window, resolution, penalty_weights, start=None):
    p = X.shape[1] + 1
    if start is None:
        start = np.full(p, 1e-2)
        start[0] = max(X.shape[0] / max(window, 1.0) * 0.5, 1e-8)
    bounds = [(1e-12, None)] + [(0.0, None)] * (p - 1)
    result = optimize.minimize(
        _objective_target, np.maximum(start, 1e-12), args=(X, compensators, beta, window, resolution, penalty_weights), jac=True, method="L-BFGS-B", bounds=bounds,
        options=dict(maxiter=500, maxfun=2000, ftol=1e-10, gtol=1e-5, maxcor=15),
    )
    gradient = result.jac
    free = (result.x > 1e-11) | (gradient < 0)
    small_gradient = bool(np.max(np.abs(gradient[free])) < 1e-2 * max(1.0, X.shape[0] ** 0.5)) if free.any() else True
    return result.x, float(result.fun), bool(result.success or small_gradient)


def _fit_all_targets(pooled, R, compensators, beta, window, resolution, active_list, penalties, starts=None):
    solutions = []
    total = 0.0
    converged = True
    for i in range(pooled.D):
        active = active_list[i]
        X = _prepare_target(pooled, R, i, active)
        pen = penalties[i][active] if penalties is not None else np.zeros(active.size)
        y, value, success = _solve_target(X, compensators[active], beta, window, resolution, pen, None if starts is None else starts[i])
        solutions.append(y)
        total += value
        converged = converged and (success or y.size == 0)
    return solutions, total, converged


def _busiest_node_beta(pooled: _Pooled) -> float:
    busiest = int(np.argmax(pooled.counts))
    baseline = ConstantBaseline(pooled.window)
    problem = ExponentialHawkes(pooled.times_list[busiest], pooled.window, baseline, resolution=pooled.resolution)
    fit = problem.fit()
    return float(problem.beta(fit.x)) if fit.converged else 1.0


def fit_multivariate(times_list, window: float, *, resolution: float | None = MERGE_RESOLUTION_SECONDS, level: float = TARGET_LEVEL, seed_beta: float | None = None) -> MultivariateFit:
    """Fit a multivariate Hawkes model, lambda_i(t) = mu_i + beta sum_j Gamma_ij sum_{t_l^j < t} exp(-beta (t - t_l^j)).

    Stages: beta starts from seed_beta or else from a univariate fit on the busiest node (1 per second when that fit
    does not converge); links are selected by a non-negative L1-penalised likelihood per target with Bernstein weights
    pen_ij = sqrt(2 x V_ij) + (x / 3) B_ij, x = log(2 (D - 1) / level), self-excitation unpenalised, and a link is kept
    when its implied share of the target's events, Gamma_ij N_j / N_i, exceeds 0.001 (self-excitation is always kept);
    the model is then refitted without penalty on the selected support, with beta profiled by an 18-step golden-ratio
    search on log beta over 0.3 to 3.3 times the starting beta, and the observed-information covariance is computed.
    The likelihood is the binned Bernoulli form at the given resolution. Selection and refit use the same data.
    Arguments: times_list holds the event times (seconds) of each node; window is the window length in seconds;
    resolution is the merge bin width in seconds (None for continuous time); level is the false-link level per target;
    seed_beta is an optional starting decay.
    Returns: a MultivariateFit.
    Reference: Hawkes (1971), Biometrika; Hansen, Reynaud-Bouret and Rivoirard (2015), Bernoulli; Lee, Sun, Sun and
    Taylor (2016), Ann. Stat. (post-selection context).
    """
    pooled = _Pooled(times_list, window, resolution)
    D = pooled.D
    beta = float(seed_beta) if seed_beta is not None else _busiest_node_beta(pooled)
    R, compensators = pooled.states(beta)
    active_all = [np.arange(D) for _ in range(D)]
    unpenalised, _, _ = _fit_all_targets(pooled, R, compensators, beta, window, resolution, active_all, None)
    penalties = []
    threshold = float(np.log(2.0 * max(D - 1, 1) / level))
    for i, y in enumerate(unpenalised):
        X = _prepare_target(pooled, R, i, active_all[i])
        lam = y[0] + beta * (X @ y[1:])
        weight = 1.0 / lam if resolution is None else resolution / (-np.expm1(-np.minimum(resolution * lam, 500.0)))
        contributions = weight[:, None] * (beta * X)
        V = np.sum(contributions ** 2, axis=0)
        B = np.max(np.abs(contributions), axis=0) if contributions.shape[0] else np.zeros(D)
        pen = np.sqrt(2.0 * threshold * V) + threshold / 3.0 * B
        pen[i] = 0.0
        penalties.append(pen)
    l1_solutions, _, _ = _fit_all_targets(pooled, R, compensators, beta, window, resolution, active_all, penalties, starts=unpenalised)
    support = np.zeros((D, D), dtype=bool)
    node_counts = pooled.counts.astype(float)
    expected_share = node_counts[None, :] / np.maximum(node_counts[:, None], 1.0)
    for i, y in enumerate(l1_solutions):
        support[i] = y[1:] * expected_share[i] > SHARE_FLOOR
    support[np.diag_indices(D)] = True
    active_list = [np.flatnonzero(support[i]) for i in range(D)]
    fit_at, fit_beta = _golden_beta(pooled, active_list, beta, window, resolution, l1_solutions)
    return _assemble(pooled, active_list, fit_beta, fit_at, window, resolution, support)


def _objective_for_beta(pooled, active_list, beta, window, resolution, starts):
    R, compensators = pooled.states(beta)
    solutions, total, converged = _fit_all_targets(pooled, R, compensators, beta, window, resolution, active_list, None, starts=starts)
    return solutions, total, converged


def _golden_beta(pooled, active_list, beta_start, window, resolution, starts_all):
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    low = np.log(beta_start * BETA_BRACKET[0])
    high = np.log(beta_start * BETA_BRACKET[1])
    first = low + (1.0 - golden) * (high - low)
    second = low + golden * (high - low)
    starts = [s[np.concatenate(([0], 1 + active))] if s.size else s for s, active in zip(starts_all, active_list)]
    solutions_first, value_first, ok_first = _objective_for_beta(pooled, active_list, float(np.exp(first)), window, resolution, starts)
    solutions_second, value_second, ok_second = _objective_for_beta(pooled, active_list, float(np.exp(second)), window, resolution, starts)
    converged = ok_first and ok_second
    for _ in range(BETA_GOLDEN_STEPS):
        if value_first <= value_second:
            high, second, value_second, solutions_second = second, first, value_first, solutions_first
            first = low + (1.0 - golden) * (high - low)
            solutions_first, value_first, ok = _objective_for_beta(pooled, active_list, float(np.exp(first)), window, resolution, solutions_second)
        else:
            low, first, value_first, solutions_first = first, second, value_second, solutions_second
            second = low + golden * (high - low)
            solutions_second, value_second, ok = _objective_for_beta(pooled, active_list, float(np.exp(second)), window, resolution, solutions_first)
        converged = converged and ok
    if value_first <= value_second:
        return (solutions_first, value_first, converged), float(np.exp(first))
    return (solutions_second, value_second, converged), float(np.exp(second))


def _hessian_weight(lam, resolution):
    if resolution is None:
        return 1.0 / lam ** 2, 1.0 / lam
    r = np.minimum(resolution * lam, 500.0)
    denominator = -np.expm1(-r)
    return resolution ** 2 * np.exp(-r) / denominator ** 2, resolution / denominator


def _assemble(pooled, active_list, beta, solution, window, resolution, support) -> MultivariateFit:
    (solutions, total_value, converged) = solution
    D = pooled.D
    R, moments, compensators, compensator_grads = pooled.states_and_moments(beta)
    blocks = []
    borders = []
    index_map: dict = {}
    position = 0
    log_likelihood = -float(total_value)
    for i in range(D):
        active = active_list[i]
        y = solutions[i]
        X = R[pooled.node_index[i]][:, active]
        M = moments[pooled.node_index[i]][:, active]
        lam = y[0] + beta * (X @ y[1:])
        second, weight = _hessian_weight(lam, resolution)
        design = np.column_stack((np.ones(lam.size), beta * X))
        block = design.T @ (design * second[:, None])
        dlam_dbeta = X @ y[1:] - beta * (M @ y[1:])
        border = design.T @ (second * dlam_dbeta)
        border[1:] -= (X.T @ weight) - beta * (M.T @ weight)
        border[1:] += compensator_grads[active]
        blocks.append(block)
        borders.append(border)
        index_map[("mu", i)] = position
        position += 1
        for j in active:
            index_map[("gamma", i, int(j))] = position
            position += 1
    index_map[("beta",)] = position
    size = position + 1
    h_bb = _beta_curvature(pooled, active_list, solutions, beta, window, resolution, R, moments, compensators, compensator_grads)
    full_cov, log_beta_se = _arrow_covariance(blocks, borders, h_bb, size, beta)
    gamma = np.zeros((D, D))
    mu = np.empty(D)
    for i in range(D):
        y = solutions[i]
        mu[i] = y[0]
        gamma[i, active_list[i]] = y[1:]
    off_support = support.copy()
    off_support[np.diag_indices(D)] = False
    return MultivariateFit(gamma, mu, beta, off_support, full_cov, index_map, log_likelihood, log_beta_se, bool(converged and full_cov is not None))


def _beta_curvature(pooled, active_list, solutions, beta, window, resolution, R, moments, compensators, compensator_grads):
    def slope(b):
        R_b, M_b, comp_b, comp_grad_b = pooled.states_and_moments(b)
        total = 0.0
        for i in range(pooled.D):
            active = active_list[i]
            y = solutions[i]
            X = R_b[pooled.node_index[i]][:, active]
            M = M_b[pooled.node_index[i]][:, active]
            lam = y[0] + b * (X @ y[1:])
            if resolution is None:
                weight = 1.0 / lam
            else:
                weight = resolution / (-np.expm1(-np.minimum(resolution * lam, 500.0)))
            dlam = X @ y[1:] - b * (M @ y[1:])
            total += -float(np.sum(weight * dlam)) + float(comp_grad_b[active] @ y[1:])
        return total
    h = 1e-4 * max(beta, 1e-3)
    return (slope(beta + h) - slope(beta - h)) / (2.0 * h)


def _arrow_covariance(blocks, borders, h_bb, size, beta):
    if not np.isfinite(h_bb):
        return None, np.nan
    try:
        factors = [linalg.cho_factor(block) for block in blocks]
    except linalg.LinAlgError:
        return None, np.nan
    solved = [linalg.cho_solve(factor, border) for factor, border in zip(factors, borders)]
    schur = h_bb - sum(float(border @ value) for border, value in zip(borders, solved))
    if not np.isfinite(schur) or schur <= 0:
        return None, np.nan
    vector = np.concatenate(solved + [np.array([-1.0])])
    cov = np.zeros((size, size))
    start = 0
    for factor, block in zip(factors, blocks):
        p = block.shape[0]
        cov[start:start + p, start:start + p] = linalg.cho_solve(factor, np.eye(p))
        start += p
    cov += np.outer(vector, vector) / schur
    return cov, float(np.sqrt(1.0 / schur)) / max(beta, 1e-12)


def simulation_truth(counts, seed: int = 20260101, n_links: int = 60, spectral_radius: float = 0.6, self_share: float = 0.3, window: float = 5400.0, beta: float = 1.0):
    """Random true parameters for a simulation test of the multivariate fit.

    n_links distinct off-diagonal links are drawn at random. Shares S_ij = Gamma_ij N_j / N_i (the fraction of i's events
    caused by j) have row sums equal to rho, so Gamma = diag(N) S diag(N)^(-1) has spectral radius rho; baseline
    rates mu_i = (1 - rho) N_i / T give expected counts N_i.
    Arguments: counts are the expected event counts N_i; seed the random seed; n_links the number of links;
    spectral_radius is rho; self_share the self-excitation share of a node with incoming links; window is T in seconds
    (default 5,400); beta the decay (per second), which does not enter the returned values.
    Returns: (mu, Gamma, links, shares), links being the boolean matrix of true links.
    """
    counts = np.asarray(counts, dtype=float)
    d = counts.size
    rng = np.random.default_rng(seed)
    pairs = np.array([(i, j) for i in range(d) for j in range(d) if i != j])
    chosen = pairs[rng.choice(len(pairs), size=n_links, replace=False)]
    shares = np.zeros((d, d))
    incoming: dict[int, list[int]] = {}
    for target, source in chosen.tolist():
        incoming.setdefault(int(target), []).append(int(source))
    for i in range(d):
        sources = incoming.get(i, [])
        shares[i, i] = self_share if sources else spectral_radius
        for j in sources:
            shares[i, j] = (spectral_radius - self_share) / len(sources)
    gamma = np.empty((d, d))
    for i in range(d):
        gamma[i] = shares[i] * counts[i] / counts
    mu = (1.0 - spectral_radius) * counts / window
    links = np.zeros((d, d), dtype=bool)
    links[chosen[:, 0], chosen[:, 1]] = True
    return mu, gamma, links, shares
