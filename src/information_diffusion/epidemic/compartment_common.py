"""Shared machinery for the compartment and Bass models: Markov jump chain, mean-field ODE, curve fitting and bootstrap intervals."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy import linalg, optimize, sparse
from scipy.special import gammaln

from information_diffusion.statistical_framework import block_bootstrap

N_NODES = 30
STEP_SECONDS = 30.0
HORIZON_SECONDS = 3600.0
TIME_GRID = np.arange(0.0, HORIZON_SECONDS + STEP_SECONDS / 2, STEP_SECONDS)

MIN_RATE, MAX_RATE = 1e-7, 5.0
DIFF_STEP = 1e-3
UNIFORMIZATION_TOL = 1e-12
DENSE_STATE_LIMIT = 64
INTERVAL_METHODS = ("one_step", "refit")
MAX_FAILED_DRAWS = 0.05
RK4_STEP_SECONDS = 5.0
RK4_MAX_RATE_STEP = 0.5


@dataclass(frozen=True, eq=False)
class Transition:
    """A transition of the jump process: counts move by delta at rate theta[parameter] * coefficient(counts, n).

    Fields: parameter is the name of the rate parameter; delta is the change in each compartment count; coefficient maps
    the count (or share) vector and the number of nodes n to the intensity coefficient.
    """

    parameter: str
    delta: tuple[int, ...]
    coefficient: Callable


@dataclass(frozen=True, eq=False)
class CompartmentSpec:
    """Specification of a compartment model.

    Fields: name of the model; parameters in the model's own order; compartments, the first of which holds every node
    except the seeds at the start; transitions; observed, the indices of the compartments whose share forms the
    absorption curve; seed_compartment, where the initial seeds start; default_seeds, the default number of seeds;
    start_grid, the candidate values of each parameter used to choose starting points of the fit.
    """

    name: str
    parameters: tuple[str, ...]
    compartments: tuple[str, ...]
    transitions: tuple[Transition, ...]
    observed: tuple[int, ...]
    seed_compartment: int
    default_seeds: int
    start_grid: Mapping[str, tuple[float, ...]]


def initial_counts(spec: CompartmentSpec, n_nodes: int = N_NODES, seeds: int | None = None) -> tuple[int, ...]:
    """Initial count of each compartment: n_nodes - seeds in the first, seeds in the seed compartment.

    Arguments: spec is the model; n_nodes the number of nodes (default 30); seeds the number of seeds (default from spec).
    Returns: the count vector as a tuple.
    """
    seeds = spec.default_seeds if seeds is None else seeds
    counts = [0] * len(spec.compartments)
    counts[0] = n_nodes - seeds
    counts[spec.seed_compartment] += seeds
    return tuple(counts)


def parameter_vector(spec: CompartmentSpec, theta: Mapping[str, float] | Sequence[float]) -> np.ndarray:
    """Parameter values as an array in the model's own order.

    Arguments: spec is the model; theta is a mapping from parameter name to value, or a sequence already in order.
    Returns: a float array of length len(spec.parameters); raises ValueError on a sequence of the wrong length.
    """
    if isinstance(theta, Mapping):
        return np.array([theta[name] for name in spec.parameters], dtype=float)
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(spec.parameters),):
        raise ValueError(f"{spec.name} takes parameters {spec.parameters}")
    return theta


def observed_share(spec: CompartmentSpec, counts: Sequence[float], n: float) -> float:
    """Share of the n nodes in the observed compartments of spec, given the count vector counts."""
    return sum(counts[i] for i in spec.observed) / n




def mean_field_compartments(spec: CompartmentSpec, theta: np.ndarray, times: np.ndarray = TIME_GRID, *, n_nodes: int = N_NODES, seeds: int | None = None, rk4_step: float = RK4_STEP_SECONDS) -> np.ndarray:
    """Solve the mean-field ODE dy/dt = sum_m theta_m c_m(y, 1) delta_m on shares y = x / N by RK4.

    Arguments: spec is the model; theta is one parameter vector or K of them as rows; times is the equally spaced grid in
    seconds; n_nodes and seeds set the initial shares; rk4_step is the largest RK4 step in seconds.
    Returns: compartment shares of shape (T, n_compartments, K).
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A.
    """
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    times = np.asarray(times, dtype=float)
    state = np.tile(np.asarray(initial_counts(spec, n_nodes, seeds), dtype=float)[:, None] / n_nodes, (1, theta.shape[0]))
    rate = dict(zip(spec.parameters, theta.T, strict=True))

    def derivative(y: np.ndarray) -> np.ndarray:
        flow = np.zeros_like(y)
        for transition in spec.transitions:
            term = rate[transition.parameter] * transition.coefficient(y, 1.0)
            for j, jump in enumerate(transition.delta):
                if jump:
                    flow[j] += jump * term
        return flow

    substeps = max(1, math.ceil((times[1] - times[0]) / rk4_step - 1e-9), math.ceil((times[1] - times[0]) * float(theta.max()) / RK4_MAX_RATE_STEP))
    h = (times[1] - times[0]) / substeps
    trajectory = np.empty((times.size, len(spec.compartments), theta.shape[0]))
    trajectory[0] = state
    for k in range(1, times.size):
        for _ in range(substeps):
            k1 = derivative(state)
            k2 = derivative(state + 0.5 * h * k1)
            k3 = derivative(state + 0.5 * h * k2)
            k4 = derivative(state + h * k3)
            state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[k] = state
    return trajectory


def mean_field_curve(spec: CompartmentSpec, theta: np.ndarray, times: np.ndarray = TIME_GRID, **kwargs) -> np.ndarray:
    """Observed share under the mean-field ODE.

    Arguments: spec is the model; theta is one parameter vector or K of them as rows; times is the grid in seconds;
    kwargs are passed to mean_field_compartments.
    Returns: shape (T,) for one parameter vector, (K, T) for several.
    """
    single = np.asarray(theta, dtype=float).ndim == 1
    trajectory = mean_field_compartments(spec, theta, times, **kwargs)
    curve = trajectory[:, list(spec.observed)].sum(axis=1).T
    return curve[0] if single else curve




class FiniteChain:
    """Finite-N Markov jump process on all count vectors reachable from the initial state, with its generator assembled once.

    Constructor arguments: spec is the model; n_nodes the number of nodes (default 30); seeds the number of seeds
    (default from spec).
    """

    def __init__(self, spec: CompartmentSpec, n_nodes: int = N_NODES, seeds: int | None = None):
        self.spec = spec
        self.n_nodes = n_nodes
        start = initial_counts(spec, n_nodes, seeds)
        index = {start: 0}
        queue = deque([start])
        entries = {}
        parameter_index = {name: i for i, name in enumerate(spec.parameters)}
        while queue:
            state = queue.popleft()
            i = index[state]
            entries.setdefault((i, i), [0.0] * len(spec.parameters))
            for transition in spec.transitions:
                c = float(transition.coefficient(state, n_nodes))
                if c <= 0:
                    continue
                target = tuple(x + d for x, d in zip(state, transition.delta, strict=True))
                if min(target) < 0:
                    raise ValueError(f"{spec.name}: transition leaves the state space from {state}")
                if target not in index:
                    index[target] = len(index)
                    queue.append(target)
                j = index[target]
                par = parameter_index[transition.parameter]
                entries.setdefault((j, i), [0.0] * len(spec.parameters))[par] += c
                entries[(i, i)][par] -= c
        self.states = np.array(sorted(index, key=index.get), dtype=float)
        self.n_states = len(index)
        order = sorted(entries)
        self._csr_rows = np.fromiter((row for row, _ in order), dtype=np.int64, count=len(order))
        self._csr_cols = np.fromiter((col for _, col in order), dtype=np.int64, count=len(order))
        self._coef = np.array([entries[key] for key in order], dtype=float)
        self._diagonal = self._csr_rows == self._csr_cols
        indptr = np.zeros(self.n_states + 1, dtype=np.int64)
        np.add.at(indptr, self._csr_rows + 1, 1)
        self._indptr = np.cumsum(indptr)
        self.start_index = 0
        self.observed = self.states[:, list(spec.observed)].sum(axis=1) / n_nodes
        self.shares = self.states / n_nodes

    def _step_operator(self, theta: np.ndarray, dt: float) -> tuple[sparse.csr_matrix, np.ndarray]:
        operator, rate = self._uniformized_operator(theta)
        lam = rate * dt
        terms = np.arange(0, int(lam + 12.0 * math.sqrt(lam + 1.0) + 40.0))
        weights = np.exp(terms * np.log(max(lam, 1e-300)) - lam - gammaln(terms + 1.0))
        last = int(np.flatnonzero(weights > UNIFORMIZATION_TOL * weights.max()).max()) + 1
        return operator, weights[:last]

    def distributions(self, theta: Sequence[float], times: np.ndarray = TIME_GRID) -> np.ndarray:
        """State probabilities on the grid, by uniformisation of the forward equation p'(t) = p(t) Q(theta).

        Arguments: theta is the parameter vector; times is the equally spaced grid in seconds.
        Returns: shape (T, n_states).
        Reference: Jensen (1953), Skandinavisk Aktuarietidskrift.
        """
        theta = np.asarray(theta, dtype=float)
        dt = float(times[1] - times[0])
        operator, weights = self._step_operator(theta, dt)
        probabilities = np.zeros(self.n_states)
        probabilities[self.start_index] = 1.0
        out = np.empty((len(times), self.n_states))
        out[0] = probabilities
        for k in range(1, len(times)):
            accumulated = weights[0] * probabilities
            v = probabilities
            for w in weights[1:]:
                v = operator @ v
                accumulated += w * v
            probabilities = accumulated
            out[k] = probabilities
        return out

    def mean_curve(self, theta: Sequence[float], times: np.ndarray = TIME_GRID) -> np.ndarray:
        """Exact finite-N expected observed share on the grid.

        Uses one uniformised sequence over the whole horizon: with s_j = observed . (start P^j),
        m(t) = sum_j Poisson(j; L t) s_j. Chains with at most 64 states use a dense matrix exponential per grid step.
        Arguments: theta is the parameter vector; times is the equally spaced grid in seconds.
        Returns: the expected observed share at each time.
        Reference: Grassmann (1977), Computers and Operations Research.
        """
        theta = np.asarray(theta, dtype=float)
        dt = float(times[1] - times[0])
        if self.n_states <= DENSE_STATE_LIMIT:
            return self._dense_mean_curve(theta, times, dt)
        operator, rate = self._uniformized_operator(theta)
        horizon = float(times[-1])
        total = rate * horizon
        n_terms = int(total + 12.0 * math.sqrt(total + 1.0) + 40.0)
        sums = np.empty(n_terms)
        vector = np.zeros(self.n_states)
        vector[self.start_index] = 1.0
        observed = self.observed
        for j in range(n_terms):
            sums[j] = observed @ vector
            vector = operator @ vector
        terms = np.arange(n_terms, dtype=float)
        log_factorials = gammaln(terms + 1.0)
        out = np.empty(len(times))
        for k, t in enumerate(times):
            intensity = rate * t
            weights = np.exp(terms * math.log(max(intensity, 1e-300)) - intensity - log_factorials) if intensity > 0 else np.eye(1, n_terms, 0).ravel()
            out[k] = weights @ sums
        return out

    def _uniformized_operator(self, theta: np.ndarray) -> tuple[sparse.csr_matrix, float]:
        data = self._coef @ theta
        rate = max(-data[self._diagonal].min() * 1.01, 1e-12)
        data = data / rate
        data[self._diagonal] += 1.0
        return sparse.csr_matrix((data, self._csr_cols, self._indptr), shape=(self.n_states, self.n_states)), rate

    def _dense_mean_curve(self, theta: np.ndarray, times: np.ndarray, dt: float) -> np.ndarray:
        generator = np.zeros((self.n_states, self.n_states))
        generator[self._csr_rows, self._csr_cols] = self._coef @ theta
        propagator = linalg.expm(generator * dt)
        probabilities = np.zeros(self.n_states)
        probabilities[self.start_index] = 1.0
        out = np.empty(len(times))
        out[0] = self.observed @ probabilities
        for k in range(1, len(times)):
            probabilities = propagator @ probabilities
            out[k] = self.observed @ probabilities
        return out


@lru_cache(maxsize=16)
def chain_for(spec: CompartmentSpec, n_nodes: int = N_NODES) -> FiniteChain:
    """Cached FiniteChain for spec on n_nodes nodes with the default seeds."""
    return FiniteChain(spec, n_nodes)




@dataclass
class CurveFit:
    """Fit of a compartment model to absorption curves.

    Fields: parameters, the parameter names; estimate, lower and upper, the estimates and 95% percentile bootstrap bounds
    (NaN bounds when more than 5% of draws fail); bootstrap_log, the bootstrap draws of the log parameters; jacobian of
    the residuals at the fit; ssr, the sum of squared residuals; converged; mean_model, "chain" or "ode"; n_events.
    """

    parameters: tuple[str, ...]
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    bootstrap_log: np.ndarray
    jacobian: np.ndarray
    ssr: float
    converged: bool
    mean_model: str
    n_events: int

    @property
    def information_condition_number(self) -> float:
        """Condition number of J'J, the Gauss-Newton information matrix at the fit."""
        return float(np.linalg.cond(self.jacobian.T @ self.jacobian))


def mean_curve(curves: np.ndarray) -> np.ndarray:
    """Equal-weight mean of the curves across events (rows), ignoring NaN; each event counts once. Returns one value per grid point."""
    return np.nanmean(np.asarray(curves, dtype=float), axis=0)


def _curve_function(spec: CompartmentSpec, mean_model: str, n_nodes: int, times: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    if mean_model == "chain":
        chain = chain_for(spec, n_nodes)
        return lambda theta: chain.mean_curve(theta, times)
    if mean_model == "ode":
        return lambda theta: mean_field_curve(spec, theta, times, n_nodes=n_nodes)
    raise ValueError(f"unknown mean model {mean_model!r}")


@lru_cache(maxsize=16)
def _start_bank(spec: CompartmentSpec, mean_model: str, n_nodes: int = N_NODES) -> tuple[np.ndarray, np.ndarray]:
    grids = [spec.start_grid[name] for name in spec.parameters]
    candidates = np.array(list(product(*grids)), dtype=float)
    if mean_model == "ode":
        return candidates, mean_field_curve(spec, candidates, TIME_GRID, n_nodes=n_nodes, rk4_step=10.0)
    chain = chain_for(spec, n_nodes)
    return candidates, np.array([chain.mean_curve(c) for c in candidates])


def starting_points(spec: CompartmentSpec, target: np.ndarray, n_nodes: int, n_starts: int, mean_model: str = "chain") -> list[np.ndarray]:
    """Starting points of the fit chosen from the start grid of spec.

    Grid candidates are ranked by squared distance of their mean curve to target; a candidate is kept only if its log
    parameters differ from every kept one by more than 0.75 in some coordinate.
    Arguments: spec is the model; target the mean curve on the standard grid; n_nodes the number of nodes; n_starts the
    number of starts; mean_model is "chain" or "ode".
    Returns: a list of log-parameter vectors.
    """
    candidates, curves = _start_bank(spec, mean_model, n_nodes)
    if curves.shape[1] != target.size:
        raise ValueError("targets must be on the standard grid")
    order = np.argsort(np.sum((curves[:, 1:] - target[1:]) ** 2, axis=1))
    chosen: list[np.ndarray] = []
    for k in order:
        u = np.log(candidates[k])
        if all(np.max(np.abs(u - other)) > 0.75 for other in chosen):
            chosen.append(u)
        if len(chosen) == n_starts:
            break
    return chosen


def fit_curve(spec: CompartmentSpec, target: np.ndarray, *, mean_model: str = "chain", n_nodes: int = N_NODES, times: np.ndarray = TIME_GRID, n_starts: int | None = None, max_nfev: int = 80, start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """Fit a model to one mean curve by bounded nonlinear least squares in u = log(theta), SSR(u) = sum_t (ybar(t) - m(t; exp u))^2.

    The point t = 0 is excluded and the best of several starts is kept.
    Arguments: spec is the model; target the mean curve; mean_model is "chain" (exact finite-N expectation) or "ode"
    (mean field); n_nodes the number of nodes; times the grid in seconds (the standard grid TIME_GRID unless start is
    given, since the start grid is computed on it); n_starts the number of starts (default 1, 2 or 5 for one, two or
    more parameters); max_nfev the evaluation limit per start; start a single log-parameter start that replaces the grid.
    Returns: (log parameters, Jacobian, SSR, converged).
    Reference: Marquardt (1963), SIAM J. Appl. Math.; Branch, Coleman and Li (1999), SIAM J. Sci. Comput.
    """
    target = np.asarray(target, dtype=float)
    model = _curve_function(spec, mean_model, n_nodes, times)
    lower, upper = math.log(MIN_RATE), math.log(MAX_RATE)
    n_starts = n_starts if n_starts is not None else {1: 1, 2: 2}.get(len(spec.parameters), 5)
    starts = [np.asarray(start, dtype=float)] if start is not None else starting_points(spec, target, n_nodes, n_starts, mean_model)

    def residual(u: np.ndarray) -> np.ndarray:
        return model(np.exp(u))[1:] - target[1:]

    best = None
    for u0 in starts:
        solution = optimize.least_squares(residual, np.clip(u0, lower + 1e-6, upper - 1e-6), jac="2-point", bounds=(lower, upper), method="trf", diff_step=DIFF_STEP, x_scale=1.0, xtol=1e-6, ftol=1e-8, gtol=1e-8, max_nfev=max_nfev)
        if best is None or solution.cost < best.cost:
            best = solution
    return best.x, best.jac, float(2.0 * best.cost), bool(best.status > 0)


def bootstrap_draws(curves: np.ndarray, u_hat: np.ndarray, jacobian: np.ndarray, spec: CompartmentSpec, *, draws: int, rng: np.random.Generator, mean_model: str, n_nodes: int, times: np.ndarray, interval_method: str = "refit") -> np.ndarray:
    """Bootstrap draws of the log parameters by resampling events with replacement.

    With interval_method "refit" each resampled mean curve is refitted from u_hat; with "one_step" the draw is the
    linearisation u*_b = u_hat + (J'J)^(-1) J' (ybar*_b - m(u_hat)).
    Arguments: curves has one row per event; u_hat and jacobian come from the fit; spec is the model; draws the number of
    draws; rng the generator; mean_model, n_nodes and times as in fit_curve; interval_method "refit" or "one_step".
    Returns: a draws x p array; a draw whose refit fails is NaN.
    Reference: Efron (1979), Annals of Statistics; Moulton and Zeger (1991), Biometrics (one-step bootstrap).
    """
    if interval_method not in INTERVAL_METHODS:
        raise ValueError(f"unknown interval method {interval_method!r}")
    n_events = curves.shape[0]
    resamples = np.stack([block_bootstrap.event_indices(n_events, rng) for _ in range(draws)])
    means = np.nanmean(curves[resamples], axis=1)
    lower, upper = math.log(MIN_RATE), math.log(MAX_RATE)
    if interval_method == "refit":
        out = np.full((draws, u_hat.size), np.nan)
        for b, target in enumerate(means):
            try:
                u, _, _, converged = fit_curve(spec, target, mean_model=mean_model, n_nodes=n_nodes, times=times, start=u_hat)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError):
                continue
            if converged:
                out[b] = u
        return out
    model = _curve_function(spec, mean_model, n_nodes, times)
    gain = np.linalg.pinv(jacobian.T @ jacobian) @ jacobian.T
    return np.clip(u_hat + (gain @ (means[:, 1:] - model(np.exp(u_hat))[1:]).T).T, lower, upper)


def fit_events(spec: CompartmentSpec, curves: np.ndarray, *, bootstrap: int = 199, rng: np.random.Generator | None = None, mean_model: str = "chain", n_nodes: int = N_NODES, times: np.ndarray = TIME_GRID, n_starts: int | None = None, interval_method: str = "refit") -> CurveFit:
    """Fit a model to the equal-weight mean of event curves, with 95% percentile bootstrap intervals.

    Arguments: spec is the model; curves has one row per event and one column per grid point; bootstrap is the number of
    draws (default 199); rng the generator; mean_model, n_nodes, times and n_starts as in fit_curve; interval_method
    "refit" or "one_step".
    Returns: a CurveFit; its bounds are NaN when more than 5% of the draws fail.
    """
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 2 or curves.shape[1] != times.size:
        raise ValueError("curves must have one row per event and one column per grid point")
    rng = np.random.default_rng() if rng is None else rng
    target = mean_curve(curves)
    u_hat, jacobian, ssr, converged = fit_curve(spec, target, mean_model=mean_model, n_nodes=n_nodes, times=times, n_starts=n_starts)
    draws = bootstrap_draws(curves, u_hat, jacobian, spec, draws=bootstrap, rng=rng, mean_model=mean_model, n_nodes=n_nodes, times=times, interval_method=interval_method)
    if np.mean(~np.isfinite(draws).all(axis=1)) > MAX_FAILED_DRAWS:
        lo = hi = np.full(u_hat.size, np.nan)
    else:
        lo, hi = np.nanquantile(draws, [0.025, 0.975], axis=0)
    return CurveFit(spec.parameters, np.exp(u_hat), np.exp(lo), np.exp(hi), draws, jacobian, ssr, converged, mean_model, curves.shape[0])
