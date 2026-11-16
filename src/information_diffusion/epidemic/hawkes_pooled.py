"""Exponential Hawkes model pooled across instruments and events with instrument fixed effects, and its forecast of counts per step."""

from __future__ import annotations

import numpy as np
from scipy import optimize

from information_diffusion.epidemic.hawkes_exponential import ExponentialHawkes, exp_states

SUBSTEPS = 30


class PooledInstrumentHawkes:
    """Exponential Hawkes model fitted jointly over instruments i and events e with instrument fixed effects.

    lambda_ie(t) = exp(alpha_i + sum_{j >= 2} g_j B_j(t)) + eta beta sum_{t_l < t} exp(-beta (t - t_l)), with B_j the
    B-spline basis of the shared baseline (which sums to one, so the first basis is dropped). The log-likelihood is the sum
    over instruments and events, with the spline penalty counted once. Parameters are (alpha_1..I, g_2..K,
    logit(eta / 0.9995), log beta).
    Constructor arguments: problems[i][e] is the ExponentialHawkes problem of instrument i in event e, all sharing one
    baseline and window.
    Reference: Hawkes (1971), Biometrika; Ozaki (1979); De Boor (2001), A Practical Guide to Splines.
    """

    def __init__(self, problems: list[list[ExponentialHawkes]]):
        self.problems = problems
        self.reference = problems[0][0]
        self.baseline = self.reference.baseline
        self.instruments = len(problems)
        self.K = self.baseline.n_coef
        self.size = self.instruments + self.K - 1 + 2

    def instrument_x(self, x: np.ndarray, i: int) -> np.ndarray:
        """Single-problem parameters of instrument i from the pooled vector x: b = alpha_i (1, ..., 1) + (0, g), then the kernel."""
        x = np.asarray(x, dtype=float)
        b = x[i] + np.concatenate(([0.0], x[self.instruments : self.instruments + self.K - 1]))
        return np.concatenate((b, x[-2:]))

    def nll_grad(self, x):
        """Pooled penalised negative log-likelihood at x and its gradient; infinite when any problem is infeasible."""
        x = np.asarray(x, dtype=float)
        value, gradient, with_penalty = 0.0, np.zeros(self.size), 0
        penalty, penalty_gradient = self.baseline.penalty_and_grad(self.instrument_x(x, 0)[: self.K])
        for i, events in enumerate(self.problems):
            xi = self.instrument_x(x, i)
            for problem in events:
                part_value, part = problem.nll_grad(xi)
                if not np.isfinite(part_value):
                    return np.inf, np.full(self.size, np.nan)
                part_b, part_kernel = part[: self.K], part[self.K :]
                with_penalty += 1
                value += part_value
                gradient[i] += float(part_b.sum())
                gradient[self.instruments : self.instruments + self.K - 1] += part_b[1:]
                gradient[-2:] += part_kernel
        extra = with_penalty - 1
        value -= extra * penalty
        gradient[self.instruments : self.instruments + self.K - 1] -= extra * penalty_gradient[1:]
        return value, gradient

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds of the pooled vector: -20 and 20 for the baseline parameters, then the kernel bounds. Returns (low, high)."""
        limits = self.reference.kernel_bounds()
        low = np.concatenate((np.full(self.size - 2, -20.0), [limit[0] for limit in limits]))
        high = np.concatenate((np.full(self.size - 2, 20.0), [limit[1] for limit in limits]))
        return low, high

    def start_points(self) -> list[np.ndarray]:
        """Data-driven starting vectors: the baseline from the pooled events and instrument levels from their mean counts, one per kernel start."""
        window = self.reference.window
        per_instrument = np.array([np.mean([p.n for p in events]) for events in self.problems], dtype=float)
        pooled_times = np.concatenate([p.t for events in self.problems for p in events])
        points = []
        for kernel in self.reference.kernel_starts():
            b = self.baseline.initial(pooled_times, max(per_instrument.mean(), 1.0) * (1.0 - float(kernel[0])) / window)
            levels = b[0] + np.log(np.maximum(per_instrument, 0.5) / max(per_instrument.mean(), 1.0))
            points.append(np.concatenate((levels, b[1:] - b[0], self.reference.encode_kernel(kernel))))
        return points

    def fit(self, start=None) -> np.ndarray | None:
        """Maximum penalised likelihood fit by L-BFGS-B from start, or from the best of start_points.

        Returns: the pooled parameter vector, or None when no finite fit is found.
        """
        low, high = self.bounds()
        candidates = [np.clip(start, low, high)] if start is not None else [np.clip(point, low, high) for point in self.start_points()]
        best, best_value = None, np.inf
        for point in candidates:
            value = self.nll_grad(point)[0]
            if value < best_value:
                best, best_value = point, value
        if best is None:
            return None
        result = optimize.minimize(self.nll_grad, best, jac=True, method="L-BFGS-B", bounds=list(zip(low, high)), options=dict(maxiter=2000, maxfun=8000, ftol=1e-11, gtol=1e-6, maxcor=25))
        estimate = np.clip(result.x, low, high)
        return estimate if np.isfinite(self.nll_grad(estimate)[0]) else None


def step_forecasts(problem: ExponentialHawkes, x: np.ndarray, edges: np.ndarray, substeps: int = SUBSTEPS) -> np.ndarray:
    """Expected event count in each step [s, s + h) given the events before s, under an exponential Hawkes model.

    Because an event's own merge bin is excluded, the observable branching ratio is eta_obs = eta exp(-beta d0), d0 the
    merge resolution. The expected excitation solves m' = -beta (1 - eta_obs) m + eta_obs beta mu(u) from
    m(s) = eta beta sum_{t_l < s} exp(-beta (s - t_l)), integrated exactly on sub-steps with mu held at each midpoint.
    Arguments: problem is the fitted ExponentialHawkes problem; x its parameters; edges the equally spaced step edges in
    seconds; substeps the number of sub-steps per step (default 30).
    Returns: the expected count in each step.
    Reference: Hawkes (1971), Biometrika; Ozaki (1979).
    """
    x = np.asarray(x, dtype=float)
    eta, beta = float(problem.branching(x)), float(problem.beta(x))
    observable = eta * np.exp(-beta * problem.lag_floor)
    starts = np.asarray(edges, dtype=float)[:-1]
    width = float(edges[1] - edges[0])
    substep = width / substeps
    past = np.zeros(starts.size)
    states = exp_states(problem.t, beta)
    last = np.searchsorted(problem.t, starts, side="left") - 1
    known = last >= 0
    past[known] = np.exp(-beta * (starts[known] - problem.t[last[known]])) * (1.0 + states[last[known]])
    midpoints = starts[:, None] + (np.arange(substeps) + 0.5) * substep
    rates = problem.baseline.rate(x[: problem.K], midpoints.ravel()).reshape(midpoints.shape)
    gamma = beta * (1.0 - observable)
    decay = np.exp(-gamma * substep)
    integral = (1.0 - decay) / gamma
    expected = np.zeros(starts.size)
    excitation = eta * beta * past
    for q in range(substeps):
        mu = rates[:, q]
        stationary = observable * beta * mu / gamma
        expected += mu * substep + excitation * integral + stationary * (substep - integral)
        excitation = excitation * decay + stationary * (1.0 - decay)
    return expected
