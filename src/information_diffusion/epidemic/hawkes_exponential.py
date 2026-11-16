"""Univariate Hawkes process with an exponential kernel, and the shared penalised likelihood engine for Hawkes fits."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, optimize
from scipy.special import expit

from information_diffusion.compiled import njit
from information_diffusion.simulation import hawkes as sim

MERGE_RESOLUTION_SECONDS = 0.001

ETA_CEILING = 0.9995
BOUND_ETA = 12.0


@njit(cache=True)
def _ozaki_states(rates, weights, states):
    state = 0.0
    for i in range(rates.size):
        state = rates[i] * (weights[i] + state)
        states[i + 1] = state


@njit(cache=True)
def _ozaki_states_moments(gaps, rates, weights, states, moments):
    state = 0.0
    moment = 0.0
    for i in range(rates.size):
        base = weights[i] + state
        moment = rates[i] * (gaps[i] * base + moment)
        state = rates[i] * base
        states[i + 1] = state
        moments[i + 1] = moment


@njit(cache=True)
def _ozaki_states_by_column(rates, weights, states):
    for source in range(weights.shape[1]):
        state = 0.0
        for i in range(rates.size):
            state = rates[i] * (weights[i, source] + state)
            states[i + 1, source] = state


@njit(cache=True)
def _ozaki_states_moments_by_column(gaps, rates, weights, states, moments):
    for source in range(weights.shape[1]):
        state = 0.0
        moment = 0.0
        for i in range(rates.size):
            base = weights[i, source] + state
            moment = rates[i] * (gaps[i] * base + moment)
            state = rates[i] * base
            states[i + 1, source] = state
            moments[i + 1, source] = moment


def exp_states(times, beta: float, weights=None, deriv: bool = False):
    """Past-sum states of an exponential kernel by the Ozaki recursion A_i = r_i (w_{i-1} + A_{i-1}), r_i = exp(-beta (t_i - t_{i-1})).

    Arguments: times are sorted event times in seconds; beta the decay rate (per second); weights the event weights
    (1 each when None); deriv also returns C_i = sum_{j<i} (t_i - t_j) w_j exp(-beta (t_i - t_j)).
    Returns: A with A_i = sum_{j<i} w_j exp(-beta (t_i - t_j)), or (A, C) when deriv is True.
    Reference: Ozaki (1979), Ann. Inst. Stat. Math.
    """
    t = np.asarray(times, dtype=float)
    n = t.size
    A = np.zeros(n)
    C = np.zeros(n) if deriv else None
    if n < 2:
        return (A, C) if deriv else A
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    gaps = np.diff(t)
    rates = np.exp(-beta * gaps)
    if deriv:
        _ozaki_states_moments(gaps, rates, w, A, C)
        return A, C
    _ozaki_states(rates, w, A)
    return A


def exp_states_by_column(times, beta: float, weights, deriv: bool = False):
    """exp_states for every column of a weight matrix on one shared time grid.

    Arguments: times are sorted event times in seconds; beta the decay rate; weights has shape (events, sources);
    deriv also returns the beta-derivative states.
    Returns: A of shape (events, sources), or (A, C) when deriv is True.
    """
    t = np.asarray(times, dtype=float)
    w = np.asarray(weights, dtype=float)
    A = np.zeros(w.shape)
    C = np.zeros(w.shape) if deriv else None
    if t.size >= 2:
        gaps = np.diff(t)
        rates = np.exp(-beta * gaps)
        if deriv:
            _ozaki_states_moments_by_column(gaps, rates, w, A, C)
        else:
            _ozaki_states_by_column(rates, w, A)
    return (A, C) if deriv else A


def _doubling_recursion(rate, increments):
    n = rate.size
    a = increments.copy()
    r = rate.copy()
    lag = 1
    while lag < n and r.max() > 1e-18:
        keep_a = a[lag:] + r[lag:] * a[:-lag]
        r_next = np.zeros(n)
        r_next[lag:] = r[lag:] * r[:-lag]
        a[lag:] = keep_a
        r = r_next
        lag *= 2
    return a


def _exp_states_doubling(t, beta, w, deriv):
    n = t.size
    gaps = np.diff(t)
    r = np.exp(-beta * gaps)
    a = _doubling_recursion(r, r * w[:-1])
    A = np.concatenate(([0.0], a))
    if not deriv:
        return A
    inhomogeneous = r * gaps * (w[:-1] + A[:-1])
    c = _doubling_recursion(r, inhomogeneous)
    return A, np.concatenate(([0.0], c))


@dataclass
class FitResult:
    """A fitted Hawkes model.

    Fields: problem, the HawkesProblem fitted; x, the transformed parameters (baseline coefficients, then kernel block);
    nll, the penalised negative log-likelihood at x; hessian of the objective (None when no start was finite) and its
    inverse cov (None when the Hessian is not positive definite); converged; newton_decrement; n_evaluations; message from the optimiser; extras for any additional output.
    """

    problem: "HawkesProblem"
    x: np.ndarray
    nll: float
    hessian: np.ndarray | None
    cov: np.ndarray | None
    converged: bool
    newton_decrement: float
    n_evaluations: int
    message: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def branching(self) -> float:
        """Branching ratio of the fitted model."""
        return self.problem.branching(self.x)

    def mean_baseline(self) -> float:
        """Mean baseline rate of the fitted model over the window (events per second)."""
        return self.problem.baseline.mean(self.x[: self.problem.baseline.n_coef])

    def mean_baseline_interval(self, level: float = 0.95):
        """Delta-method interval for the mean baseline, built on the log scale from the observed information.

        Arguments: level is the coverage (default 0.95).
        Returns: (lower, upper), NaN when the covariance is not available.
        """
        if self.cov is None:
            return np.nan, np.nan
        k = self.problem.baseline.n_coef
        grad = self.problem.baseline.mean_log_gradient(self.x[:k])
        variance = float(grad @ self.cov[:k, :k] @ grad)
        if not np.isfinite(variance) or variance <= 0:
            return np.nan, np.nan
        z = _normal_quantile(level)
        centre = np.log(self.mean_baseline())
        return float(np.exp(centre - z * np.sqrt(variance))), float(np.exp(centre + z * np.sqrt(variance)))

    def kernel_interval(self, index: int, level: float = 0.95):
        """Wald interval for one kernel parameter in its transformed coordinate.

        Arguments: index is the position in the kernel block; level is the coverage (default 0.95).
        Returns: (lower, upper) on the transformed scale, NaN when the covariance is not available.
        """
        if self.cov is None:
            return np.nan, np.nan
        k = self.problem.baseline.n_coef + index
        variance = self.cov[k, k]
        if not np.isfinite(variance) or variance <= 0:
            return np.nan, np.nan
        z = _normal_quantile(level)
        half = z * np.sqrt(variance)
        return self.x[k] - half, self.x[k] + half


def _normal_quantile(level: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(0.5 + level / 2))


def finite_difference_hessian(gradient, x: np.ndarray, step: float = 1e-4) -> np.ndarray:
    """Symmetrised central-difference Hessian of an analytic gradient.

    Arguments: gradient maps a parameter vector to its gradient; x is the point; step is the relative step, scaled by
    max(1, |x_j|).
    Returns: the p x p Hessian.
    """
    p = x.size
    h = step * np.maximum(1.0, np.abs(x))
    hessian = np.empty((p, p))
    for j in range(p):
        forward = x.copy()
        backward = x.copy()
        forward[j] += h[j]
        backward[j] -= h[j]
        hessian[j] = (gradient(forward) - gradient(backward)) / (2.0 * h[j])
    return 0.5 * (hessian + hessian.T)


class HawkesProblem:
    """Penalised negative log-likelihood of a univariate Hawkes model in transformed coordinates, with fitting.

    The parameter vector is x = (baseline coefficients b, kernel block k). Subclasses define the kernel block through
    kernel_terms(k) -> (phi, dphi, comp, dcomp): the excitation at each event (n), its kernel-block gradient (q x n), the
    compensator mass of the excitation over the window and its gradient (q).
    Constructor arguments: times are the event times in seconds from the window start; window is the window length in
    seconds; baseline is a baseline object (constant or log-spline); marks are the print counts of the merged events;
    resolution is the merge bin width in seconds (default 0.001; None for the continuous-time likelihood); print_weighted
    makes an event of m prints count and excite as m prints.
    """

    n_kernel = 0
    name = "hawkes"

    def __init__(self, times, window: float, baseline, marks=None, resolution: float | None = MERGE_RESOLUTION_SECONDS, print_weighted: bool = False):
        self.t = np.asarray(times, dtype=float)
        self.n = self.t.size
        self.window = float(window)
        self.baseline = baseline
        self.marks = None if marks is None else np.asarray(marks, dtype=float)
        self.resolution = resolution
        self.print_weighted = bool(print_weighted)
        if self.print_weighted and self.marks is None:
            raise ValueError("print weighting needs the print count of each merged event")
        self.prints = self.marks if self.print_weighted else None
        self.event_weight = self.marks if self.print_weighted else np.ones(self.n)
        self.K = baseline.n_coef
        self.design = baseline.design(self.t)
        self.design_transpose = None if baseline.constant else self.design.T
        self.lag_floor = 0.0 if resolution is None else float(resolution)
        self.remaining = np.maximum(self.window - self.t, self.lag_floor)
        self.evaluations = 0
        self._prepare()

    def _prepare(self) -> None:
        pass

    def kernel_terms(self, k):
        """Kernel terms (phi, dphi, comp, dcomp) for the kernel block k; defined by each subclass."""
        raise NotImplementedError

    def kernel_starts(self) -> list[np.ndarray]:
        """Starting kernel parameters on the natural scale; defined by each subclass."""
        raise NotImplementedError

    def kernel_bounds(self) -> list[tuple[float, float]]:
        """Bounds of the kernel block in transformed coordinates; defined by each subclass."""
        raise NotImplementedError

    def branching(self, x) -> float:
        """Branching ratio at parameters x; defined by each subclass."""
        raise NotImplementedError

    def simulate(self, x, rng, max_events=None):
        """Simulate one raw event set (before merging) at parameters x.

        Arguments: x the parameters; rng the generator; max_events an optional cap on the number of events.
        Returns: event times, or (times, marks) for marked models.
        """
        raise NotImplementedError

    def spawn(self, times, marks=None):
        """New problem of the same model, window, baseline and settings on other event times and marks."""
        return self.__class__(times, self.window, self.baseline, marks=marks, resolution=self.resolution, print_weighted=self.print_weighted, **self._spawn_kwargs())

    def _spawn_kwargs(self) -> dict:
        return {}

    def nll_grad(self, x):
        """Penalised negative log-likelihood of the merged (binned) process and its gradient.

        With resolution d, each event bin has occupancy probability 1 - exp(-d lambda_i) and the compensator excludes
        an event's own bin: l = sum_i log(expm1(d lambda_i) / d) - int mu - sum_j int_{d}^{T - t_j} phi.
        With resolution None the continuous-time likelihood is used. With print weighting, an event of m_i prints
        excites as m_i prints and the bin count is Poisson: l = sum_i m_i log lambda_i - int mu - sum_j m_j int phi.
        Arguments: x is the transformed parameter vector.
        Returns: (value, gradient); the value is infinite when the intensity is not positive at some event.
        Reference: Ogata (1978), Ann. Inst. Stat. Math.; Haslinger, Pipa and Brown (2010), Neural Computation.
        """
        self.evaluations += 1
        x = np.asarray(x, dtype=float)
        b, k = x[: self.K], x[self.K :]
        if self.baseline.constant:
            mu = np.exp(b[0]) * np.ones(self.n)
        else:
            mu = np.exp(self.design @ b)
        phi, dphi, comp, dcomp = self.kernel_terms(k)
        lam = mu + phi
        if np.any(~np.isfinite(lam)) or np.any(lam <= 0.0):
            return np.inf, np.full(x.size, np.nan)
        integral, integral_grad = self.baseline.integral_and_grad(b)
        penalty, penalty_grad = self.baseline.penalty_and_grad(b)
        if self.print_weighted:
            log_rate = self.event_weight * np.log(lam)
            weight = self.event_weight / lam
        elif self.resolution is None:
            log_rate = np.log(lam)
            weight = 1.0 / lam
        else:
            d = self.resolution
            r = np.minimum(d * lam, 500.0)
            log_rate = np.log(np.expm1(r) / d)
            weight = d / (-np.expm1(-r))
        value = -float(np.sum(log_rate)) + integral + float(comp) + penalty
        score_b = np.array([float(np.sum(mu * weight))]) if self.baseline.constant else self.design_transpose @ (mu * weight)
        grad_b = integral_grad + penalty_grad - score_b
        grad_k = dcomp - np.sum(dphi * weight, axis=1) if self.n_kernel else np.zeros(0)
        return value, np.concatenate((grad_b, grad_k))

    def nll(self, x) -> float:
        """Penalised negative log-likelihood at the transformed parameters x."""
        return float(self.nll_grad(x)[0])

    def bounds(self):
        """Lower and upper bounds of x: -20 and 20 for the baseline coefficients, then the kernel bounds. Returns (low, high)."""
        low = np.concatenate((-20.0 * np.ones(self.K), [limit[0] for limit in self.kernel_bounds()]))
        high = np.concatenate((20.0 * np.ones(self.K), [limit[1] for limit in self.kernel_bounds()]))
        return low, high

    def start_points(self) -> list[np.ndarray]:
        """Starting parameter vectors, one per kernel start: the baseline at rate n (1 - eta) / T for that start's eta."""
        points = []
        for start in self.kernel_starts():
            eta0 = float(start[0])
            rate = self.n * (1.0 - eta0) / self.window
            b = self.baseline.initial(self.t, rate)
            points.append(np.concatenate((b, self.encode_kernel(start))))
        return points

    def encode_kernel(self, natural) -> np.ndarray:
        """Kernel block in transformed coordinates from natural kernel parameters; defined by each subclass."""
        raise NotImplementedError

    def fit(self, x0=None, maxiter: int = 2000) -> FitResult:
        """Maximum penalised likelihood fit by L-BFGS-B with analytic gradients, polished by Newton steps.

        The descent starts from x0, or from the best of start_points by objective. The Hessian is a finite-difference
        Hessian of the analytic gradient (the observed information).
        Arguments: x0 an optional starting vector; maxiter the L-BFGS-B iteration limit.
        Returns: a FitResult.
        Reference: Byrd, Lu, Nocedal and Zhu (1995), SIAM J. Sci. Comput.
        """
        low, high = self.bounds()
        bounds = list(zip(low, high))
        candidates = [np.clip(np.asarray(x0, dtype=float), low, high)] if x0 is not None else [np.clip(p, low, high) for p in self.start_points()]
        best, best_value = None, np.inf
        for point in candidates:
            value = self.nll(point)
            if value < best_value:
                best, best_value = point, value
        if best is None:
            return FitResult(self, np.full(low.size, np.nan), np.nan, None, None, False, np.inf, self.evaluations, "no finite starting value")
        result = optimize.minimize(self.nll_grad, best, jac=True, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=maxiter, maxfun=4 * maxiter, ftol=1e-11, gtol=1e-6, maxcor=25))
        x = np.clip(result.x, low, high)
        return self._finish(x, result.message if isinstance(result.message, str) else str(result.message))

    def _finish(self, x, message="") -> FitResult:
        value, gradient = self.nll_grad(x)
        hessian = finite_difference_hessian(lambda v: self.nll_grad(v)[1], x)
        hessian, value, gradient, x, decrement, polished = _newton_polish(self, x, value, gradient, hessian)
        cov, decrement_final = _invert(hessian, gradient)
        converged = bool(np.isfinite(value) and cov is not None and decrement_final < 1e-1 and polished)
        return FitResult(self, x, float(value), hessian, cov, converged, float(decrement_final), self.evaluations, message)

    def refit_warm(self, x0, hessian, *, max_iter: int = 80, tolerance: float = 1e-5, chol=None):
        """Warm-started refit by BFGS from the fitted inverse Hessian, finished by L-BFGS-B when that does not converge.

        The BFGS stage stops when the Newton decrement g' H^(-1) g falls below tolerance.
        Arguments: x0 is the starting point; hessian the Hessian at the original fit; max_iter the BFGS step limit;
        tolerance the decrement threshold; chol an optional Cholesky factor of hessian.
        Returns: the refitted parameters, or None when the refit fails.
        """
        low, high = self.bounds()
        x = np.clip(np.asarray(x0, dtype=float), low, high)
        value, gradient = self.nll_grad(x)
        if not np.isfinite(value):
            return None
        try:
            factor = linalg.cho_factor(hessian) if chol is None else chol
            inverse = linalg.cho_solve(factor, np.eye(x.size))
        except linalg.LinAlgError:
            inverse = None
        if inverse is not None:
            for _ in range(max_iter):
                direction = inverse @ gradient
                decrement = float(gradient @ direction)
                if not np.isfinite(decrement) or decrement < 0.0:
                    break
                if decrement < tolerance:
                    return x
                step = 1.0
                accepted = False
                for _attempt in range(12):
                    candidate = np.clip(x - step * direction, low, high)
                    new_value, new_gradient = self.nll_grad(candidate)
                    if np.isfinite(new_value) and new_value <= value - 1e-4 * step * decrement + 1e-9 * max(abs(value), 1.0):
                        accepted = True
                        break
                    step *= 0.5
                if not accepted:
                    break
                s = candidate - x
                y = new_gradient - gradient
                curvature = float(s @ y)
                if curvature > 1e-10 * float(np.linalg.norm(s) * np.linalg.norm(y)):
                    rho = 1.0 / curvature
                    inverse_y = inverse @ y
                    inverse = inverse - rho * (np.outer(s, inverse_y) + np.outer(inverse_y, s)) + rho * (1.0 + rho * float(y @ inverse_y)) * np.outer(s, s)
                x, value, gradient = candidate, new_value, new_gradient
        result = optimize.minimize(self.nll_grad, x, jac=True, method="L-BFGS-B", bounds=list(zip(low, high)), options=dict(maxiter=800, maxfun=3000, ftol=1e-11, gtol=1e-5, maxcor=25))
        if not (np.isfinite(result.fun) and (result.success or _projected_gradient_small(self, result.x, low, high))):
            return None
        return np.clip(result.x, low, high)


def _projected_gradient_small(problem, x, low, high, tolerance: float = 1.0) -> bool:
    gradient = problem.nll_grad(x)[1]
    projected = np.where((x <= low) & (gradient > 0), 0.0, gradient)
    projected = np.where((x >= high) & (gradient < 0), 0.0, projected)
    return float(np.max(np.abs(projected))) < tolerance


def _newton_polish(problem, x, value, gradient, hessian, steps: int = 4):
    low, high = problem.bounds()
    polished = True
    for _ in range(steps):
        cov, decrement = _invert(hessian, gradient)
        if cov is None:
            polished = False
            break
        if decrement < 1e-8:
            break
        direction = linalg.solve(hessian, gradient, assume_a="sym")
        step = 1.0
        accepted = False
        for _attempt in range(10):
            candidate = np.clip(x - step * direction, low, high)
            new_value, new_gradient = problem.nll_grad(candidate)
            if np.isfinite(new_value) and new_value <= value + 1e-7 * max(abs(value), 1.0):
                x, value, gradient = candidate, new_value, new_gradient
                accepted = True
                break
            step *= 0.5
        if not accepted:
            polished = decrement < 1.0
            break
    hessian = finite_difference_hessian(lambda v: problem.nll_grad(v)[1], x)
    return hessian, value, gradient, x, float(np.dot(gradient, linalg.solve(hessian, gradient, assume_a="sym"))) if _is_positive_definite(hessian) else np.inf, polished


def _is_positive_definite(hessian) -> bool:
    try:
        linalg.cholesky(hessian)
        return True
    except linalg.LinAlgError:
        return False


def _invert(hessian, gradient):
    if hessian is None or not np.all(np.isfinite(hessian)):
        return None, np.inf
    try:
        factor = linalg.cho_factor(hessian)
    except linalg.LinAlgError:
        return None, np.inf
    cov = linalg.cho_solve(factor, np.eye(hessian.shape[0]))
    decrement = float(gradient @ linalg.cho_solve(factor, gradient))
    return cov, decrement


class ExponentialHawkes(HawkesProblem):
    """Hawkes model with an exponential kernel, lambda(t) = mu(t) + eta beta sum_{t_j<t} exp(-beta (t - t_j)).

    The kernel block is (logit(eta / 0.9995), log beta). Constructor arguments as for HawkesProblem.
    Reference: Hawkes (1971), Biometrika.
    """

    n_kernel = 2
    name = "exponential"

    def kernel_terms(self, k):
        """Excitation at each event, its gradient, the compensator mass and its gradient for the kernel block k."""
        eta = ETA_CEILING * expit(k[0])
        deta_dk = eta * (1.0 - expit(k[0]))
        beta = float(np.exp(k[1]))
        A, C = exp_states(self.t, beta, self.prints, deriv=True)
        decay = np.exp(-beta * self.remaining)
        bin_decay = np.exp(-beta * self.lag_floor)
        phi = eta * beta * A
        dphi_eta = beta * A * deta_dk
        dphi_logbeta = eta * (A - beta * C) * beta
        mass = float(np.sum(self.event_weight * (bin_decay - decay)))
        comp = eta * mass
        dcomp_eta = deta_dk * mass
        dcomp_logbeta = eta * float(np.sum(self.event_weight * (self.remaining * decay - self.lag_floor * bin_decay))) * beta
        return phi, np.vstack((dphi_eta, dphi_logbeta)), comp, np.array([dcomp_eta, dcomp_logbeta])

    def kernel_bounds(self):
        """Bounds of the kernel block: logit(eta / 0.9995) in [-12, 12], beta in [0.001, 1000] per second."""
        return [(-BOUND_ETA, BOUND_ETA), (np.log(1e-3), np.log(1e3))]

    def kernel_starts(self):
        """Starting (eta, beta) pairs."""
        return [np.array([0.3, 1.0]), np.array([0.7, 1.0]), np.array([0.5, 4.0]), np.array([0.5, 0.2])]

    def encode_kernel(self, natural):
        """Transformed kernel block from natural parameters (eta, beta)."""
        eta, beta = natural
        return np.array([np.log(eta / ETA_CEILING / (1 - eta / ETA_CEILING)), np.log(beta)])

    def branching(self, x) -> float:
        """Branching ratio eta at parameters x."""
        return float(ETA_CEILING * expit(x[self.K]))

    def beta(self, x) -> float:
        """Kernel decay rate beta (per second) at parameters x."""
        return float(np.exp(x[self.K + 1]))

    def simulate(self, x, rng, max_events=None):
        """Simulate one raw event set at parameters x by the cluster representation; max_events caps the number of events."""
        sampler = self.baseline.sampler(x[: self.K])
        immigrants = sampler.sample(rng)
        kernel = sim.ExponentialKernel(self.beta(x))
        return sim.simulate_cluster(immigrants, kernel, self.branching(x), self.window, rng, max_events=max_events)

    def excitation_integral_increments(self, x):
        """Excitation mass between consecutive events at parameters x, for time-rescaling residuals; index 0 covers 0 to t_0."""
        eta, beta = self.branching(x), self.beta(x)
        A = exp_states(self.t, beta, self.prints)
        increments = np.zeros(self.n)
        increments[1:] = eta * ((self.event_weight[:-1] + A[:-1]) - A[1:])
        return increments

    def event_bin_masses(self, x, resolution: float):
        """Excitation mass in each event bin [t_i, t_i + d) at parameters x, with d = resolution in seconds.

        Returns: (mass from past events only, mass including event i itself).
        """
        eta, beta = self.branching(x), self.beta(x)
        A = exp_states(self.t, beta, self.prints)
        bin_decay = -np.expm1(-beta * resolution)
        return eta * bin_decay * A, eta * bin_decay * (self.event_weight + A)
