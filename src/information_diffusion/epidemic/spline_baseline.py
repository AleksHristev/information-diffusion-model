"""Hawkes baseline rates: constant, or a log-linear cubic B-spline with a jump at the release time, with samplers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline

GAUSS_NODES = 12
PENALTY_DEFAULT = 0.1


@dataclass(frozen=True)
class ConstantBaseline:
    """Constant baseline mu(t) = exp(b_0) on [0, window].

    Fields: window, the window length in seconds; n_coef, the number of coefficients (1); constant (True).
    """

    window: float
    n_coef: int = 1
    constant: bool = True

    def design(self, times) -> None:
        """No design matrix for a constant baseline; returns None."""
        return None

    def integral_and_grad(self, b):
        """Integral of mu over the window and its gradient in b_0. Returns (value, gradient)."""
        value = float(np.exp(b[0]) * self.window)
        return value, np.array([value])

    def penalty_and_grad(self, b):
        """Roughness penalty and its gradient, both zero for a constant baseline. Returns (0.0, zeros)."""
        return 0.0, np.zeros(1)

    def mean(self, b) -> float:
        """Mean rate exp(b_0) (events per second)."""
        return float(np.exp(b[0]))

    def mean_log_gradient(self, b):
        """Gradient of the log mean rate in b, a vector of ones."""
        return np.ones(1)

    def initial(self, times, rate) -> np.ndarray:
        """Starting coefficient log(rate) for a baseline rate in events per second; times are not used."""
        return np.array([np.log(max(rate, 1e-10))])

    def cumulative(self, b, times) -> np.ndarray:
        """Integrated rate from 0 to each time (seconds), exp(b_0) t."""
        return np.exp(b[0]) * np.asarray(times, dtype=float)

    def sampler(self, b) -> "ConstantSampler":
        """Sampler of the Poisson process with this baseline at coefficients b."""
        return ConstantSampler(float(np.exp(b[0])), self.window)

    def rate(self, b, times):
        """Rate exp(b_0) at each time."""
        return np.exp(b[0]) * np.ones_like(np.asarray(times, dtype=float))


@dataclass(frozen=True)
class ConstantSampler:
    """Sampler of a homogeneous Poisson process. Fields: rate (events per second) and window (seconds)."""

    rate: float
    window: float

    def sample(self, rng: np.random.Generator):
        """Draw one set of event times on [0, window] (unsorted) with the generator rng."""
        return rng.uniform(0.0, self.window, rng.poisson(self.rate * self.window))

    def expected_count(self) -> float:
        """Expected number of events, rate times window."""
        return self.rate * self.window


def default_internal_knots(window: float, release: float, n_events: int) -> np.ndarray:
    """Default distinct interior knots, denser after the release and fewer when events are scarce.

    Three knot sets apply: for at least 10,000 events, for at least 1,500, and below.
    Arguments: window is the window length and release the release time, both in seconds from the window start;
    n_events is the number of events.
    Returns: the knots in seconds, including the release.
    """
    if window <= release or release <= 0:
        raise ValueError("release must lie strictly inside the window")
    if n_events >= 10_000:
        before = np.array([-1200.0, -600.0, -180.0, -30.0])
        after = np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
    elif n_events >= 1_500:
        before = np.array([-900.0, -120.0])
        after = np.array([30.0, 80.0, 200.0, 500.0, 1200.0])
    else:
        before = np.array([-900.0])
        after = np.array([30.0, 120.0, 480.0, 1800.0])
    pre = release + before[before > -release]
    post = release + after[after < window - release]
    return np.concatenate((pre, [release], post))


class SplineBaseline:
    """Log-linear cubic B-spline baseline, log mu(t) = sum_k b_k B_k(t) on [0, window], with a free jump at the release time.

    The release is an interior knot of multiplicity 4, so mu may jump there. Integrals use 12-node Gauss-Legendre
    quadrature on each span. A second-difference penalty 0.5 pen ||D_2 b||^2 on each side of the jump keeps coefficients
    bounded on spans with no events.
    Constructor arguments: window and release in seconds; n_events chooses the default knots and is required unless
    internal_knots is given; internal_knots replaces them (increasing, inside the window, including the release);
    penalty is pen (default 0.1).
    Reference: De Boor (2001), A Practical Guide to Splines; Eilers and Marx (1996), Stat. Sci.
    """

    n_coef: int
    constant = False

    def __init__(self, window: float, release: float, n_events: int | None = None, internal_knots=None, penalty: float = PENALTY_DEFAULT):
        self.window = float(window)
        self.release = float(release)
        distinct = np.asarray(default_internal_knots(window, release, n_events) if internal_knots is None else internal_knots, dtype=float)
        if not (np.all(np.diff(distinct) > 0) and distinct[0] > 0 and distinct[-1] < window and np.any(distinct == release)):
            raise ValueError("interior knots must increase inside the window and include the release")
        self.interior_knots = distinct
        pre = distinct[distinct < release]
        post = distinct[distinct > release]
        self.knots = np.concatenate(([0.0] * 4, pre, [release] * 4, post, [self.window] * 4))
        self.n_coef = len(self.knots) - 4
        self.penalty_weight = float(penalty)
        self.n_left = len(pre) + 4
        self._build_quadrature()
        self._build_penalty()

    def _build_quadrature(self) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(GAUSS_NODES)
        edges = np.unique(self.knots)
        spans = np.diff(edges)
        points = np.concatenate([edges[i] + 0.5 * spans[i] * (nodes + 1.0) for i in range(len(spans))])
        self.q_weights = np.concatenate([0.5 * spans[i] * weights for i in range(len(spans))])
        self.q_design = self.design(points).toarray()
        self.q_points = points

    def _build_penalty(self) -> None:
        def second_difference(size: int) -> np.ndarray:
            if size < 3:
                return np.zeros((0, size))
            matrix = np.zeros((size - 2, size))
            for row in range(size - 2):
                matrix[row, row:row + 3] = (1.0, -2.0, 1.0)
            return matrix
        left, right = second_difference(self.n_left), second_difference(self.n_coef - self.n_left)
        penalty = np.zeros((left.shape[0] + right.shape[0], self.n_coef))
        penalty[:left.shape[0], :self.n_left] = left
        penalty[left.shape[0]:, self.n_left:] = right
        self.penalty_matrix = self.penalty_weight * penalty.T @ penalty

    def design(self, times) -> sparse.csr_array:
        """Sparse B-spline design matrix at the given times (seconds, clipped to the window)."""
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.window)
        return BSpline.design_matrix(times, self.knots, 3, extrapolate=False)

    def log_rate(self, b, times) -> np.ndarray:
        """Log rate sum_k b_k B_k(t) at the given times."""
        return self.design(times) @ np.asarray(b, dtype=float)

    def rate(self, b, times) -> np.ndarray:
        """Rate mu(t) at the given times (events per second)."""
        return np.exp(self.log_rate(b, times))

    def integral_and_grad(self, b):
        """Integral of mu over the window by quadrature, and its gradient in b. Returns (value, gradient)."""
        values = np.exp(self.q_design @ b) * self.q_weights
        return float(values.sum()), self.q_design.T @ values

    def penalty_and_grad(self, b):
        """Second-difference penalty 0.5 b' P b and its gradient P b. Returns (value, gradient)."""
        gradient = self.penalty_matrix @ b
        return 0.5 * float(b @ gradient), gradient

    def mean(self, b) -> float:
        """Mean rate over the window (events per second)."""
        return float(self.integral_and_grad(b)[0] / self.window)

    def mean_log_gradient(self, b) -> np.ndarray:
        """Gradient of the log mean rate, d log(mean mu) / d b_k."""
        integral, gradient = self.integral_and_grad(b)
        return gradient / integral

    def initial(self, times, rate: float) -> np.ndarray:
        """Starting coefficients: the log of the event rate per span, scaled to total rate times window and read at the Greville points.

        Arguments: times are the event times in seconds; rate is the target baseline rate (events per second).
        Returns: one coefficient per basis function.
        """
        times = np.asarray(times, dtype=float)
        edges = np.unique(self.knots)
        counts, _ = np.histogram(times, bins=edges)
        widths = np.diff(edges)
        centres = 0.5 * (edges[:-1] + edges[1:])
        total = counts.sum()
        raw = (counts + 0.5) / widths * (rate * self.window) / max(total + 0.5 * len(counts), 1.0)
        greville = np.array([self.knots[k + 1:k + 4].mean() for k in range(self.n_coef)])
        coef = np.log(np.interp(greville, centres, np.maximum(raw, 1e-8)))
        return coef

    def cumulative(self, b, times) -> np.ndarray:
        """Integrated rate int_0^t mu(s) ds at each time (seconds), by Gauss-Legendre quadrature on the spans."""
        times = np.clip(np.asarray(times, dtype=float), 0.0, self.window)
        edges = np.unique(self.knots)
        full = np.zeros(len(edges))
        nodes, weights = np.polynomial.legendre.leggauss(GAUSS_NODES)
        for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            points = left + 0.5 * (right - left) * (nodes + 1.0)
            full[i + 1] = full[i] + 0.5 * (right - left) * float(np.sum(weights * np.exp(self.log_rate(b, points))))
        span = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
        left = edges[span]
        half = 0.5 * (times - left)
        points = np.repeat(left, GAUSS_NODES) + np.tile(0.5 * (nodes + 1.0), times.size) * np.repeat(times - left, GAUSS_NODES)
        values = np.exp(self.design(points) @ b).reshape(times.size, GAUSS_NODES)
        return full[span] + half * (values @ weights)

    def sampler(self, b) -> "SplineSampler":
        """Thinning sampler of the Poisson process with this baseline at coefficients b."""
        return SplineSampler(self, np.asarray(b, dtype=float))


class SplineSampler:
    """Exact thinning sampler for a log-spline rate, with the bound exp(max active b_k) on each span between knots.

    Constructor arguments: baseline is the SplineBaseline; b its coefficients.
    Reference: Lewis and Shedler (1979), Naval Research Logistics Quarterly.
    """

    def __init__(self, baseline: SplineBaseline, b: np.ndarray):
        self.baseline = baseline
        self.b = b
        self.edges = np.unique(baseline.knots)
        widths = np.diff(self.edges)
        upper = np.empty(len(widths))
        knots = baseline.knots
        for i in range(len(widths)):
            left = np.searchsorted(knots, self.edges[i], side="right") - 1
            upper[i] = b[max(left - 3, 0):min(left + 1, baseline.n_coef)].max()
        self.bound = np.exp(upper)
        self.mass = widths * self.bound
        self.widths = widths

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one set of event times on the window with the generator rng (ordered by span)."""
        counts = rng.poisson(self.mass)
        total = int(counts.sum())
        if total == 0:
            return np.empty(0)
        span = np.repeat(np.arange(len(counts)), counts)
        proposals = self.edges[span] + self.widths[span] * rng.random(total)
        accept = rng.random(total) * self.bound[span] < self.baseline.rate(self.b, proposals)
        return proposals[accept]

    def expected_count(self) -> float:
        """Expected number of events, the integral of the rate over the window."""
        return float(self.baseline.integral_and_grad(self.b)[0])
