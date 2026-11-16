"""Hawkes process simulators by the cluster representation, delay kernels, baseline rates, marks and expected counts."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


class ExplosionError(RuntimeError):
    """Raised when a simulation produces more events than its cap allows, as near the critical branching ratio."""




@dataclass(frozen=True)
class ExponentialKernel:
    """Exponential delay law with density beta exp(-beta t). Field: beta, the decay rate (per second)."""

    beta: float

    def cdf(self, lag):
        """Delay distribution function 1 - exp(-beta t) at the given lags (seconds)."""
        return 1.0 - np.exp(-self.beta * np.asarray(lag, dtype=float))

    def sample(self, size: int, rng: np.random.Generator):
        """Draw size delays (seconds) with the generator rng."""
        return rng.exponential(1.0 / self.beta, size)


@dataclass(frozen=True)
class SumExponentialsKernel:
    """Mixture of exponential delay laws, density sum_m w_m beta_m exp(-beta_m t). Fields: betas, the decay rates; weights, summing to one."""

    betas: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self):
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("component weights must sum to one")

    def cdf(self, lag):
        """Delay distribution function sum_m w_m (1 - exp(-beta_m t)) at the given lags (seconds)."""
        lag = np.asarray(lag, dtype=float)
        return sum(w * (1.0 - np.exp(-b * lag)) for b, w in zip(self.betas, self.weights))

    def sample(self, size: int, rng: np.random.Generator):
        """Draw size delays (seconds) with the generator rng."""
        component = rng.choice(len(self.betas), size=size, p=np.asarray(self.weights))
        return rng.exponential(1.0, size) / np.asarray(self.betas)[component]


@dataclass(frozen=True)
class PowerLawKernel:
    """Power-law delay law with density theta c^theta / (t + c)^(1 + theta). Fields: c, the offset in seconds; theta, the tail exponent."""

    c: float
    theta: float

    def cdf(self, lag):
        """Delay distribution function 1 - (c / (t + c))^theta at the given lags (seconds)."""
        return 1.0 - (self.c / (np.asarray(lag, dtype=float) + self.c)) ** self.theta

    def sample(self, size: int, rng: np.random.Generator):
        """Draw size delays (seconds) by inversion with the generator rng."""
        return self.c * (rng.random(size) ** (-1.0 / self.theta) - 1.0)




@dataclass(frozen=True)
class FlatRate:
    """Constant immigrant rate on [0, window]. Fields: rate (events per second) and window (seconds)."""

    rate: float
    window: float

    def sample(self, rng: np.random.Generator):
        """Draw one set of immigrant times (unsorted) with the generator rng."""
        return rng.uniform(0.0, self.window, rng.poisson(self.rate * self.window))

    def cumulative(self, t):
        """Integrated rate from 0 to each time t (seconds)."""
        return self.rate * np.asarray(t, dtype=float)

    def mean(self) -> float:
        """Mean rate over the window (events per second)."""
        return self.rate


@dataclass(frozen=True)
class SpikingRate:
    """Immigrant rate with a decaying spike at the release, mu(t) = mu0 (1 + amplitude exp(-(t - release) / decay) 1{t >= release}).

    Fields: mu0, the base rate (events per second); window and release in seconds; amplitude (default 30); decay, the
    spike time constant in seconds (default 60).
    """

    mu0: float
    window: float
    release: float
    amplitude: float = 30.0
    decay: float = 60.0

    def sample(self, rng: np.random.Generator):
        """Draw one set of immigrant times (unsorted) with the generator rng."""
        flat = rng.uniform(0.0, self.window, rng.poisson(self.mu0 * self.window))
        horizon = self.window - self.release
        mass = self.mu0 * self.amplitude * self.decay * (1.0 - np.exp(-horizon / self.decay))
        count = rng.poisson(mass)
        if count == 0:
            return flat
        tail = 1.0 - np.exp(-horizon / self.decay)
        spike = self.release - self.decay * np.log1p(-rng.random(count) * tail)
        return np.concatenate((flat, spike))

    def cumulative(self, t):
        """Integrated rate from 0 to each time t (seconds)."""
        t = np.asarray(t, dtype=float)
        over = np.clip(t - self.release, 0.0, None)
        return self.mu0 * (t + self.amplitude * self.decay * (1.0 - np.exp(-over / self.decay)))

    def mean(self) -> float:
        """Mean rate over the window (events per second)."""
        return float(self.cumulative(self.window) / self.window)




@dataclass(frozen=True)
class GeometricMarks:
    """Geometric mark law for print counts, P(m = k) = p (1 - p)^(k - 1), k >= 1, mean 1 / p. Field: p (default 0.5)."""

    p: float = 0.5

    def sample(self, size: int, rng: np.random.Generator):
        """Draw size marks with the generator rng."""
        return rng.geometric(self.p, size).astype(float)

    def mean_power(self, gamma: float) -> float:
        """E[m^gamma] under the geometric law, by the series truncated at a tail mass of 1e-16."""
        k = np.arange(1, int(np.ceil(np.log(1e-16) / np.log(1 - self.p))) + 2, dtype=float)
        return float(np.sum(self.p * (1 - self.p) ** (k - 1) * k ** gamma))




def simulate_cluster(immigrants, kernel, branching, window: float, rng: np.random.Generator, *, marks_law=None, gamma: float = 0.0, max_events: int | None = None):
    """Simulate one Hawkes realisation on [0, window] by the cluster representation, generation by generation.

    Each event has Poisson(branching) children, or Poisson(branching m^gamma) for an event with mark m, at delays drawn
    from the kernel; generations continue until none fall inside the window.
    Arguments: immigrants are the immigrant times; kernel a delay law with sample(); branching the branching ratio (eta,
    or eta_0 for marks); window in seconds; rng the generator; marks_law an optional mark law with sample(); gamma the
    mark exponent; max_events a cap that raises ExplosionError (default 100,000,000).
    Returns: sorted event times, or (times, marks) when marks_law is given.
    Reference: Hawkes and Oakes (1974), J. Appl. Probab.
    """
    current = np.asarray(immigrants, dtype=float)
    current_marks = None if marks_law is None else marks_law.sample(current.size, rng)
    all_times, all_marks = [current], [current_marks]
    total = current.size
    guard = max_events if max_events is not None else 100_000_000
    while current.size:
        means = branching if marks_law is None else branching * current_marks ** gamma
        offspring = rng.poisson(means, size=current.size) if marks_law is not None else rng.poisson(branching, size=current.size)
        parents = np.repeat(current, offspring)
        if parents.size == 0:
            break
        children = parents + kernel.sample(parents.size, rng)
        current = children[children < window]
        total += current.size
        if total > guard:
            raise ExplosionError(f"more than {guard} events simulated")
        all_times.append(current)
        if marks_law is not None:
            current_marks = marks_law.sample(current.size, rng)
            all_marks.append(current_marks)
    times = np.concatenate(all_times)
    if marks_law is None:
        return np.sort(times)
    order = np.argsort(times, kind="stable")
    return times[order], np.concatenate(all_marks)[order]


def simulate_multivariate(rates, branching_matrix, beta: float, window: float, rng: np.random.Generator, *, max_events: int | None = None):
    """Simulate a D-node Hawkes process with a shared exponential kernel by the cluster representation.

    Arguments: rates are the constant baseline rates (events per second); branching_matrix[i, j] is the expected number
    of children in i of an event in j; beta the decay (per second); window in seconds; rng the generator; max_events a
    cap that raises ExplosionError (default 200,000,000).
    Returns: a list of sorted event times, one array per node.
    """
    rates = np.asarray(rates, dtype=float)
    gamma = np.asarray(branching_matrix, dtype=float)
    d = rates.size
    guard = max_events if max_events is not None else 200_000_000
    times = [rng.uniform(0.0, window, rng.poisson(rates[i] * window)) for i in range(d)]
    current_times = np.concatenate(times)
    current_nodes = np.concatenate([np.full(len(t), i, dtype=np.int64) for i, t in enumerate(times)])
    collected_times, collected_nodes = [current_times], [current_nodes]
    total = current_times.size
    totals = gamma.sum(axis=0)
    probabilities = [gamma[:, j] / totals[j] if totals[j] > 0 else None for j in range(d)]
    while current_times.size:
        next_times, next_nodes = [], []
        for j in range(d):
            parents = current_times[current_nodes == j]
            if parents.size == 0 or probabilities[j] is None:
                continue
            counts = rng.poisson(totals[j], parents.size)
            n_children = int(counts.sum())
            if n_children == 0:
                continue
            child_times = np.repeat(parents, counts) + rng.exponential(1.0 / beta, n_children)
            child_nodes = rng.choice(d, size=n_children, p=probabilities[j])
            keep = child_times < window
            next_times.append(child_times[keep])
            next_nodes.append(child_nodes[keep])
        if not next_times:
            break
        current_times = np.concatenate(next_times)
        current_nodes = np.concatenate(next_nodes)
        total += current_times.size
        if total > guard:
            raise ExplosionError(f"more than {guard} events simulated")
        collected_times.append(current_times)
        collected_nodes.append(current_nodes)
    all_times = np.concatenate(collected_times)
    all_nodes = np.concatenate(collected_nodes)
    return [np.sort(all_times[all_nodes == i]) for i in range(d)]




def renewal_progeny(kernel, branching: float, window: float, step: float = 0.01) -> np.ndarray:
    """Renewal function on a lag grid: m_k = delta_k0 + sum_{j<=k} g_{k-j} m_j, with g the kernel mass per cell times branching.

    Arguments: kernel a delay law with cdf(); branching the branching ratio; window and step (default 0.01) in seconds.
    Returns: the expected number of descendants (including the immigrant itself) in each of the ceil(window / step) cells.
    """
    cells = int(np.ceil(window / step))
    edges = np.arange(cells + 1) * step
    masses = np.diff(kernel.cdf(edges)) * branching
    size = int(2 ** np.ceil(np.log2(2 * cells)))
    transform = np.fft.rfft(masses, size)
    progeny = np.fft.irfft(1.0 / (1.0 - transform), size)[:cells]
    return progeny


def expected_count(baseline, kernel, branching: float, window: float, step: float = 0.01) -> float:
    """Expected event count E[N(window)] = integral of mu(s) R(window - s) ds, R the cumulative renewal function.

    Arguments: baseline has cumulative(t); kernel a delay law with cdf(); branching the branching ratio; window and step in seconds.
    Returns: the expected count.
    """
    progeny = renewal_progeny(kernel, branching, window, step)
    cumulative_progeny = np.cumsum(progeny)
    cells = len(progeny)
    edges = np.arange(cells + 1) * step
    baseline_mass = np.diff(baseline.cumulative(edges))
    return float(np.sum(baseline_mass * cumulative_progeny[::-1]))


def calibrate_rate(shape_baseline, kernel, branching: float, window: float, target_count: float, step: float = 0.01) -> float:
    """Scale factor for a unit-level baseline so that the expected count equals target_count; other arguments as in expected_count."""
    return target_count / expected_count(shape_baseline, kernel, branching, window, step)


@functools.lru_cache(maxsize=256)
def scenario_baseline(kind: str, kernel, branching: float, window: float, release: float, target_count: float):
    """True immigrant baseline for a simulation: flat or spiking, scaled so that the expected count equals target_count.

    The result is cached, since the renewal calculation is costly and the kernels and baselines are immutable.
    Arguments: kind is "flat" or "spiking"; kernel the delay law; branching the branching ratio; window and release in
    seconds; target_count the expected number of events.
    Returns: a FlatRate or SpikingRate.
    """
    if kind == "flat":
        shape = FlatRate(1.0, window)
        rate = calibrate_rate(shape, kernel, branching, window, target_count)
        return FlatRate(rate, window)
    if kind == "spiking":
        shape = SpikingRate(1.0, window, release)
        rate = calibrate_rate(shape, kernel, branching, window, target_count)
        return SpikingRate(rate, window, release)
    raise ValueError(f"unknown baseline kind {kind}")
