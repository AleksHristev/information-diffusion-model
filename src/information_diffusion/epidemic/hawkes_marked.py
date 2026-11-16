"""Marked Hawkes process with an exponential kernel, the mark entering the excitation as m^gamma."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from information_diffusion.epidemic.hawkes_exponential import BOUND_ETA, ETA_CEILING, HawkesProblem, exp_states
from information_diffusion.simulation import hawkes as sim


class MarkedHawkes(HawkesProblem):
    """Marked Hawkes model, lambda(t) = mu(t) + eta_0 beta sum_{t_j<t} m_j^gamma exp(-beta (t - t_j)).

    The marks m_j are print counts (at least 1). The branching ratio is eta = eta_0 mean(m_j^gamma), the expected number
    of direct offspring of an event with a mark drawn from the sample. The kernel block is (logit(eta / 0.9995), gamma,
    log beta). Constructor arguments as for HawkesProblem; marks are required and print weighting is not allowed.
    Reference: Hawkes (1971), Biometrika; Hawkes and Oakes (1974), J. Appl. Probab.; Ozaki (1979) for the recursion.
    """

    n_kernel = 3
    name = "marked"

    def _prepare(self) -> None:
        if self.marks is None:
            raise ValueError("marked Hawkes needs marks")
        if self.print_weighted:
            raise ValueError("the marked model carries the mark through gamma, so print weighting does not apply")
        if np.any(self.marks < 1.0):
            raise ValueError("marks are print counts and must be at least 1")
        self.log_marks = np.log(self.marks)

    def kernel_terms(self, k):
        """Excitation at each event, its gradient, the compensator mass and its gradient for the kernel block k."""
        eta = float(ETA_CEILING * expit(k[0]))
        deta = eta * (1.0 - float(expit(k[0])))
        gamma = float(k[1])
        beta = float(np.exp(k[2]))
        weights = np.exp(gamma * self.log_marks)
        mean_weight = float(weights.mean())
        scale = eta / mean_weight
        dmean_dgamma = float(np.mean(weights * self.log_marks))
        dscale_dgamma = -eta * dmean_dgamma / mean_weight ** 2
        A, C = exp_states(self.t, beta, weights, deriv=True)
        dA_dgamma = exp_states(self.t, beta, weights * self.log_marks)
        phi = scale * beta * A
        dphi = np.empty((3, self.n))
        dphi[0] = deta / mean_weight * beta * A
        dphi[1] = scale * beta * dA_dgamma + dscale_dgamma * beta * A
        dphi[2] = scale * (A - beta * C) * beta
        decay = np.exp(-beta * self.remaining)
        bin_decay = np.exp(-beta * self.lag_floor)
        lag_mass = bin_decay - decay
        unit_mass = np.sum(weights * lag_mass)
        comp = scale * float(unit_mass)
        dcomp = np.array([
            deta / mean_weight * float(unit_mass),
            dscale_dgamma * float(unit_mass) + scale * float(np.sum(weights * self.log_marks * lag_mass)),
            scale * float(np.sum(weights * (self.remaining * decay - self.lag_floor * bin_decay))) * beta,
        ])
        return phi, dphi, comp, dcomp

    def kernel_bounds(self):
        """Bounds of the kernel block: logit(eta / 0.9995) in [-12, 12], gamma in [-3, 3], beta in [0.001, 1000] per second."""
        return [(-BOUND_ETA, BOUND_ETA), (-3.0, 3.0), (np.log(1e-3), np.log(1e3))]

    def kernel_starts(self):
        """Starting (eta, gamma, beta) triples."""
        return [np.array([0.5, 0.0, 1.0]), np.array([0.3, 0.3, 1.0]), np.array([0.7, 0.3, 1.0]), np.array([0.5, 0.0, 4.0]), np.array([0.5, 0.0, 0.25])]

    def encode_kernel(self, natural):
        """Transformed kernel block from natural parameters (eta, gamma, beta)."""
        eta, gamma, beta = natural
        return np.array([float(logit(eta / ETA_CEILING)), gamma, np.log(beta)])

    def branching(self, x) -> float:
        """Branching ratio eta = eta_0 mean(m^gamma) at parameters x."""
        return float(ETA_CEILING * expit(x[self.K]))

    def gamma(self, x) -> float:
        """Mark exponent gamma at parameters x."""
        return float(x[self.K + 1])

    def beta(self, x) -> float:
        """Kernel decay rate beta (per second) at parameters x."""
        return float(np.exp(x[self.K + 2]))

    def simulate(self, x, rng, max_events=None):
        """Simulate one event set at parameters x, with marks resampled from the observed marks.

        Arguments: x the parameters; rng the generator; max_events an optional cap on the number of events.
        Returns: (times, marks).
        """
        sampler = self.baseline.sampler(x[: self.K])
        immigrants = sampler.sample(rng)
        law = _EmpiricalMarks(self.marks)
        eta = self.branching(x)
        gamma = self.gamma(x)
        eta_0 = eta / float(np.mean(self.marks ** gamma))
        return sim.simulate_cluster(immigrants, sim.ExponentialKernel(self.beta(x)), eta_0, self.window, rng, marks_law=law, gamma=gamma, max_events=max_events)

    def excitation_integral_increments(self, x):
        """Excitation mass between consecutive events at parameters x, for time-rescaling residuals; index 0 covers 0 to t_0."""
        eta = self.branching(x)
        gamma = self.gamma(x)
        beta = self.beta(x)
        weights = self.marks ** gamma
        scale = eta / float(weights.mean())
        A = exp_states(self.t, beta, weights)
        increments = np.zeros(self.n)
        increments[1:] = scale * ((weights[:-1] + A[:-1]) - A[1:])
        return increments

    def event_bin_masses(self, x, resolution: float):
        """Excitation mass in each event bin [t_i, t_i + d) at parameters x, d = resolution in seconds.

        Returns: (mass from past events only, mass including event i itself).
        """
        eta = self.branching(x)
        gamma = self.gamma(x)
        beta = self.beta(x)
        weights = self.marks ** gamma
        scale = eta / float(weights.mean())
        A = exp_states(self.t, beta, weights)
        bin_decay = -np.expm1(-beta * resolution)
        return scale * bin_decay * A, scale * bin_decay * (weights + A)


class _EmpiricalMarks:

    def __init__(self, marks):
        self.marks = np.asarray(marks, dtype=float)

    def sample(self, size: int, rng: np.random.Generator):
        return self.marks[rng.integers(0, self.marks.size, size)]
