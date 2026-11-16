"""Univariate Hawkes process with a power-law kernel, approximated by a fixed sum of exponentials."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, gammaln, logit, psi

from information_diffusion.epidemic.hawkes_exponential import BOUND_ETA, ETA_CEILING, HawkesProblem, _ozaki_states
from information_diffusion.simulation import hawkes as sim

THETA_BOUNDS = (0.25, 5.0)
C_BOUNDS = (2e-3, 20.0)
GRID_STEP = 0.45
GRID_LOW = -17.75
GRID_HIGH = 10.25
DECAYS = np.exp(np.arange(GRID_LOW, GRID_HIGH + 1e-9, GRID_STEP))
STATED_RELATIVE_ERROR = 1e-4


def exponential_weights(theta: float, c: float) -> np.ndarray:
    """Weights a_k so that sum_k a_k exp(-s_k t) approximates theta c^theta / (t + c)^(1+theta).

    The weights discretise (t + c)^-(1+theta) = (1 / Gamma(1+theta)) int_0^infty s^theta exp(-s (t + c)) ds by the
    trapezoid rule on the fixed log-spaced decay grid s_k (step 0.45 in log s, 63 nodes); the relative error of the kernel
    is below 1e-4 for theta in [0.25, 5], c in [0.002, 20] and t in [0.001, 5400] seconds.
    Arguments: theta is the tail exponent; c the offset in seconds.
    Returns: one weight per grid decay.
    Reference: Trefethen and Weideman (2014), SIAM Review (trapezoid rule).
    """
    log_prefactor = np.log(theta) + theta * np.log(c) - gammaln(1.0 + theta) + np.log(GRID_STEP)
    log_terms = (1.0 + theta) * np.log(DECAYS) - DECAYS * c
    return np.exp(log_prefactor + log_terms)


def exact_kernel(theta: float, c: float, lags) -> np.ndarray:
    """Exact normalised power-law kernel theta c^theta / (t + c)^(1+theta) at the given lags (seconds), for exponent theta and offset c."""
    return theta * c ** theta * (np.asarray(lags, dtype=float) + c) ** (-(1.0 + theta))


def approximate_kernel(theta: float, c: float, lags) -> np.ndarray:
    """Sum-of-exponentials approximation of the normalised power-law kernel at the given lags (seconds), for exponent theta and offset c."""
    lags = np.asarray(lags, dtype=float)
    return np.exp(-np.outer(lags, DECAYS)) @ exponential_weights(theta, c)


def exp_sum_states(times, weights=None) -> np.ndarray:
    """Past-sum states on the fixed decay grid, A[i, k] = sum_{j<i} w_j exp(-s_k (t_i - t_j)), by the Ozaki recursion.

    Arguments: times are sorted event times in seconds; weights the event weights (1 each when None).
    Returns: shape (n events, M decays).
    """
    t = np.asarray(times, dtype=float)
    states = np.zeros((DECAYS.size, t.size))
    if t.size >= 2:
        w = np.ones(t.size) if weights is None else np.asarray(weights, dtype=float)
        gaps = np.diff(t)
        rates = np.empty(gaps.size)
        for k, s in enumerate(DECAYS.tolist()):
            np.multiply(-s, gaps, out=rates)
            np.exp(rates, out=rates)
            _ozaki_states(rates, w, states[k])
    return states.T


class PowerLawHawkes(HawkesProblem):
    """Hawkes model with a power-law kernel, phi(t) = eta theta c^theta / (t + c)^(1 + theta), which integrates to eta.

    The excitation uses the sum-of-exponentials approximation, so the likelihood costs O(M n); the compensator is in
    closed form. The kernel block is (logit(eta / 0.9995), log theta, log c). Constructor arguments as for HawkesProblem.
    Reference: Hawkes (1971), Biometrika; Ozaki (1979); Bochud and Challet (2007), Quant. Finance; Hardiman, Bercot and
    Bouchaud (2013), Eur. Phys. J. B.
    """

    n_kernel = 3
    name = "power_law"

    def _prepare(self) -> None:
        self.states = exp_sum_states(self.t, self.prints)

    def _decode(self, k):
        eta = float(ETA_CEILING * expit(k[0]))
        deta = eta * (1.0 - float(expit(k[0])))
        theta = float(np.exp(k[1]))
        c = float(np.exp(k[2]))
        return eta, deta, theta, c

    def kernel_terms(self, k):
        """Excitation at each event, its gradient, the compensator mass and its gradient for the kernel block k."""
        eta, deta, theta, c = self._decode(k)
        weights = exponential_weights(theta, c)
        log_s = np.log(DECAYS)
        dlog_a_dlogtheta = theta * (1.0 / theta + np.log(c) - psi(1.0 + theta) + log_s)
        dlog_a_dlogc = c * (theta / c - DECAYS)
        products = self.states @ np.column_stack((weights, weights * dlog_a_dlogtheta, weights * dlog_a_dlogc))
        current = products[:, 0]
        phi = eta * current
        dphi = np.empty((3, self.n))
        dphi[0] = deta * current
        dphi[1] = eta * products[:, 1]
        dphi[2] = eta * products[:, 2]
        lag = self.lag_floor
        head_base = c / (lag + c)
        head = head_base ** theta
        tail_base = c / (self.remaining + c)
        log_tail_base = np.log(tail_base)
        tail = np.exp(theta * log_tail_base)
        weight = self.event_weight
        comp = eta * float(np.sum(weight * (head - tail)))
        dcomp = np.array([
            deta * float(np.sum(weight * (head - tail))),
            eta * theta * float(np.sum(weight * (head * np.log(head_base) - tail * log_tail_base))),
            eta * theta * float(np.sum(weight * (head * (lag / (lag + c)) - tail * (self.remaining / (self.remaining + c))))),
        ])
        return phi, dphi, comp, dcomp

    def kernel_bounds(self):
        """Bounds of the kernel block: logit(eta / 0.9995) in [-12, 12], theta in [0.25, 5], c in [0.002, 20] seconds."""
        return [(-BOUND_ETA, BOUND_ETA), (np.log(THETA_BOUNDS[0]) + 1e-9, np.log(THETA_BOUNDS[1]) - 1e-9), (np.log(C_BOUNDS[0]) + 1e-9, np.log(C_BOUNDS[1]) - 1e-9)]

    def kernel_starts(self):
        """Starting (eta, theta, c) triples."""
        return [np.array([0.5, 1.5, 0.1]), np.array([0.3, 1.0, 0.5]), np.array([0.7, 2.0, 0.05]), np.array([0.5, 1.0, 0.02])]

    def encode_kernel(self, natural):
        """Transformed kernel block from natural parameters (eta, theta, c)."""
        eta, theta, c = natural
        return np.array([float(logit(eta / ETA_CEILING)), np.log(theta), np.log(c)])

    def branching(self, x) -> float:
        """Branching ratio eta at parameters x."""
        return float(ETA_CEILING * expit(x[self.K]))

    def natural_kernel(self, x):
        """Natural kernel parameters (eta, theta, c) at parameters x."""
        eta, _, theta, c = self._decode(x[self.K :])
        return eta, theta, c

    def simulate(self, x, rng, max_events=None):
        """Simulate one raw event set at parameters x with the exact power-law kernel; max_events caps the number of events."""
        sampler = self.baseline.sampler(x[: self.K])
        immigrants = sampler.sample(rng)
        eta, theta, c = self.natural_kernel(x)
        kernel = sim.PowerLawKernel(c, theta)
        return sim.simulate_cluster(immigrants, kernel, eta, self.window, rng, max_events=max_events)

    def excitation_integral_increments(self, x):
        """Excitation mass between consecutive events at parameters x, through the exponential representation; index 0 covers 0 to t_0."""
        eta, theta, c = self.natural_kernel(x)
        weights = exponential_weights(theta, c) / DECAYS
        increments = np.zeros(self.n)
        increments[1:] = eta * (((self.event_weight[:-1, None] + self.states[:-1]) - self.states[1:]) @ weights)
        return increments

    def event_bin_masses(self, x, resolution: float):
        """Excitation mass in each event bin [t_i, t_i + d) at parameters x, d = resolution in seconds.

        Returns: (mass from past events only, mass including event i itself).
        """
        eta, theta, c = self.natural_kernel(x)
        coefficient = exponential_weights(theta, c) / DECAYS * (-np.expm1(-DECAYS * resolution))
        past = eta * (self.states @ coefficient)
        own = past + eta * float(coefficient.sum()) * self.event_weight
        return past, own
