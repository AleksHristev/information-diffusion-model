"""Univariate Hawkes process with a sum-of-exponentials kernel."""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from information_diffusion.epidemic.hawkes_exponential import BOUND_ETA, ETA_CEILING, HawkesProblem, exp_states
from information_diffusion.simulation import hawkes as sim

MERGE_RESOLUTION_SECONDS = 0.001


class SumExponentialsHawkes(HawkesProblem):
    """Hawkes model with lambda(t) = mu(t) + sum_{t_j<t} sum_m eta w_m beta_m exp(-beta_m (t - t_j)), sum_m w_m = 1.

    The branching ratio is eta. Decays are kept ordered beta_1 > beta_2 > ... by log beta_m = log beta_M + sum_{q>=m} exp(z_q),
    which removes label switching; weights use a stick-breaking logit transform. The kernel block is (logit(eta / 0.9995),
    M - 1 stick logits, log beta_M, M - 1 log gaps).
    Constructor arguments as for HawkesProblem, plus components, the number M of exponentials (at least 2, default 2).
    Reference: Hawkes (1971), Biometrika; Ogata (1978); Ozaki (1979).
    """

    name = "sum_exponentials"

    def __init__(self, times, window, baseline, marks=None, resolution=MERGE_RESOLUTION_SECONDS, print_weighted: bool = False, components: int = 2):
        self.components = int(components)
        if self.components < 2:
            raise ValueError("use ExponentialHawkes for a single component")
        self.n_kernel = 1 + (self.components - 1) + self.components
        super().__init__(times, window, baseline, marks=marks, resolution=resolution, print_weighted=print_weighted)

    def _spawn_kwargs(self) -> dict:
        return {"components": self.components}

    def _decode(self, k):
        m = self.components
        eta = float(ETA_CEILING * expit(k[0]))
        deta_dk0 = eta * (1.0 - float(expit(k[0])))
        sticks = expit(k[1 : m])
        weights = np.empty(m)
        dsticks = np.empty((m - 1, m))
        remaining = 1.0
        for i, v in enumerate(sticks):
            weights[i] = remaining * v
            remaining *= 1.0 - v
        weights[-1] = remaining
        for i, v in enumerate(sticks):
            delta = np.zeros(m)
            scale = v * (1.0 - v)
            rem_before = np.prod(1.0 - sticks[:i]) if i > 0 else 1.0
            delta[i] = rem_before * scale
            if i + 1 < m - 1:
                tail = rem_before * (-scale)
                for j in range(i + 1, m - 1):
                    delta[j] = tail * sticks[j]
                    tail *= 1.0 - sticks[j]
                delta[m - 1] = tail
            else:
                delta[m - 1] = -rem_before * scale
            dsticks[i] = delta
        log_beta_last = k[m]
        gaps = np.exp(k[m + 1 :])
        cumulative = np.zeros(m)
        for i in range(m - 2, -1, -1):
            cumulative[i] = cumulative[i + 1] + gaps[i]
        log_betas = log_beta_last + cumulative
        dlogb_dgap = np.zeros((m - 1, m))
        for i in range(m - 1):
            dlogb_dgap[i, : i + 1] = gaps[i]
        return eta, deta_dk0, weights, dsticks, log_betas, dlogb_dgap

    def kernel_terms(self, k):
        """Excitation at each event, its gradient, the compensator mass and its gradient for the kernel block k."""
        m = self.components
        eta, deta_dk0, weights, dsticks, log_betas, dlogb_dgap = self._decode(k)
        betas = np.exp(log_betas)
        q = self.n_kernel
        phi = np.zeros(self.n)
        dphi = np.zeros((q, self.n))
        comp = 0.0
        dcomp = np.zeros(q)
        for i in range(m):
            beta = betas[i]
            A, C = exp_states(self.t, beta, self.prints, deriv=True)
            decay = np.exp(-beta * self.remaining)
            bin_decay = np.exp(-beta * self.lag_floor)
            excitation = beta * A
            eta_m = eta * weights[i]
            phi += eta_m * excitation
            mass = float(np.sum(self.event_weight * (bin_decay - decay)))
            comp += eta_m * mass
            dphi[0] += deta_dk0 * weights[i] * excitation
            dcomp[0] += deta_dk0 * weights[i] * mass
            for j in range(m - 1):
                dphi[1 + j] += eta * dsticks[j, i] * excitation
                dcomp[1 + j] += eta * dsticks[j, i] * mass
            dexcitation_dlogbeta = beta * (A - beta * C)
            dmass_dlogbeta = beta * float(np.sum(self.event_weight * (self.remaining * decay - self.lag_floor * bin_decay)))
            dphi[m] += eta_m * dexcitation_dlogbeta
            dcomp[m] += eta_m * dmass_dlogbeta
            for g in range(m - 1):
                factor = dlogb_dgap[g, i]
                if factor != 0.0:
                    dphi[m + 1 + g] += eta_m * dexcitation_dlogbeta * factor
                    dcomp[m + 1 + g] += eta_m * dmass_dlogbeta * factor
        return phi, dphi, comp, dcomp

    def kernel_bounds(self):
        """Bounds of the kernel block: logit(eta / 0.9995) in [-12, 12], stick logits in [-8, 8], beta_M in [1e-4, 1000] per second, gaps log beta_m - log beta_{m+1} in [0.001, 12]."""
        m = self.components
        limits = [(-BOUND_ETA, BOUND_ETA)]
        limits += [(-8.0, 8.0)] * (m - 1)
        limits += [(np.log(1e-4), np.log(1e3))]
        limits += [(np.log(1e-3), np.log(12.0))] * (m - 1)
        return limits

    def kernel_starts(self):
        """Starting natural kernel vectors (eta, first M - 1 weights, M decays)."""
        m = self.components
        if m != 2:
            start_betas = np.geomspace(5.0, 0.2, m)
            return [np.concatenate(([0.5], np.full(m - 1, 1.0 / m), start_betas))]
        return [np.array([0.5, 0.5, 10.0, 0.1]), np.array([0.3, 0.5, 5.0, 0.5]), np.array([0.7, 0.5, 20.0, 0.05]), np.array([0.5, 0.8, 3.0, 0.3])]

    def encode_kernel(self, natural):
        """Transformed kernel block from a natural vector (eta, first M - 1 weights, M decays in any order)."""
        m = self.components
        eta = float(natural[0])
        weights = np.asarray(natural[1 : m], dtype=float)
        betas = np.asarray(natural[m : 2 * m], dtype=float)
        sticks = []
        remaining = 1.0
        for w in weights:
            sticks.append(float(np.clip(w / remaining, 1e-6, 1 - 1e-6)))
            remaining -= w
        log_beta = np.log(betas)
        order = np.argsort(-betas)
        log_beta = log_beta[order]
        log_last = log_beta[-1]
        gaps = np.maximum(log_beta[:-1] - log_beta[1:], 1e-6)
        return np.concatenate(([float(logit(eta / ETA_CEILING))], logit(np.asarray(sticks)), [log_last], np.log(gaps)))

    def branching(self, x) -> float:
        """Branching ratio eta at parameters x."""
        return float(ETA_CEILING * expit(x[self.K]))

    def natural_kernel(self, x):
        """Natural kernel parameters at x: (eta, weights, betas), betas in decreasing order."""
        eta, _, weights, _, log_betas, _ = self._decode(x[self.K :])
        return eta, weights, np.exp(log_betas)

    def simulate(self, x, rng, max_events=None):
        """Simulate one raw event set at parameters x by the cluster representation; max_events caps the number of events."""
        sampler = self.baseline.sampler(x[: self.K])
        immigrants = sampler.sample(rng)
        eta, weights, betas = self.natural_kernel(x)
        kernel = sim.SumExponentialsKernel(tuple(betas.tolist()), tuple(weights.tolist()))
        return sim.simulate_cluster(immigrants, kernel, eta, self.window, rng, max_events=max_events)

    def excitation_integral_increments(self, x):
        """Excitation mass between consecutive events at parameters x, for time-rescaling residuals; index 0 covers 0 to t_0."""
        eta, weights, betas = self.natural_kernel(x)
        increments = np.zeros(self.n)
        for w, beta in zip(weights, betas):
            A = exp_states(self.t, beta, self.prints)
            increments[1:] += eta * w * ((self.event_weight[:-1] + A[:-1]) - A[1:])
        return increments

    def event_bin_masses(self, x, resolution: float):
        """Excitation mass in each event bin [t_i, t_i + d) at parameters x, d = resolution in seconds.

        Returns: (mass from past events only, mass including event i itself).
        """
        eta, weights, betas = self.natural_kernel(x)
        past = np.zeros(self.n)
        own = np.zeros(self.n)
        for w, beta in zip(weights, betas):
            A = exp_states(self.t, beta, self.prints)
            bin_decay = -np.expm1(-beta * resolution)
            past += eta * w * bin_decay * A
            own += eta * w * bin_decay * (self.event_weight + A)
        return past, own
