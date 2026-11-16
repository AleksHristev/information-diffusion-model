"""Exponential Hawkes models on an event clock of bin width d: left-edge and exact bin-mass likelihoods."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import event_merging
from information_diffusion.epidemic.hawkes_exponential import BOUND_ETA, ExponentialHawkes

DECAY_FLOOR = 1e-3


class LeftEdgeCoarseClockHawkes(ExponentialHawkes):
    """Exponential Hawkes on a clock of bin width d, with the left-edge bin intensity H_a = d lambda(u_a) and beta capped at 1 / d.

    Events sit at the start of each occupied bin. The left-edge form overstates the in-bin excitation by the factor
    x / (1 - exp(-x)), x = beta d, so at coarse clocks its score at eta = 0 is biased upwards.
    Constructor arguments as for HawkesProblem; resolution is the bin width d in seconds and must be positive.
    Reference: Hawkes (1971), Biometrika; Haslinger, Pipa and Brown (2010), Neural Computation.
    """

    name = "exponential_coarse_clock_left_edge"

    def _prepare(self) -> None:
        if self.resolution is None or self.resolution <= 0.0:
            raise ValueError("the coarse clock needs a positive bin width")

    def kernel_bounds(self):
        """Bounds of the kernel block: logit(eta / 0.9995) in [-12, 12], beta in [0.001, 1 / d] per second."""
        return [(-BOUND_ETA, BOUND_ETA), (np.log(DECAY_FLOOR), np.log(1.0 / self.resolution))]

    def kernel_starts(self):
        """Starting (eta, beta) pairs, with beta scaled to the bin width."""
        scale = 0.2 / self.resolution
        return [np.array([0.3, scale]), np.array([0.7, scale]), np.array([0.5, 4.0 * scale]), np.array([0.5, 0.2 * scale])]

    def simulate(self, x, rng, max_events=None):
        """Simulate one event set at parameters x and return it already merged onto the clock of width d."""
        return event_merging.merge_events(super().simulate(x, rng, max_events=max_events), resolution=self.resolution)[0]


class CoarseClockHawkes(LeftEdgeCoarseClockHawkes):
    """Exponential Hawkes on a clock of bin width d, with the occupied-bin term built from the exact kernel mass of the bin.

    H_a = d mu(u_a) + eta (1 - exp(-beta d)) sum_{u_j < u_a} exp(-beta (u_a - u_j)), which matches the compensator, so the
    score at eta = 0 has mean zero at every beta d. The fitted eta describes the occupied-bin process, not the original events.
    Constructor arguments as for LeftEdgeCoarseClockHawkes.
    Reference: Hawkes (1971), Biometrika; Haslinger, Pipa and Brown (2010), Neural Computation.
    """

    name = "exponential_coarse_clock"

    def kernel_terms(self, k):
        """Kernel terms of the exponential model with phi scaled by f(x) = (1 - exp(-x)) / x, x = beta d, and its gradient adjusted."""
        phi, dphi, comp, dcomp = super().kernel_terms(k)
        x = float(np.exp(k[1])) * self.resolution
        factor = -np.expm1(-x) / x
        return phi * factor, np.vstack((dphi[0] * factor, dphi[1] * factor + phi * (np.exp(-x) - factor))), comp, dcomp
