"""SIR (susceptible-infected-removed) compartment model: specification and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="SIR",
    parameters=("beta", "gamma"),
    compartments=("S", "I", "R"),
    transitions=(
        common.Transition("beta", (-1, 1, 0), lambda c, n: c[0] * c[1] / n),
        common.Transition("gamma", (0, -1, 1), lambda c, n: c[1]),
    ),
    observed=(1, 2),
    seed_compartment=1,
    default_seeds=1,
    start_grid={"beta": tuple(np.geomspace(1e-3, 0.1, 12)), "gamma": tuple(np.geomspace(2e-4, 0.05, 10))},
)


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the SIR model: S -> I at rate beta S I / N, I -> R at rate gamma I, from one activated node.

    The observed curve is the share ever activated, I + R, so it can be built from first activation times.
    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimates of beta and gamma (per second) and their bootstrap intervals.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A.
    """
    return common.fit_events(SPEC, curves, **kwargs)
