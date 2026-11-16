"""SIRS compartment model (SIR with waning removal back to susceptible): specification and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="SIRS",
    parameters=("beta", "gamma", "xi"),
    compartments=("S", "I", "R"),
    transitions=(
        common.Transition("beta", (-1, 1, 0), lambda c, n: c[0] * c[1] / n),
        common.Transition("gamma", (0, -1, 1), lambda c, n: c[1]),
        common.Transition("xi", (1, 0, -1), lambda c, n: c[2]),
    ),
    observed=(1,),
    seed_compartment=1,
    default_seeds=1,
    start_grid={"beta": tuple(np.geomspace(2e-3, 0.08, 8)), "gamma": tuple(np.geomspace(5e-4, 0.03, 6)), "xi": tuple(np.geomspace(2e-4, 0.02, 5))},
)


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the SIRS model: S -> I at rate beta S I / N, I -> R at rate gamma I, R -> S at rate xi R, from one activated node.

    The observed curve is the current activated share, I.
    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimates of beta, gamma and xi (per second) and their bootstrap intervals.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A; Hethcote (2000), SIAM Review.
    """
    return common.fit_events(SPEC, curves, **kwargs)
