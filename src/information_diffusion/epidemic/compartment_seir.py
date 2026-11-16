"""SEIR compartment model (SIR with a latent exposed stage): specification and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="SEIR",
    parameters=("beta", "sigma", "gamma"),
    compartments=("S", "E", "I", "R"),
    transitions=(
        common.Transition("beta", (-1, 1, 0, 0), lambda c, n: c[0] * c[2] / n),
        common.Transition("sigma", (0, -1, 1, 0), lambda c, n: c[1]),
        common.Transition("gamma", (0, 0, -1, 1), lambda c, n: c[2]),
    ),
    observed=(2, 3),
    seed_compartment=2,
    default_seeds=1,
    start_grid={"beta": tuple(np.geomspace(3e-3, 0.08, 6)), "sigma": tuple(np.geomspace(1e-3, 0.12, 6)), "gamma": tuple(np.geomspace(5e-4, 0.03, 5))},
)


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the SEIR model: S -> E at rate beta S I / N, E -> I at rate sigma E, I -> R at rate gamma I, from one infectious node.

    The observed curve is the share ever activated, I + R; exposed nodes are not observed.
    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimates of beta, sigma and gamma (per second) and their bootstrap intervals.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A; Hethcote (2000), SIAM Review.
    """
    return common.fit_events(SPEC, curves, **kwargs)
