"""SI (susceptible-infected) compartment model: specification, logistic mean-field curve and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="SI",
    parameters=("beta",),
    compartments=("S", "I"),
    transitions=(common.Transition("beta", (-1, 1), lambda c, n: c[0] * c[1] / n),),
    observed=(1,),
    seed_compartment=1,
    default_seeds=1,
    start_grid={"beta": tuple(np.geomspace(5e-4, 0.1, 16))},
)


def logistic(beta: float, times: np.ndarray = common.TIME_GRID, *, initial_share: float = 1 / common.N_NODES) -> np.ndarray:
    """Mean-field activated share of the SI model, I(t) = I0 e^(beta t) / (1 + I0 (e^(beta t) - 1)).

    Arguments: beta is the infection rate (per second); times the grid in seconds; initial_share is I0 (default 1/30).
    Returns: the activated share at each time.
    Reference: Kermack and McKendrick (1927), Proc. R. Soc. A.
    """
    grow = np.exp(beta * np.asarray(times, dtype=float))
    return initial_share * grow / (1.0 + initial_share * (grow - 1.0))


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the SI model: S -> I at rate beta S I / N, from one activated node; the observed curve is I.

    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimate of beta (per second) and its bootstrap interval.
    """
    return common.fit_events(SPEC, curves, **kwargs)
