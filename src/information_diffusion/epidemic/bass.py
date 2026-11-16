"""Bass diffusion model: specification, closed-form adoption curve and fit to absorption curves."""

from __future__ import annotations

import numpy as np

from information_diffusion.epidemic import compartment_common as common

SPEC = common.CompartmentSpec(
    name="Bass",
    parameters=("p", "q"),
    compartments=("U", "A"),
    transitions=(
        common.Transition("p", (-1, 1), lambda c, n: c[0]),
        common.Transition("q", (-1, 1), lambda c, n: c[0] * c[1] / n),
    ),
    observed=(1,),
    seed_compartment=1,
    default_seeds=0,
    start_grid={"p": tuple(np.geomspace(1e-4, 0.03, 10)), "q": tuple(np.geomspace(1e-3, 0.1, 10))},
)


def closed_form(p: float, q: float, times: np.ndarray = common.TIME_GRID) -> np.ndarray:
    """Cumulative adopted share of the Bass model, F(t) = (1 - exp(-(p + q) t)) / (1 + (q / p) exp(-(p + q) t)).

    Arguments: p is the innovation rate and q the imitation rate (per second); times are the grid points in seconds.
    Returns: the adopted share at each time.
    Reference: Bass (1969), Management Science.
    """
    decay = np.exp(-(p + q) * np.asarray(times, dtype=float))
    return (1.0 - decay) / (1.0 + (q / p) * decay)


def fit(curves: np.ndarray, **kwargs) -> common.CurveFit:
    """Fit the Bass model to absorption curves; the finite-N chain has each non-adopter adopt at rate p + q A / N.

    Arguments: curves has one row per event and one column per grid point; kwargs are passed to compartment_common.fit_events.
    Returns: a CurveFit with the estimates of p and q and their bootstrap intervals.
    Reference: Niu (2002), Mathematical Problems in Engineering.
    """
    return common.fit_events(SPEC, curves, **kwargs)
