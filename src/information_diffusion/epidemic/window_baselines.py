"""Spline baselines for windows with no release jump or with two shocks, and the baseline for each window shape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from information_diffusion.epidemic.spline_baseline import PENALTY_DEFAULT, SplineBaseline


@dataclass(frozen=True)
class WindowShape:
    """Shape of an event window, all times in seconds.

    Fields: seconds, the window length; release, the first shock time from the window start (None for no shock);
    second_shock, an optional second shock time from the window start.
    """

    seconds: float
    release: float | None
    second_shock: float | None = None


SHAPES = {"0800-0930": WindowShape(5400.0, 1800.0), "0930-1000": WindowShape(1800.0, None), "1330-1530": WindowShape(7200.0, 1800.0, 3600.0)}


def _open_knots(n_events: int) -> np.ndarray:
    if n_events >= 10_000:
        return np.array([15.0, 30.0, 60.0, 120.0, 240.0, 480.0, 900.0, 1350.0])
    if n_events >= 1_500:
        return np.array([30.0, 90.0, 300.0, 900.0])
    return np.array([120.0, 900.0])


def _two_shock_knots(release: float, second: float, window: float, n_events: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_events >= 10_000:
        before, between, after = np.array([-1200.0, -600.0, -180.0, -30.0]), np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1400.0]), np.array([20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
    elif n_events >= 1_500:
        before, between, after = np.array([-900.0, -120.0]), np.array([30.0, 80.0, 200.0, 500.0, 1200.0]), np.array([30.0, 80.0, 200.0, 500.0, 1200.0, 2400.0])
    else:
        before, between, after = np.array([-900.0]), np.array([30.0, 120.0, 480.0]), np.array([30.0, 120.0, 480.0, 1800.0])
    return release + before[before > -release], release + between[release + between < second], second + after[second + after < window]


class OpenWindowBaseline(SplineBaseline):
    """Log-linear cubic B-spline baseline on [0, window] with no jump, knots dense near the window start.

    Constructor arguments: window in seconds; n_events chooses the default knots; internal_knots replaces them;
    penalty is the second-difference penalty weight (default 0.1).
    Reference: De Boor (2001), A Practical Guide to Splines; Eilers and Marx (1996), Stat. Sci.
    """

    def __init__(self, window: float, n_events: int | None = None, internal_knots=None, penalty: float = PENALTY_DEFAULT):
        self.window = float(window)
        self.release = None
        distinct = np.asarray(_open_knots(0 if n_events is None else n_events) if internal_knots is None else internal_knots, dtype=float)
        if not (np.all(np.diff(distinct) > 0) and distinct[0] > 0 and distinct[-1] < window):
            raise ValueError("interior knots must increase inside the window")
        self.interior_knots = distinct
        self.knots = np.concatenate(([0.0] * 4, distinct, [self.window] * 4))
        self.n_coef = len(self.knots) - 4
        self.penalty_weight = float(penalty)
        self.n_left = self.n_coef
        self._build_quadrature()
        self._build_penalty()


class TwoShockBaseline(SplineBaseline):
    """Log-linear cubic B-spline baseline on [0, window] with a free jump at each of two shocks.

    Each shock is an interior knot of multiplicity 4, and the penalty applies to each of the three pieces separately.
    Constructor arguments: window, release and second_shock in seconds (0 < release < second_shock < window); n_events
    chooses the default knots; penalty is the second-difference penalty weight (default 0.1).
    Reference: De Boor (2001), A Practical Guide to Splines; Eilers and Marx (1996), Stat. Sci.
    """

    def __init__(self, window: float, release: float, second_shock: float, n_events: int | None = None, penalty: float = PENALTY_DEFAULT):
        if not 0 < release < second_shock < window:
            raise ValueError("the shocks must lie strictly inside the window, in order")
        self.window, self.release, self.second_shock = float(window), float(release), float(second_shock)
        pre, middle, post = _two_shock_knots(self.release, self.second_shock, self.window, 0 if n_events is None else n_events)
        self.interior_knots = np.concatenate((pre, [self.release], middle, [self.second_shock], post))
        self.knots = np.concatenate(([0.0] * 4, pre, [self.release] * 4, middle, [self.second_shock] * 4, post, [self.window] * 4))
        self.n_coef = len(self.knots) - 4
        self.penalty_weight = float(penalty)
        self.pieces = (len(pre) + 4, len(middle) + 4, len(post) + 4)
        self.n_left = self.pieces[0]
        self._build_quadrature()
        self._build_penalty()

    def _build_penalty(self) -> None:
        rows = []
        first = 0
        for size in self.pieces:
            for row in range(size - 2):
                difference = np.zeros(self.n_coef)
                difference[first + row : first + row + 3] = (1.0, -2.0, 1.0)
                rows.append(difference)
            first += size
        penalty = np.array(rows)
        self.penalty_matrix = self.penalty_weight * penalty.T @ penalty


def baseline_for(label: str, n_events: int) -> SplineBaseline:
    """Baseline for a window label in SHAPES, with knots chosen by the number of events.

    The labels are New York clock times: "0800-0930" is a 5,400-second window with the release 1,800 seconds in;
    "0930-1000" a 1,800-second window with no release; "1330-1530" a 7,200-second window with shocks 1,800 and 3,600
    seconds in.
    Arguments: label is the key in SHAPES; n_events the number of events.
    Returns: a SplineBaseline, OpenWindowBaseline or TwoShockBaseline.
    """
    shape = SHAPES[label]
    if shape.second_shock is not None:
        return TwoShockBaseline(shape.seconds, shape.release, shape.second_shock, n_events=n_events)
    if shape.release is None:
        return OpenWindowBaseline(shape.seconds, n_events=n_events)
    return SplineBaseline(shape.seconds, shape.release, n_events=n_events)
