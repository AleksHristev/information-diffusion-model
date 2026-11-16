"""Checks of the spline Poisson model on the Hawkes spline baseline: step integrals, derivatives, penalty, fit and simulation."""

import numpy as np
import pytest

from information_diffusion.epidemic.spline_baseline import SplineBaseline
from information_diffusion.predictive import spline_poisson as sp

WINDOW, RELEASE, STEP = 5400.0, 1800.0, 30.0
GAMMA = np.array([0.1, 0.2, 0.2, 0.3, 1.5, 1.3, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3])


@pytest.fixture(scope="module")
def baseline() -> SplineBaseline:
    return SplineBaseline(WINDOW, RELEASE, n_events=500)


def central_difference(function, point: np.ndarray, step: float = 1e-5) -> np.ndarray:
    """d function / d point, with one row for each coordinate of the point."""
    rows = []
    for index in range(point.size):
        shift = np.zeros(point.size)
        shift[index] = step
        rows.append((function(point + shift) - function(point - shift)) / (2.0 * step))
    return np.array(rows)


class Problem:
    """A small aggregated likelihood: two instruments, six events."""

    def __init__(self, baseline: SplineBaseline):
        self.quadrature = sp.step_quadrature(baseline, STEP)
        counts = sp.simulate(np.array([-2.0, -3.0]), GAMMA, baseline, 6, np.random.default_rng(10), step_seconds=STEP)
        self.y = counts.sum(axis=1).ravel()
        self.x = np.eye(2)[np.repeat(np.arange(2), self.quadrature.n_steps)]
        self.steps = np.tile(np.arange(self.quadrature.n_steps), 2)
        self.offset = np.full(self.y.size, np.log(6.0))
        self.penalty = baseline.penalty_matrix[1:, 1:]

    def parts(self, theta: np.ndarray):
        return sp._penalised(self.y, self.offset, self.x, self.steps, self.quadrature, self.penalty, theta[:2], theta[2:])

    def value(self, theta: np.ndarray) -> float:
        return self.parts(theta)[0]

    def derivatives(self, theta: np.ndarray, y: np.ndarray | None = None):
        _, mean, exposure, _ = self.parts(theta)
        return sp._score_and_information(self.y if y is None else y, self.x, self.steps, self.quadrature, self.penalty, theta[2:], mean, exposure)

    def score(self, theta: np.ndarray) -> np.ndarray:
        return self.derivatives(theta)[0]


@pytest.fixture(scope="module")
def problem(baseline) -> Problem:
    return Problem(baseline)


@pytest.fixture(scope="module")
def theta() -> np.ndarray:
    return np.concatenate((np.array([-1.9, -3.1]), GAMMA * 0.9))


def test_step_integrals_add_up_to_the_hawkes_integral(baseline):
    b = np.concatenate(([0.0], GAMMA))
    quadrature = sp.step_quadrature(baseline, STEP)
    assert quadrature.n_steps == 180
    assert quadrature.integrals(GAMMA).sum() == pytest.approx(baseline.integral_and_grad(b)[0], rel=1e-9)


@pytest.mark.parametrize("step", [0, 59, 60, 61, 179])
def test_each_step_integral_matches_the_hawkes_rate(baseline, step):
    b = np.concatenate(([0.0], GAMMA))
    grid = np.linspace(step * STEP + 1e-7, (step + 1) * STEP - 1e-7, 20001)
    numerical = float(np.trapezoid(baseline.rate(b, grid), grid))
    assert sp.step_quadrature(baseline, STEP).integrals(GAMMA)[step] == pytest.approx(numerical, rel=1e-6)


def test_the_rate_can_jump_at_the_release(baseline):
    b = np.concatenate(([0.0], GAMMA))
    before, after = baseline.rate(b, np.array([RELEASE - 1e-6, RELEASE + 1e-6]))
    assert after / before == pytest.approx(np.exp(b[baseline.n_left] - b[baseline.n_left - 1]), rel=1e-4)


def test_the_benchmark_baseline_is_the_hawkes_baseline(baseline):
    alpha = -1.2
    b = np.concatenate(([alpha], alpha + GAMMA))
    times = np.array([12.0, 900.0, 1799.0, 1800.0, 1900.0, 5000.0])
    np.testing.assert_allclose(alpha + baseline.design(times).toarray()[:, 1:] @ GAMMA, baseline.log_rate(b, times), rtol=1e-12)
    assert (np.exp(alpha) * sp.step_quadrature(baseline, STEP).integrals(GAMMA)).sum() == pytest.approx(baseline.integral_and_grad(b)[0], rel=1e-9)


def test_the_penalty_is_the_hawkes_penalty_and_ignores_the_level(baseline):
    alpha = 0.7
    b = np.concatenate(([alpha], alpha + GAMMA))
    assert baseline.penalty_and_grad(b)[0] == pytest.approx(0.5 * GAMMA @ baseline.penalty_matrix[1:, 1:] @ GAMMA, rel=1e-12)


def test_the_score_is_the_gradient_of_the_objective(problem, theta):
    np.testing.assert_allclose(problem.score(theta), central_difference(problem.value, theta), rtol=1e-5, atol=1e-3)


def test_the_observed_information_is_minus_the_derivative_of_the_score(problem, theta):
    observed = problem.derivatives(theta)[1]
    np.testing.assert_allclose(observed, -central_difference(problem.score, theta), rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(observed, observed.T, atol=1e-8)


def test_the_two_informations_agree_when_the_data_equal_the_mean(problem, theta):
    mean = problem.parts(theta)[1]
    _, observed, expected = problem.derivatives(theta, y=mean)
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-9)


def test_the_fit_is_the_maximum_of_the_penalised_likelihood(problem):
    result = sp.fit_with_baseline(problem.y, problem.x, problem.steps, problem.quadrature, problem.penalty, offset=problem.offset)
    assert result.converged
    best = problem.value(result.coefficients)
    for index in range(result.coefficients.size):
        shift = np.zeros(result.coefficients.size)
        shift[index] = 1e-3
        assert best >= problem.value(result.coefficients + shift) and best >= problem.value(result.coefficients - shift)


def test_grouped_fit_equals_fit_on_individual_rows(baseline):
    counts = sp.simulate(np.array([-2.0, -3.0]), GAMMA, baseline, 15, np.random.default_rng(1), step_seconds=STEP)
    grouped = sp.fit(counts, baseline, step_seconds=STEP)
    quadrature = sp.step_quadrature(baseline, STEP)
    instruments = np.repeat([0, 1], 15 * quadrature.n_steps)
    steps = np.tile(np.arange(quadrature.n_steps), 2 * 15)
    individual = sp.fit_with_baseline(counts.ravel(), np.eye(2)[instruments], steps, quadrature, baseline.penalty_matrix[1:, 1:])
    np.testing.assert_allclose(np.concatenate((grouped.alpha, grouped.gamma)), individual.coefficients, atol=1e-4)
    np.testing.assert_allclose(np.concatenate((grouped.alpha_se, grouped.gamma_se)), np.sqrt(np.diag(individual.covariance)), rtol=1e-3)


def test_fit_recovers_known_coefficients(baseline):
    alpha = np.array([-1.0, -2.0, -3.0])
    fit = sp.fit(sp.simulate(alpha, GAMMA, baseline, 33, np.random.default_rng(2), step_seconds=STEP), baseline, step_seconds=STEP)
    assert fit.fit.converged
    assert fit.alpha == pytest.approx(alpha, abs=0.1)
    assert fit.gamma == pytest.approx(GAMMA, abs=0.15)


def test_rate_is_the_level_times_the_step_integral(baseline):
    fit = sp.fit(sp.simulate(np.array([-1.0, -2.0]), GAMMA, baseline, 10, np.random.default_rng(3), step_seconds=STEP), baseline, step_seconds=STEP)
    expected = np.exp(fit.alpha[1]) * sp.step_quadrature(baseline, STEP).integrals(fit.gamma)[45]
    assert fit.rate(np.array([1]), np.array([45]))[0] == pytest.approx(expected, rel=1e-12)


def test_simulated_counts_have_the_model_mean(baseline):
    counts = sp.simulate(np.array([-1.0]), GAMMA, baseline, 4000, np.random.default_rng(4), step_seconds=STEP)
    assert counts.shape == (1, 4000, 180)
    np.testing.assert_allclose(counts.mean(axis=(0, 1)), np.exp(-1.0) * sp.step_quadrature(baseline, STEP).integrals(GAMMA), rtol=0.05)


def test_deviance_matches_the_formula_with_zero_counts():
    y = np.array([0.0, 1.0, 5.0])
    mu = np.array([1.5, 0.5, 4.0])
    assert sp.deviance(y, mu) == pytest.approx(2 * (1.5 + (np.log(2.0) - 0.5) + (5 * np.log(5 / 4) - 1.0)))


def test_shapes_that_do_not_fit_the_window_are_refused(baseline):
    with pytest.raises(ValueError):
        sp.step_quadrature(baseline, 31.0)
    with pytest.raises(ValueError):
        sp.fit(np.zeros((2, 3)), baseline)
    with pytest.raises(ValueError):
        sp.fit(np.zeros((1, 2, 10)), baseline)
