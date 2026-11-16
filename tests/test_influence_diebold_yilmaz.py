"""Generalised FEVD and connectedness against hand-computable cases."""

import numpy as np
import pytest

from information_diffusion.influence import diebold_yilmaz as dy


def test_zero_var_with_identity_covariance_has_no_connectedness():
    theta = dy.generalised_fevd(np.zeros((1, 3, 3)), np.eye(3), horizon=5)
    np.testing.assert_allclose(theta, np.eye(3))
    assert dy.total_connectedness(theta) == 0.0


def test_contemporaneous_correlation_at_horizon_one():
    rho = 0.5
    theta = dy.generalised_fevd(np.zeros((1, 2, 2)), np.array([[1.0, rho], [rho, 1.0]]), horizon=1)
    expected_off = rho**2 / (1 + rho**2)
    np.testing.assert_allclose(theta, [[1 - expected_off, expected_off], [expected_off, 1 - expected_off]])
    assert dy.total_connectedness(theta) == pytest.approx(100 * expected_off)


def test_directional_spillover_with_two_step_horizon():
    a = 0.5
    theta = dy.generalised_fevd(np.array([[[0.0, 0.0], [a, 0.0]]]), np.eye(2), horizon=2)
    np.testing.assert_allclose(theta, [[1.0, 0.0], [a**2 / (1 + a**2), 1 / (1 + a**2)]])


def test_moving_average_matches_the_companion_power():
    a1 = np.array([[0.5, 0.1], [0.0, 0.3]])
    a2 = np.array([[0.1, 0.0], [0.2, 0.1]])
    psi = dy.moving_average(np.stack([a1, a2]), 4)
    np.testing.assert_allclose(psi[1], a1)
    np.testing.assert_allclose(psi[2], a1 @ a1 + a2)
    np.testing.assert_allclose(psi[3], a1 @ (a1 @ a1 + a2) + a2 @ a1)


def test_rows_sum_to_one():
    a = np.array([[[0.2, 0.1, 0.0], [0.0, 0.1, 0.05], [0.1, 0.0, 0.2]]])
    theta = dy.generalised_fevd(a, np.eye(3) + 0.2 * (np.ones((3, 3)) - np.eye(3)), horizon=8)
    np.testing.assert_allclose(theta.sum(axis=1), 1.0)


def test_common_scale_of_the_covariance_does_not_matter():
    a = np.array([[[0.3, 0.0], [0.2, 0.1]]])
    sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
    np.testing.assert_allclose(dy.generalised_fevd(a, sigma, 6), dy.generalised_fevd(a, 4.0 * sigma, 6))
