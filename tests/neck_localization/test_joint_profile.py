"""Variable-projection algebra and recovery checks; no rendering/GT in estimator."""
import numpy as np
import pytest
from scipy.optimize import least_squares

from neck_localization.geometry import basis
from neck_localization.joint_profile import AxisSplineProblem, JointSettings, supported_interval
from neck_localization.depth_profile import ProfileRejection


def analytic_surface():
    normal = np.array([.05, -.07, 1.]); normal /= np.linalg.norm(normal)
    U = np.array(basis(normal)).T
    s = np.repeat(np.linspace(-23, -5, 40), 32)
    phi = np.tile(np.linspace(-np.pi, np.pi, 32, endpoint=False), 40)
    radius = 52.+.04*(s+14)**2
    origin = np.array([2., -3., 600.])
    points = origin+s[:, None]*normal+radius[:, None]*(np.cos(phi)[:, None]*U[:, 0]+np.sin(phi)[:, None]*U[:, 1])
    return points, origin, normal


def test_reduced_jacobian_includes_profile_and_variable_rhs_derivatives():
    points, origin, normal = analytic_surface()
    problem = AxisSplineProblem(points, origin, normal, np.arange(len(points)) % 3)
    x = np.array([1.2, -.8, .03, -.02])
    analytic = problem.jac(x)
    steps = [1e-4, 1e-4, 1e-6, 1e-6]
    numerical = np.column_stack([(problem.fun(x+np.eye(4)[k]*step)-
                                  problem.fun(x-np.eye(4)[k]*step))/(2*step)
                                 for k, step in enumerate(steps)])
    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-5)
    # At each trial the profile, including its regularizer, is solved optimally.
    state = problem.evaluate(x)
    B = problem.basis(state['s'])
    A = np.vstack((problem.sqrt_weight[:, None]*B, problem.regularizer))
    np.testing.assert_allclose(A.T @ state['residual'], 0, atol=1e-8)


def test_wrong_initial_axis_can_be_corrected_without_true_axis_constraint():
    points, origin, normal = analytic_surface()
    wrong_normal = normal+np.array([.035, -.025, 0]); wrong_normal /= np.linalg.norm(wrong_normal)
    problem = AxisSplineProblem(points, origin+np.array([3., -2., 0]), wrong_normal,
                               np.arange(len(points)) % 3, JointSettings(smoothing_mm3=1e-6))
    fit = least_squares(problem.fun, np.zeros(4), jac=problem.jac,
                        x_scale=[5, 5, .1, .1], max_nfev=50)
    state = problem.evaluate(fit.x)
    assert fit.success
    assert np.linalg.norm(state['origin']-origin) < .02  # mm
    assert np.rad2deg(np.arccos(np.clip(state['normal'] @ normal, -1, 1))) < .02
    assert state['objective'] < problem.evaluate(np.zeros(4))['objective']*.001


def test_constant_radius_patch_without_interval_ends_is_rejected():
    s = np.repeat(np.linspace(-22, -5, 35), 32)
    phi = np.tile(np.linspace(0, 2*np.pi, 32, endpoint=False), 35)
    points = np.c_[52*np.cos(phi), 52*np.sin(phi), s]
    problem = AxisSplineProblem(points, np.zeros(3), np.array([0, 0, 1.]), np.zeros(len(points), int))
    state = problem.evaluate(np.zeros(4))
    with pytest.raises(ProfileRejection, match="search_boundary"):
        supported_interval(problem, state, points, JointSettings())
