"""Checks needed for changing statistical weights and feedback support."""
import numpy as np
import pytest

from neck_localization.support_feedback import (WeightedAxisSplineProblem, FeedbackSettings,
    camera_weights, huber_weights, huber_cost, region_weights)
from neck_localization.joint_profile import JointSettings
from neck_localization.depth_profile import ProfileRejection
from test_joint_profile import analytic_surface


def test_equal_trust_does_not_silently_equalize_camera_total_weights():
    ids = np.r_[np.zeros(100, int), np.ones(50, int)]
    w = camera_weights(ids, 'equal_trust')
    np.testing.assert_allclose(w, 1.)
    balanced = camera_weights(ids, 'balanced')
    assert np.sum(balanced[ids == 0]) == pytest.approx(np.sum(balanced[ids == 1]))
    trusted = camera_weights(ids, 'equal_trust', [2, 1])
    assert trusted[0]/trusted[-1] == pytest.approx(2.)
    assert balanced.sum() == pytest.approx(w.sum())


def test_weighted_variable_projection_jacobian_on_nonuniform_weights():
    points, origin, normal = analytic_surface()
    p = WeightedAxisSplineProblem(points, origin, normal, np.arange(len(points)) % 3)
    weights = np.linspace(.01, 2., len(points))
    p.set_weights(weights)
    x = np.array([.7, -.3, .01, -.02])
    J = p.jac(x)
    numerical = np.column_stack([(p.fun(x+np.eye(4)[k]*h)-p.fun(x-np.eye(4)[k]*h))/(2*h)
        for k,h in enumerate([1e-4,1e-4,1e-6,1e-6])])
    np.testing.assert_allclose(J, numerical, rtol=1e-4, atol=1e-5)
    before = p.evaluate(x)['objective']
    p.set_weights(weights[::-1])
    assert abs(before-p.evaluate(x)['objective']) > 1e-4  # cache invalidation matters


def test_huber_irls_weights_match_objective_derivative():
    r = np.array([-3., -.2, 0., .03, .4])
    delta = .1
    numeric = (huber_cost(r+1e-6,delta)-huber_cost(r-1e-6,delta))/2e-6
    np.testing.assert_allclose(numeric, 2*huber_weights(r,delta)*r, atol=1e-9)


def test_region_refuses_constant_patch_with_unobserved_boundaries():
    s = np.repeat(np.linspace(-20,-5,40),32)
    phi = np.tile(np.linspace(0,2*np.pi,32,endpoint=False),40)
    points = np.c_[52*np.cos(phi),52*np.sin(phi),s]
    p = WeightedAxisSplineProblem(points,np.zeros(3),np.array([0.,0.,1.]),np.zeros(len(points),int))
    with pytest.raises(ProfileRejection, match='boundary'):
        region_weights(p,np.zeros(4),FeedbackSettings(),JointSettings())
